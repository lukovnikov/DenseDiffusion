from drawlib import *
from gligen.utils import seed_everything, _tokenize_annotated_prompt, segimg_to_masks, mask_to_bbox
import fire
import random
import re
import numpy as np
import traceback

from diffusers import StableDiffusionGLIGENPipeline
from dataset import COCOPanopticDataset2


def process_example_for_gligen(example, nobgr=False, augmentonly=True, upscale_to=512, device=torch.device("cpu")):
    caption = example.captions[0]
    
    caption = example.captions[0] if not augmentonly else ""
    
    extraexpressions = ["This image contains", "In this image are", "In this picture are", "This picture contains"]
    tojoin = []
    for rgbcode, seginfo in example.seg_info.items():
        tojoin.append(f"{{{seginfo['caption']}:{rgbcode}}}")
    caption += " " + random.choice(extraexpressions) + " " + ", ".join(tojoin) + "."
    
    promptpieces, regioncodes = _tokenize_annotated_prompt(caption)
    seg_img = example.load_seg_image()   #Image.open(self.panoptic_db[example_id]["segments_map"]).convert("RGB")
    
    if upscale_to is not None:
        upscalefactor = upscale_to / min(seg_img.size)
        newsize = [int(s * upscalefactor) for s in seg_img.size]
        seg_img = seg_img.resize(newsize, resample=Image.BOX)
        
    seg_imgtensor = torch.tensor(np.array(seg_img)).permute(2, 0, 1).to(device)
    
    cropsize = min((min(seg_imgtensor[0].shape) // 64) * 64, 512)
    crop = (random.randint(0, seg_imgtensor.shape[1] - cropsize), 
            random.randint(0, seg_imgtensor.shape[2] - cropsize))
    seg_imgtensor = seg_imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
    
    masks = segimg_to_masks(seg_imgtensor, regioncodes)
    # print(masks[0])
    bboxes = []
    bboxprompts = []
    for mask, maskprompt in masks:
        if torch.all(mask == 0):
            continue
        bbox = mask_to_bbox(mask)
        bboxes.append(bbox)
        bboxprompts.append(maskprompt)
    if nobgr:
        bboxes, bboxprompts = bboxes[:-1], bboxprompts[:-1]
    return "".join(promptpieces), bboxes, bboxprompts    


def run_example(example, numgen=5, gligen_tau=1., num_inference_steps=50, pipe=None, nobgr=False, device=torch.device("cpu")):
    caption, bboxes, prompts = process_example_for_gligen(example, nobgr=nobgr, device=device)
    rets = []
    for i in range(numgen):
        gen_image = pipe(
            prompt=caption,
            gligen_phrases=prompts,
            gligen_boxes=bboxes,
            gligen_scheduled_sampling_beta=gligen_tau,    # this is actually \tau from the paper
            output_type="pil",
            num_inference_steps=num_inference_steps,
        ).images[0]
        ret = {
            "caption": caption,
            "bboxes": bboxes,
            "bbox_captions": prompts,
            "seg_img": example.load_seg_image(),
            "image": gen_image,
        }
        rets.append(ret)
    return rets, (caption, bboxes, prompts)


class CASMode():
    def __init__(self, name):
        self.chunks = name.split("+")
        self.basename = self.chunks[0]
        self.threshold_lot = -1.
        
        self.localonlytil = False
        for chunk in self.chunks:
            m = re.match(r"lot(\d\.\d+)", chunk)
            if m:
                self.localonlytil = True
                self.threshold_lot = float(m.group(1))
                
        self._use_global_prompt_only = None
        self._augment_global_caption = None
        
    @property
    def name(self):
        return "+".join(self.chunks)
    
    @property
    def use_global_prompt_only(self):
        if self._use_global_prompt_only is not None:
            return self._use_global_prompt_only
        if self.localonlytil:
            return False
        if self.basename.startswith("posattn") or \
           self.basename.startswith("legacy") or \
           self.basename in ("cac", "dd", "global") :
            return True
        else:
            return False
        
    @use_global_prompt_only.setter
    def use_global_prompt_only(self, val:bool):
        self._use_global_prompt_only = val
        
    @property
    def augment_global_caption(self):
        if self._augment_global_caption is not None:
            return self._augment_global_caption
        if "keepprompt" in self.chunks or self.is_test:
            return False
        else:
            if self.name == "doublecross":
                return True
            # TODO
            return True
        
    @property
    def augment_only(self):
        return "augmentonly" in self.chunks or "augment_only" in self.chunks
        
    @augment_global_caption.setter
    def augment_global_caption(self, val:bool):
        self._augment_global_caption = val
        
    @property
    def is_test(self):
        return "test" in self.chunks
        
    @property
    def replace_layerids_with_encoder_layerids(self):
        if self.localonlytil:
            return False
        if self.use_global_prompt_only:
            return False
        else:
            return True
        return ("uselocal" in self.casmodechunks) or (self.casmode in ("cac", "posattn", "posattn2", "posattn3", "dd"))       # use local annotations on global prompt and discard local prompts
        
    def addchunk(self, chunk:str):
        self.chunks.append(chunk)
        return self
    
    def __add__(self, chunk:str):
        return self.addchunk(chunk)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other:str):
        return self.name.__eq__(other)
    
    def __hash__(self):
        return hash(self.name)
    
    
def main(
        outpath="gligen_outputs",
        datadir="/USERSPACE/lukovdg1/coco2017/",
        resultpath="",
        nobgr=False,
        seed=123456,
        device=0,
        tau=1.0,
    ):
    
    args = locals().copy()
    print(json.dumps(args, indent=4))
    
    device = torch.device("cuda", device)
    
    print(f"Doing tau={tau}")
    # Generate an image described by the prompt and
    # insert objects described by text at the region defined by bounding boxes
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16, safety_checker=None,
    )
    pipe = pipe.to(device)

    
    outpath = Path(outpath) / ("without_bgr" if nobgr else "with_bgr")

    seed_everything(seed)
    
    print(f"Doing path COCO 2017 val")
    if resultpath == "":
        i = 1
        while resultpath == "" or resultpath.exists():
            resultpath = outpath / f"tau={tau}" / ("coco2017val" + f"_{i}_out")
            i += 1
    else:
        resultpath = Path(resultpath)    
        
    resultpath.mkdir(parents=True, exist_ok=True)
    (resultpath / "inputs").mkdir(parents=True, exist_ok=True)
    print(f"Saving in : {resultpath}")

    cas = CASMode("cac+coco+augmentonly")
    
    valid_ds = COCOPanopticDataset2(maindir=datadir, split="valid", casmode=cas, simpleencode=False,
                    mergeregions=False, limitpadding=False,
                    max_masks=100, min_masks=1, min_size=128, upscale_to=512)

    for j, inpexample in enumerate(iter(valid_ds)):
        savepath = resultpath / f"{inpexample.image_path.stem}.png"
        if savepath.exists():
            print(f"{savepath} exists")
            continue
        try:
            outputs, inputs = run_example(inpexample, numgen=1, pipe=pipe, gligen_tau=tau, nobgr=nobgr, device=device)
            imgs = [output["image"] for output in outputs]
            assert len(imgs) == 1
            img = imgs[0]
            while savepath.exists():
                k = 1
                resultpath / f"{inpexample.image_path.stem}_{k}.png"
                k += 1
            # imgs = [display_bboxes(output["bboxes"], _img=img) for img, output in zip(imgs, outputs)]
            DrawImage(img, imgsize=512).drawself().save(savepath)
            with open(resultpath / "inputs" / f"{inpexample.image_path.stem}.json", "w") as outf:
                json.dump(inputs, outf, indent=4)
        except Exception as e:
            print(f"Example {j} at {savepath} failed.")
            print(traceback.print_exc())


if __name__ == "__main__":
    fire.Fire(main)