import re
import numpy as np
import torch
from PIL import Image, ImageDraw


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def region_code_to_rgb(rcode):
    B = rcode // 256**2
    rcode = rcode % 256**2
    G = rcode // 256
    R = rcode % 256
    return (R, G, B)


def _tokenize_annotated_prompt(prompt):    # gets prompt with localized rgbcode annotations and returns prompt and rgb-local prompt map
    prompt = re.split(r"(\{[^\}]+\})", prompt)
    _prompt = []
    _layer_id = []
    for e in prompt:
        m = re.match(r"\{(.+):(\d+)\}", e)
        if m:
            _prompt.append(m.group(1))
            _layer_id.append(int(m.group(2)))
        else:
            _prompt.append(e)
            _layer_id.append(0)
    # print(_prompt)
    # print(_layer_id)
    ret_prompt = "".join(_prompt)
    ret_rgb_map = {region_code_to_rgb(regioncode):subprompt for regioncode, subprompt in zip(_layer_id, _prompt) if regioncode != 0}
    return ret_prompt, ret_rgb_map


def segimg_to_masks(segimg, rgbmap):
    masks = []
    for rgb, prompt in rgbmap.items():
        mask = segimg == torch.tensor(rgb, dtype=segimg.dtype, device=segimg.device)[:, None, None]
        mask = mask.all(0)
        masks.append((mask, prompt))
    return masks


def mask_to_bbox(mask):    # takes a mask and return a tight bbox around the non-zero elements of given mask
    width, height = mask.shape
    topleft = mask.nonzero().min(0)[0]
    bottomright = mask.nonzero().max(0)[0]
    bbox = list(topleft.cpu().numpy())[::-1] + list(bottomright.cpu().numpy())[::-1]
    bbox = (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height)
    # print(bbox)
    return bbox


def display_bboxes(bboxes, prompts=None, size=(512, 512), _img=None, linewidth=3):
    img = Image.new(mode="RGB", size=size) if _img is None else _img.copy()
    width, height = img.size
    imdraw = ImageDraw.Draw(img)
    if prompts is None:
        prompts = [None] * len(bboxes)
    for bbox, prompt in zip(bboxes, prompts):
        imdraw.rectangle([(int(bbox[0] * width), int(bbox[1] * height)), (int(bbox[2] * width), int(bbox[3] * height))], outline=(255, 0, 0), width=linewidth)
        if prompt is not None:
            imdraw.text((int(bbox[0] * width), int(bbox[1] * height)), prompt, font_size=32)
    return img


def process_example_for_gligen(example, nobgr=False):
    assert len(example.captions) == 1
    caption = example.captions[0]
    promptpieces, regioncodes = _tokenize_annotated_prompt(caption)
    seg_img = example.load_seg_image()   #Image.open(self.panoptic_db[example_id]["segments_map"]).convert("RGB")
    seg_imgtensor = torch.tensor(np.array(seg_img)).permute(2, 0, 1)
    masks = segimg_to_masks(seg_imgtensor, regioncodes)
    # print(masks[0])
    bboxes = []
    bboxprompts = []
    for mask, maskprompt in masks:
        bbox = mask_to_bbox(mask)
        bboxes.append(bbox)
        bboxprompts.append(maskprompt)
    if nobgr:
        bboxes, bboxprompts = bboxes[:-1], bboxprompts[:-1]
    return "".join(promptpieces), bboxes, bboxprompts    


def run_example(example, numgen=5, gligen_tau=1., num_inference_steps=50, pipe=None, nobgr=False):
    caption, bboxes, prompts = process_example_for_gligen(example, nobgr=nobgr)
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
    return rets