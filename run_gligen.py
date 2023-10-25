from drawlib import *
from gligen.utils import *
import fire

from diffusers import StableDiffusionGLIGENPipeline


def main(paths = [
            "/USERSPACE/lukovdg1/controlnet11/evaldata/extradev.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/threeorange1.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/threeballs1.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/threecoloredballs1.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/threesimplefruits1.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/openair1.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/foursquares2.pkl",
        ],
        nobgr=False,
        outpath="gligen_outputs/without_bgr",
        seed=123456,
        device=0,
        taus=[1.0, 0., 0.1, 0.3, 0.5],
    ):
    
    args = locals().copy()
    print(json.dumps(args, indent=4))
    
    for tau in taus:
        print(f"Doing tau={tau}")
        # Generate an image described by the prompt and
        # insert objects described by text at the region defined by bounding boxes
        pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16, safety_checker=None,
        )
        pipe = pipe.to(torch.device("cuda", device))

        
        outpath = Path(outpath)

        for path in [Path(p) for p in paths]:
            seed_everything(seed)
            
            i = 1
            print(f"Doing path: {path}")
            resultpath = None
            while resultpath is None or resultpath.exists():
                resultpath = outpath / f"tau={tau}" / (path.stem + f"_{i}_out")
                i += 1
                
            resultpath.mkdir(parents=True, exist_ok=True)
            (resultpath / "inputs").mkdir(parents=True, exist_ok=True)
            print(f"Saving in : {resultpath}")

            with open(path, "rb") as f:
                inpexamples = pkl.load(f)

            allout = []
            for j, inpexample in enumerate(inpexamples):
                outputs = run_example(inpexample, pipe=pipe, gligen_tau=tau, nobgr=nobgr)
                allout.append(outputs)
                imgs = [output["image"] for output in outputs]
                imgs = [display_bboxes(output["bboxes"], _img=img) for img, output in zip(imgs, outputs)]
                draw = DrawRow(*[DrawImage(img) for img in imgs])
                draw.drawself().save(resultpath / f"{j}.png")
                with open(resultpath / "inputs" / f"{j}.inputs.json", "w") as outf:
                    json.dump(process_example_for_gligen(inpexample), outf, indent=4)
                
            with open(resultpath / "outbatches.pkl", "wb") as outf:
                pkl.dump(allout, outf)


if __name__ == "__main__":
    fire.Fire(main)