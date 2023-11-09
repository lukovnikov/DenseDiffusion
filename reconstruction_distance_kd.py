import torch
import requests
import PIL
from PIL import Image
import numpy as np

from typing import Union, List, Tuple, Dict, Callable, Optional, Any

from transformers import BlipForConditionalGeneration, BlipProcessor

import diffusers
from diffusers import DDIMScheduler, DDIMInverseScheduler, KandinskyPipeline, KandinskyPriorPipeline, ImagePipelineOutput

from diffusers.utils import BaseOutput

from torchvision.transforms.functional import pil_to_tensor

from dataclasses import dataclass
from clip_interrogator import Config, Interrogator


class BLIPCaptioner:

    def __init__(self, captioner_ckpt = "Salesforce/blip-image-captioning-large"):
        self.captioner_ckpt = captioner_ckpt
        self.caption_processor = BlipProcessor.from_pretrained(self.captioner_ckpt)
        self.caption_generator = BlipForConditionalGeneration.from_pretrained(self.captioner_ckpt, low_cpu_mem_usage=True)
        self._execution_device = torch.device("cuda")

    @torch.no_grad()
    def generate_caption(self, image):
        """Generates caption for a given image."""
        text = ""
    
        prev_device = self.caption_generator.device
    
        device = self._execution_device
        inputs = self.caption_processor(image, text, return_tensors="pt").to(
            device=device, dtype=self.caption_generator.dtype
        )
        self.caption_generator.to(device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)
    
        # offload caption generator
        self.caption_generator.to(prev_device)
    
        caption = self.caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption


class CLIPInterrogator:
    def __init__(self, clip_model_name="ViT-L-14/openai"):
        self.interrogator = Interrogator(Config(clip_model_name=clip_model_name))

    @torch.no_grad()
    def generate_caption(self, image):
        return self.interrogator.interrogate(image)
    
    
@dataclass
class InversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    latents: torch.FloatTensor
    images: Union[List[PIL.Image.Image], np.ndarray]


def get_new_h_w(h, w, scale_factor=8):
    new_h = h // scale_factor**2
    if h % scale_factor**2 != 0:
        new_h += 1
    new_w = w // scale_factor**2
    if w % scale_factor**2 != 0:
        new_w += 1
    return new_h * scale_factor, new_w * scale_factor


class KandinskyPipelinePartialInversion(KandinskyPipeline):
    silent = False
    
    @torch.no_grad()
    def denoise(
        self,
        prompt: Union[str, List[str]],
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        denoise_from: int = 0,      # which of the selected timesteps to start denoising from, by default from the very beginning (x_T)
        denoise_steps: int = None,   # how many steps to run the denoising for, if None, the until the very end (x_0)
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_latent: bool = False,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image_embeds (`torch.FloatTensor` or `List[torch.FloatTensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            negative_image_embeds (`torch.FloatTensor` or `List[torch.FloatTensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, _ = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=prompt_embeds.dtype, device=device
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_tensor = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels

        height, width = get_new_h_w(height, width, self.movq_scale_factor)

        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_encoder_hidden_states.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        
        # print(f"Normal timesteps: {timesteps}")
        denoise_steps = denoise_steps if denoise_steps is not None else num_inference_steps - denoise_from
        timesteps_tensor = timesteps_tensor[denoise_from:denoise_from + denoise_steps]
        if not self.silent:
            print(f"Selected timesteps: {timesteps_tensor}")

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            added_cond_kwargs = {"text_embeds": prompt_embeds, "image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            ).prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # post-processing
        image = self.movq.decode(latents, force_not_quantize=True)["sample"]
        numimg = image.shape[0]
        image = self.image_postprocess(image, output_type=output_type)

        if return_latent:
            if not return_dict:
                ret = (latents, image, [False] * numimg)
            else:
                ret = InversionPipelineOutput(latents=latents, images=image)
        else:
            if not return_dict:
                ret = (image, [False] * numimg)
            else:
                ret = ImagePipelineOutput(images=image)

        return ret
    
    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.movq.encode(image).latents

            # latents = self.movq.config.scaling_factor * latents     # do we still need this? --> NO

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                print("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents
    
    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        """
        Convert a PIL image or a list of PIL images to NumPy arrays.
        """
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images):
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Converts an image to RGB format.
        """
        image = image.convert("RGB")
        return image
    
    def image_preprocess(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)
        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            image = [self.convert_to_rgb(i) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = self.numpy_to_pt(image)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)
            _, channel, height, width = image.shape

            # don't need any preprocess if the image is latents
            if channel == 4:
                return image

        # expected range [0,1], normalize to [-1,1]
        do_normalize = True
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        return image
    
    
    def image_postprocess(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        output_type=None,
    ) -> torch.Tensor:
        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")

        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image
    
    @torch.no_grad()
    def invert(
        self,
        prompt: Union[str, List[str]],
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ],
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        invert_steps: Tuple[int] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image_embeds (`torch.FloatTensor` or `List[torch.FloatTensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            negative_image_embeds (`torch.FloatTensor` or `List[torch.FloatTensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, _ = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        image = self.image_preprocess(image)
        
        # 4. Prepare latent variables
        latents = self.prepare_image_latents(image, batch_size, self.movq.dtype, device, generator)
        
        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=prompt_embeds.dtype, device=device
            )

        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_tensor = self.inverse_scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels

        # # create initial latent
        # latents = self.prepare_latents(
        #     (batch_size, num_channels_latents, height, width),
        #     text_encoder_hidden_states.dtype,
        #     device,
        #     generator,
        #     latents,
        #     self.scheduler,
        # )
        
        # 7. Denoising loop where we obtain the cross-attention maps.
        return_single = False
        if invert_steps is None:
            invert_steps = (num_inference_steps,)
        elif isinstance(invert_steps, int):
            invert_steps = (invert_steps,)
            return_single = True
            
        invert_from = 0
        # print(f"Normal timesteps: {timesteps}")
        timesteps_tensor = timesteps_tensor[invert_from:invert_from + max(invert_steps)]
        if not self.silent:
            print(f"Selected timesteps: {timesteps_tensor}")
        return_timesteps = [timesteps_tensor[i-1] for i in invert_steps]
        if not self.silent:
            print(f"Return timesteps: {return_timesteps}")

        ret_latents = {}

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            added_cond_kwargs = {"text_embeds": prompt_embeds, "image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inverse_scheduler.step(
                noise_pred,
                t,
                latents
            ).prev_sample
            
            if t in return_timesteps:
                ret_latents[t.cpu().item()] = latents

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
                
        rets = []
        # print(ret_latents.keys())
        for select_t in return_timesteps:
            latents = ret_latents[select_t.cpu().item()]
            inverted_latents = latents.detach().clone()

            # 8. Post-processing
            image = self.movq.decode(latents, force_not_quantize=True)["sample"]
            image = self.image_postprocess(image, output_type=output_type)
            # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

            if not return_dict:
                rets.append((inverted_latents, image))
            else:
                rets.append(InversionPipelineOutput(latents=inverted_latents, images=image))
                
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if len(rets) == 1 and return_single:
            return rets[0]
        else:
            return rets
        # # post-processing
        # image = self.movq.decode(latents, force_not_quantize=True)["sample"]

        # if output_type not in ["pt", "np", "pil"]:
        #     raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")

        # if output_type in ["np", "pil"]:
        #     image = image * 0.5 + 0.5
        #     image = image.clamp(0, 1)
        #     image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        # if not return_dict:
        #     return (image,)

        # return ImagePipelineOutput(images=image)
    
    def compute_reconstruction(self, x0 : PIL.Image.Image, 
                               reconstruction_steps : int, 
                               prompt : str = None, 
                               num_inference_steps=50):
        if prompt is None:
            prompt = self.generate_caption(x0)
        priorout = self.priorpipeline(prompt)
                
        x_inv = self.invert(prompt, x0, 
                            image_embeds=priorout.image_embeds, negative_image_embeds=priorout.negative_image_embeds,
                            invert_steps=reconstruction_steps, num_inference_steps=num_inference_steps).latents
        x_recon = self.denoise(prompt, latents=x_inv, 
                               image_embeds=priorout.image_embeds, negative_image_embeds=priorout.negative_image_embeds,
                               denoise_from=num_inference_steps-reconstruction_steps, num_inference_steps=num_inference_steps).images
        return x_recon[0]

    def compute_reconstruction_distance(self, x0 : PIL.Image.Image, 
                                        reconstruction_steps : int, 
                                        prompt : str = None, 
                                        num_inference_steps=50, distance="l2"):
        xrecon = self.compute_reconstruction(x0, reconstruction_steps, prompt=prompt, num_inference_steps=num_inference_steps)
        if distance == "l2":
            x0_pt = pil_to_tensor(x0).float()/127.5-1
            xrecon_pt = pil_to_tensor(xrecon).float()/127.5-1
            x0_pt = x0_pt.flatten()[None, None]
            xrecon_pt = xrecon_pt.flatten()[None, None]
            dist = torch.cdist(x0_pt, xrecon_pt, p=2)[0, 0]
        else:
            raise Exception("Unsupported distance")
        return dist

    def compute_stepwise_reconstruction_distance(self, x0 : PIL.Image.Image, 
                                               reconstruction_steps : int, 
                                               prompt : str = None,
                                               extra_steps : int = 1, 
                                               num_inference_steps=50, use_latent=True, distance="l2"):
        """ reconstruction_steps specifies how many inference steps to go back to obtain x-tilde from paper, 
            extra_steps specifies how many inference steps to go back and forth to obtain a reconstruction of x-tilde 
            note that inference steps skip over multiple original DDPM training steps, rather than the original DDPM steps used in training.
        """
        if prompt is None:
            prompt = self.generate_caption(x0)
        priorout = self.priorpipeline(prompt)
        ret = self.invert(prompt, x0, 
                          image_embeds=priorout.image_embeds, negative_image_embeds=priorout.negative_image_embeds,
                          invert_steps=(reconstruction_steps, reconstruction_steps+extra_steps), num_inference_steps=num_inference_steps)
        x_inv, x_inv_extra = ret
        x_extra_recon = self.denoise(prompt, latents=x_inv_extra.latents, 
                                     image_embeds=priorout.image_embeds, negative_image_embeds=priorout.negative_image_embeds,
                                     denoise_from=num_inference_steps-reconstruction_steps-extra_steps, 
                                     denoise_steps=extra_steps,
                                     num_inference_steps=num_inference_steps, return_latent=True)
        # print(x_inv.latents.shape, x_extra_recon.shape, x_inv.latents.min(), x_inv.latents.max(), x_inv.latents.mean())
        if distance == "l2":
            if use_latent:
                x_inv = x_inv.latents[0]
                x_extra_recon = x_extra_recon.latents[0]
                x_inv = x_inv.flatten()[None, None]
                x_extra_recon = x_extra_recon.flatten()[None, None]
                dist = torch.cdist(x_inv, x_extra_recon, p=2)[0, 0]
            else:
                x0_pt = pil_to_tensor(x_inv.images[0]).float()/127.5-1
                xrecon_pt = pil_to_tensor(x_extra_recon.images[0]).float()/127.5-1
                x0_pt = x0_pt.flatten()[None, None]
                xrecon_pt = xrecon_pt.flatten()[None, None]
                dist = torch.cdist(x0_pt, xrecon_pt, p=2)[0, 0]
        else:
            raise Exception("Unsupported distance")
        return dist
    
    def generate_caption(self, image):
        return self.captioner.generate_caption(image)
        
        
def compute_diff(img1, img2):
    x_diff = PIL.ImageChops.subtract(img1, img2, offset=127, scale=1)
    return x_diff


def create_pipeline(kd_model_ckpt="kandinsky-community/kandinsky-2-1", 
                    blip_ckpt="Salesforce/blip-image-captioning-large", 
                    clip_interrogate_ckpt="ViT-L-14/openai", 
                    use_blip_only=False):
    if use_blip_only:
        captioner = BLIPCaptioner(captioner_ckpt=blip_ckpt)
    else:
        captioner = CLIPInterrogator(clip_model_name=clip_interrogate_ckpt)
    
    pipeline = KandinskyPipelinePartialInversion.from_pretrained(kd_model_ckpt)
    priorpipeline = KandinskyPriorPipeline.from_pretrained(kd_model_ckpt + "-prior")

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    
    pipeline.priorpipeline = priorpipeline
    
    pipeline.enable_model_cpu_offload()
    
    pipeline.captioner = captioner
    return pipeline


# EXAMPLE USAGE:
def main(device=0):
    
    """ - 'reconstruction_steps' specifies how many inference steps to go back to obtain x-tilde from paper, 
        - 'extra_steps' specifies how many inference steps to go back and forth to obtain a reconstruction of x-tilde 
        - note that inference steps skip over multiple original DDPM training steps, rather than the original DDPM steps used in training.
        - note that only single images are supported for now
    """
    print(f"Diffusers version: {diffusers.__version__}")
    device = torch.device("cuda", device)
    
    # create pipeline, set 'use_blip_only' to False (default) to use CLIP Interrogator (this is slower than BLIP)
    pipeline = create_pipeline(use_blip_only=True)
    pipeline.to(device)
    pipeline.priorpipeline.to(device)

    # generate a caption for an image
    img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"
    img_url = "https://image.geo.de/30143964/t/cC/v3/w1440/r0/-/mona-lisa-p-1024727412-jpg--81961-.jpg"
    rawimage = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
    print(pipeline.generate_caption(rawimage))

    # generate an image using a prompt
    prompt = "a photo of a banana taped to a wall with black duct tape"
    priorout = pipeline.priorpipeline(prompt)
    x0 = pipeline.denoise(prompt, image_embeds=priorout.image_embeds, negative_image_embeds=priorout.negative_image_embeds).images[0]
    
    # compute reconstruction by going 5 steps backwards using the inverse DDIM scheduler 
    # and then denoising 5 steps using the regular DDIM scheduler
    xrecon = pipeline.compute_reconstruction(x0, 5, prompt=prompt, num_inference_steps=50)
    xrecon2 = pipeline.compute_reconstruction(x0, 25, prompt=prompt, num_inference_steps=50)
    
    # compute reconstruction distance with the reconstruction computed as in the previous line
    recondist = pipeline.compute_reconstruction_distance(x0, 5, prompt=prompt, num_inference_steps=50)
    print(f"Reconstruction distance with prompt: {recondist}")
    # if prompt is not specified, a prompt is internally computed using the set captioning method
    recondist = pipeline.compute_reconstruction_distance(x0, 5, num_inference_steps=50)
    print(f"Reconstruction distance without prompt: {recondist}")
    
    # compute latent reconstruction distance by taking 5 inverse DDIM steps using the inverse DDIM scheduler (x-tilde),
    # then taking one more inverse DDIM step
    # and then denoising one step using the regular DDIM scheduler (x-tilde reconstructed)
    # the distance is computed between the latents x-tilde and its reconstruction
    latentrecondist = pipeline.compute_stepwise_reconstruction_distance(x0, 5, prompt=prompt, num_inference_steps=50)
    print(f"Stepwise reconstruction distance with prompt (in latent space): {latentrecondist}")

    # to compute step-wise reconstruction distance in pixel space, set 'use_latent' to False (not sure how much sense this makes though)    
    latentrecondist = pipeline.compute_stepwise_reconstruction_distance(x0, 5, prompt=prompt, num_inference_steps=50, use_latent=False)
    print(f"Stepwise reconstruction distance with prompt (in pixel space): {latentrecondist}")
    
    print("Done.")

    
# EXAMPLE USAGE:
def main_recons(device=3):
    
    """ - 'reconstruction_steps' specifies how many inference steps to go back to obtain x-tilde from paper, 
        - 'extra_steps' specifies how many inference steps to go back and forth to obtain a reconstruction of x-tilde 
        - note that inference steps skip over multiple original DDPM training steps, rather than the original DDPM steps used in training.
        - note that only single images are supported for now
    """
    print(f"Diffusers version: {diffusers.__version__}")
    device = torch.device("cuda", device)
    
    # create pipeline, set 'use_blip_only' to False (default) to use CLIP Interrogator (this is slower than BLIP)
    pipeline, priorpipeline = create_pipeline(use_blip_only=True)
    pipeline.to(device)
    priorpipeline.to(device)

    # generate a caption for an image
    img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"
    img_url = "https://image.geo.de/30143964/t/cC/v3/w1440/r0/-/mona-lisa-p-1024727412-jpg--81961-.jpg"
    rawimage = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
    print(pipeline.generate_caption(rawimage))

    # generate an image using a prompt
    prompt = "a photo of a banana taped to a wall with black duct tape"
    priorout = priorpipeline(prompt)
    x_inv = pipeline.invert(prompt, rawimage, 
                            image_embeds=priorout.image_embeds, 
                            negative_image_embeds=priorout.negative_image_embeds,
                            invert_steps=5).images[0]
    print ("done")
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)