# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

from ...models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from ...schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from ...utils import PIL_INTERPOLATION, logging
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from . import FlaxStableDiffusionPipelineOutput
from .safety_checker_flax import FlaxStableDiffusionSafetyChecker

import contextlib
import time
import alpa

@contextlib.contextmanager
def timer(name: str):
    begin = time.time_ns()
    try:
        yield begin
    finally:
        print(f'Timer {name}[ms] {(time.time_ns() - begin) / int(1e6)}')    

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False


class FlaxStableDiffusionImg2ImgPipeline(FlaxDiffusionPipeline):
    r"""
    Pipeline for image-to-image generation using Stable Diffusion.

    This model inherits from [`FlaxDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`FlaxAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`FlaxCLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.FlaxCLIPTextModel),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`FlaxUNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`], or
            [`FlaxDPMSolverMultistepScheduler`].
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        if safety_checker is None:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_inputs(self, prompt: Union[str, List[str]], image: Union[Image.Image, List[Image.Image]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if not isinstance(image, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        if isinstance(image, Image.Image):
            image = [image]

        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids, processed_images

    def get_timestep_start(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)

        return t_start

    def loop_body(self, step, args):
        latents, scheduler_state, params, context, guidance_scale = args
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        latents_input = jnp.concatenate([latents] * 2)

        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])

        latents_input = self.scheduler.scale_model_input(scheduler_state, latents_input, t)

        # predict the noise residual
        noise_pred = self.unet.apply(
            {"params": params["unet"]},
            jnp.array(latents_input),
            jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context,
        ).sample
        # perform guidance
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        return latents, scheduler_state, params, context, guidance_scale


    def _generate(
        self,
        prompt_ids: jnp.array,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        start_timestep: int,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        noise: Optional[jnp.array] = None,
        neg_prompt_ids: Optional[jnp.array] = None,
        ray_enabled: bool = False,
    ):
        def _wrap_outside_a_text_encoder(prompt_ids):
            state = {
                "text_encoder": params["text_encoder"]
            }
            batch = {
                "prompt_ids": prompt_ids
            }
            static_args = {
                "pipeline": self
            }
            return jnp.array(_outside_a_text_encoder(state, batch, static_args))
        
        def _wrap_outside_a_loop_body(step, args):
            latents, scheduler_state, params, context, guidance_scale = args
            state = {
                "step": step,
                "unet": params["unet"],
                "scheduler_state": scheduler_state,
                "context": context,
                "guidance_scale": guidance_scale
            }
            batch = {
                "latents":latents
            }
            static_args = {
                "pipeline": self,
            }
            ret = _outside_a_loop_body(state, batch, static_args)
            return ret

        def _wrap_outside_a_vae_apply(latents, method_is_encode:bool):
            state = {
                "vae": params["vae"],
                "prng_seed": prng_seed
            }
            batch = {
                "latents": latents
            }
            static_args = {
                "pipeline": self
            }
            _outside_a_vae_apply = _outside_a_vae_encode_apply if method_is_encode else _outside_a_vae_decode_apply

            return jnp.array(_outside_a_vae_apply(state, batch, static_args))

        # Before main loop

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_embeddings = _wrap_outside_a_text_encoder(prompt_ids)

        batch_size = prompt_ids.shape[0]

        max_length = prompt_ids.shape[-1]

        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            ).input_ids
        else:
            uncond_input = neg_prompt_ids
    
        uncond_embeddings = _wrap_outside_a_text_encoder(uncond_input)
    
        context = jnp.concatenate([uncond_embeddings, text_embeddings])

        latents_shape = (
            batch_size,
            self.unet.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if noise is None:
            noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if noise.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {noise.shape}, expected {latents_shape}")
    
        if ray_enabled:
            init_latents_dist = self.vae.apply({"params": params["vae"]}, image, method=self.vae.encode).latent_dist
            init_latents = init_latents_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
        else:
            init_latents = _wrap_outside_a_vae_apply(image, True)

        init_latents = 0.18215 * init_latents

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
        )

        latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)

        latents = self.scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        args = latents, scheduler_state, params, context, guidance_scale

        
        # main loop
        for i in range(start_timestep, num_inference_steps):
            args = _wrap_outside_a_loop_body(i, args)
        latents = jnp.array(args[0])
          
        # after main loop
        latents = 1 / 0.18215 * latents
        if ray_enabled:
            image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample
        else:
            image = _wrap_outside_a_vae_apply(latents, False)        
        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        
        return image

    def __call__(
        self,
        prompt_ids: jnp.array,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, jnp.array] = 7.5,
        noise: jnp.array = None,
        neg_prompt_ids: jnp.array = None,
        return_dict: bool = True,
        jit: bool = False,
        ray_enabled: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt_ids (`jnp.array`):
                The prompt or prompts to guide the image generation.
            image (`jnp.array`):
                Array representing an image batch, that will be used as the starting point for the process.
            params (`Dict` or `FrozenDict`): Dictionary containing the model parameters/weights
            prng_seed (`jax.random.KeyArray` or `jax.Array`): Array containing random number generator key
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            noise (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. tensor will ge generated
                by sampling using the supplied random `generator`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions. NOTE: This argument
                exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a future release.

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]

        start_timestep = self.get_timestep_start(num_inference_steps, strength)

        images = self._generate(
            prompt_ids,
            image,
            params,
            prng_seed,
            start_timestep,
            num_inference_steps,
            height,
            width,
            guidance_scale,
            noise,
            neg_prompt_ids,
            ray_enabled
        )

        images = np.asarray(images)
        has_nsfw_concept = False

        if not return_dict:
            return (images, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)


def preprocess(image, dtype):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2.0 * image - 1.0

@partial(alpa.parallelize, static_argnums=(2,))
def _outside_a_text_encoder(state, batch, static_args):
    pipeline = static_args["pipeline"]
    prompt_ids = batch["prompt_ids"]

    return pipeline.text_encoder(
        prompt_ids,
        params=state["text_encoder"]
    )[0]

@partial(alpa.parallelize, static_argnums=(2,))
def _outside_a_loop_body(state, batch, static_args):
    pipeline = static_args["pipeline"]
    return pipeline.loop_body(
        state["step"],
        (
            batch["latents"],
            state["scheduler_state"],
            {"unet": state["unet"]},
            state["context"],
            state["guidance_scale"]
        )
    )

@partial(alpa.parallelize, static_argnums=(2,))
def _outside_a_vae_encode_apply(state, batch, static_args):
    pipeline = static_args["pipeline"]
    latents = batch["latents"]
    prng_seed = state["prng_seed"]

    ret_lat = pipeline.vae.apply(
        {"params":state["vae"]},
        latents, 
        method=pipeline.vae.encode
    ).latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))

    return ret_lat

@partial(alpa.parallelize, static_argnums=(2,))
def _outside_a_vae_decode_apply(state, batch, static_args):
    pipeline = static_args["pipeline"]
    latents = batch["latents"]

    ret_sample = pipeline.vae.apply(
        {"params":state["vae"]},
        latents, 
        method=pipeline.vae.decode
    ).sample

    return ret_sample
