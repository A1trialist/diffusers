import alpa
import ray
import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

def run_alpa(pipeline, prompts, init_img, ray_enabled):

    rng = create_key(0)
    
    num_samples = 1
    rng = jax.random.split(rng, num_samples)
    prompt_ids, processed_image = pipeline.prepare_inputs(prompt=[prompts]*num_samples, image = [init_img]*num_samples)

    output = pipeline(
        prompt_ids=prompt_ids, 
        image=processed_image, 
        params=params, 
        prng_seed=rng, 
        strength=0.75, 
        num_inference_steps=50, 
        jit=False, 
        height=512,
        width=768,
        ray_enabled=ray_enabled)

    output_images = pipeline.numpy_to_pil(np.asarray(output.images.reshape((num_samples,) + output.images.shape[-3:])))

    output_images[0].save("/data/wly/po.png")

ray_enabled = True

if ray_enabled:
    ray.init()
    alpa.init(cluster="ray")
    print("Ray enabled")
else:
    num_devices = jax.device_count()
    print(f"Found {num_devices} JAX devices:")
    for device in jax.devices():
        print(device.device_kind)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))

prompts = "A fantasy landscape, trending on artstation"

pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="flax",
    dtype=jnp.bfloat16,
)

run_alpa(pipeline, prompts, init_img, ray_enabled)
run_alpa(pipeline, prompts, init_img, ray_enabled)
