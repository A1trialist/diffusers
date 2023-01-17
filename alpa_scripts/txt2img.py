import alpa
import ray
import jax
import numpy as np
import contextlib
import time
import contextlib
import time

from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline
from functools import partial

@contextlib.contextmanager
def timer(name: str):
    begin = time.time_ns()
    try:
        yield begin
    finally:
        print(f'Timer {name}[ms] {(time.time_ns() - begin) / int(1e6)}')  

def run_alpa(pipeline, params, prompt):
    num_samples = 1
    prompt = [prompt]*num_samples
    prompt_ids = pipeline.prepare_inputs(prompt)
    
    prng_seed = jax.random.PRNGKey(0)
    local_prng_seed = jax.random.split(prng_seed, 1)
    
    num_inference_steps = 50
    
    images = pipeline(prompt_ids, params, local_prng_seed, num_inference_steps, jit=False, ray_enabled=ray_enabled).images
    images = pipeline.numpy_to_pil(np.asarray(images.reshape((1,) + images.shape[-3:])))
    images[0].save("/data/wly/po.png")

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

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="bf16", dtype=jax.numpy.bfloat16
)

prompt = "a photo of an astronaut riding a horse on mars"

run_alpa(pipeline, prompt)
run_alpa(pipeline, prompt)
