from diffusers import DiffusionPipeline
import torch

model_path = "stabilityai/stable-diffusion-xl-base-1.0" # <-- change this
pipe = DiffusionPipeline.from_pretrained(model_path, variant = "fp16",torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "A naruto with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("naruto.png")