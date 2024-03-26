from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
import json

base_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "/home/junseok/model/controlnet_llava"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker = None
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

image_list = os.listdir('/home/junseok/dataset/test/images')
conditioning_list = os.listdir('/home/junseok/dataset/test/conditioning_images')
# JSONL 파일 열기
with open('/home/junseok/dataset/test/metadata.jsonl', 'r') as f:
    # 각 라인을 읽어서 리스트에 저장
    lines = f.readlines()

# JSONL 파일의 각 라인을 파싱하여 리스트에 저장
data = []
for line in lines:
    data.append(json.loads(line))

os.makedirs('./model/output/', exist_ok=True)

for i in range(len(data)):
    file = data[i]['conditioning_image']
    image_file = '/home/junseok/dataset/test/conditioning_images/' + file
    
    control_image = load_image(image_file)
    prompt = data[i]['text']

    # generate image
    generator = torch.manual_seed(0)
    image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=control_image
    ).images[0]
    image.save("./model/output/output" + file)