import torch
import os
import json
import redis
from PIL import Image
import io
from celery import Celery




#redis 설정
r = redis.Redis(host='10.0.3.6', port=6379, db=1)

# Pub/Sub 시스템에 연결
pubsub = r.pubsub()

app = Celery('tasks', broker='redis://10.0.3.6:6379/0', backend='redis://10.0.3.6:6379/1')


@app.task
def sketch_to_real(sketch_id):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    print(sketch_id)

    
    base_model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_path = "/home/rtboa/controlnet_base"
    
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker = None
    )


    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    
    
    #redis에서 이미지와 프롬프트 가져옴
    img_byte_arr = r.get(f'task:{sketch_id}:image')
    prompt = r.get(f'task:{sketch_id}:prompt').decode('utf-8')
    
    control_image = Image.open(io.BytesIO(img_byte_arr))
    control_image = control_image.convert("RGB") # diffusers가 처리할 수 있는 형식으로 변환
    
    # 이미지 생성
    generator = torch.manual_seed(0)
    output_image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]
    
    
    # 결과 이미지를 byte array로 변환하여 Redis에 저장
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    r.set(f'task:{sketch_id}:result', img_byte_arr)
    
    # 작업 상태를 'completed'로 설정
    r.set(f'task:{sketch_id}:status', 'completed')


# 채널에서 메시지를 구독하는 함수
def start_listening():
    # 'sketch_channel' 채널을 구독하고 콜백 함수를 설정
    print('#####')
    pubsub.subscribe('sketch_channel')
    
    # 메시지 대기 루프
    for message in pubsub.listen():
        if message['type'] == 'message':
            sketch_id = message['data'].decode('utf-8')
            # Celery에 sketch_to_real 작업을 추가
            sketch_to_real.delay(sketch_id)



if __name__ == "__main__":
    # 메시지 수신 시작
    start_listening()