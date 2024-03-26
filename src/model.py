from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np

class SketchToRealModel:
    def __init__(self, base_model_path, controlnet_path):
        # ControlNet 모델 로드
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        # StableDiffusionControlNetPipeline 설정
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=self.controlnet, torch_dtype=torch.float16, safety_checker=None
        )

        # 속도 향상을 위한 설정
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        # 메모리 최적화 설정
        self.enable_memory_optimizations()

    def enable_memory_optimizations(self):
        # xformers가 설치되어 있는 경우 아래 줄 주석 해제
        # self.pipe.enable_xformers_memory_efficient_attention()
        # 모델 CPU 오프로드를 활성화하여 메모리 최적화
        self.pipe.enable_model_cpu_offload()

    def predict(self, image: Image.Image, prompt):
        # PIL.Image 객체를 모델이 처리할 수 있는 형태로 변환
        image = self.preprocess_image(image)

        # 생성 과정의 재현성을 위해 generator 설정
        generator = torch.manual_seed(0)

        # 이미지 생성
        with torch.no_grad():
            generated_image = self.pipe(prompt=prompt, num_inference_steps=20, generator=generator, image=image).images[0]

        return generated_image

    def preprocess_image(self, image: Image.Image):
        image = image.resize((224, 224))
        image = np.array(image) / 255.0  
        image = np.transpose(image, (2, 0, 1))  # 채널을 앞으로 이동
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가

        return image


if __name__ == "__main__":
    main()
    # # 모델 경로 설정
    # base_model_path = "runwayml/stable-diffusion-v1-5"
    # controlnet_path = "controlnet_base"

    # # 모델 인스턴스 생성
    # model = SketchToRealModel(base_model_path, controlnet_path)
    
   