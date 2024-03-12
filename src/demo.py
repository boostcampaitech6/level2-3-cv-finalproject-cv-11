import streamlit as st
from PIL import Image
from model import SketchToRealModel  
import io

# from model import ControlNet

base_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "controlnet_base"
model = SketchToRealModel(base_model_path, controlnet_path)

# 스케치 이미지를 실제 의상 이미지로 변환하는 함수
def convert_sketch_to_real(sketch_image_file, prompt):
    # 이미지 파일을 받아 PIL 이미지 객체로 변환
    sketch_image = Image.open(io.BytesIO(sketch_image_file.read())).convert("RGB")
    generated_image = model.predict(sketch_image, prompt)
    
    return generated_image

# 실제 의상 이미지와 사람의 이미지를 합치는 함수
def create_wear_shot(person_image, real_image):
    wear_shot = dci_vton_model.predict(person_image, real_clothes_image)
    
    animated_wear_shot = magic_animate_model.predict(wear_shot)
    return wear_shot, animated_wear_shot

# 페이지 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'upload_sketch'

# 스케치 업로드 페이지
if st.session_state.page == 'upload_sketch':
    st.title("의상 스케치 및 설명 업로드")
    
    sketch_image_file = st.file_uploader("스케치된 의상 이미지를 업로드하세요.", type=['jpg', 'jpeg', 'png'])
    prompt = st.text_input("스케치에 대한 설명을 입력하세요. (예: 빨간색 긴 소매 드레스)")
    
    if sketch_image_file is not None and prompt:
        st.image(sketch_image_file, caption='업로드된 스케치', use_column_width=True)
        
        if st.button('스케치를 실제 의상으로 변환'):
            real_dress_image = convert_sketch_to_real(sketch_image_file, prompt)
            st.image(real_dress_image, caption='변환된 실제 의상', use_column_width=True)
            st.session_state.real_dress_image = real_dress_image  # 변환된 이미지를 세션 상태에 저장
            st.session_state.page = 'ask_for_tryon'

# 착용샷 여부 질문 페이지
if st.session_state.page == 'ask_for_tryon':
    if st.button('실제 착용샷을 보시겠습니까?'):
        st.session_state.page = 'upload_person'
    if st.button('아니요, 처음으로 돌아가기'):
        st.session_state.page = 'upload_sketch'

# 사람 이미지 업로드 페이지
if st.session_state.page == 'upload_person':
    st.title("사람 이미지 업로드")
    person_image_file = st.file_uploader("사람의 이미지를 업로드하세요.", type=['jpg', 'jpeg', 'png'])
    if person_image_file is not None:
        person_image = Image.open(person_image_file)
        st.image(person_image, caption='업로드된 사람 이미지', use_column_width=True)
        wear_shot,animated_wear_shot  = create_wear_shot(person_image, real_image)
        st.image(wear_shot,animated_wear_shot, caption='AI 기반 착용샷', use_column_width=True)
