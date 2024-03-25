export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./model/controlnet_base"
export CONTROLNET_DIR="lllyasviel/sd-controlnet-canny" 

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path $MODEL_DIR \
 --controlnet_model_name_or_path $CONTROLNET_DIR\
 --output_dir $OUTPUT_DIR \
 --train_data_dir "/home/junseok/controlnet_dataset/train/" \
 --resolution 512 \
 --learning_rate 1e-5 \
 --train_batch_size 4 \
 --validation_image "/home/junseok/controlnet_dataset/test/conditioning_images/00006_00.jpg" "/home/junseok/controlnet_dataset/test/conditioning_images/00008_00.jpg" "/home/junseok/controlnet_dataset/test/conditioning_images/00013_00.jpg" \
 --validation_prompt "high quality garment photo of black  long sleeve t-shirt with white background." "high quality garment photo of long-sleeve tee with white background." "high quality garment photo of a mid t-shirt with white background" \
 --num_train_epochs 20 \
 --gradient_accumulation_steps 4 \
 --report_to "wandb" \
 --checkpointing_steps  2000\
 --validation_steps 1000 \
 --checkpoints_total_limit 0 \
 --proportion_empty_prompts 0.5 \
