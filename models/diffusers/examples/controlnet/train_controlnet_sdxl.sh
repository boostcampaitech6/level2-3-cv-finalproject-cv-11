export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/junseok/model/controlnet_sdxl"
export CONTROLNET_DIR="diffusers/controlnet-canny-sdxl-1.0" 
accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path $MODEL_DIR \
 --output_dir $OUTPUT_DIR \
 --controlnet_model_name_or_path $CONTROLNET_DIR \
 --dataset_name "/home/junseok/dataset/train/" \
 --mixed_precision "fp16" \
 --resolution 512 \
 --learning_rate 1e-5 \
 --max_train_steps 15000 \
 --validation_image "/home/junseok/dataset/test/conditioning_images/00006_00.jpg" "/home/junseok/dataset/test/conditioning_images/00008_00.jpg" "/home/junseok/dataset/test/conditioning_images/00013_00.jpg" \
 --validation_prompt "The cloth is a black shirt, and the background is white." "The cloth is a red and white striped shirt. The background is white, which provides a contrasting color to the red and white shirt." "The cloth is a pink shirt, and the background is white." \
 --validation_steps 1000 \
 --checkpointing_steps  1000 \
 --train_batch_size 2 \
 --checkpoints_total_limit 0 \
 --proportion_empty_prompts 0.5 \
 --gradient_accumulation_steps 4 \
 --report_to "wandb" \
 --seed 42 \
