export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
export WORLD_SIZE=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 inference.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path="checkpoints/lmy_white_72mins/checkpoint-10000/transformer3d-checkpoint-10000.pt" \
  --pretrained_wav2vec_path="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="TestData/inference/lmy_white_small.png" \
  --validation_driven_audio_path="TestData/audio/lmy_60s_16k.wav" \
  --output_dir="TestData/output/lmy-white-10000" \
  --validation_prompts="A realistic video of a short-haired woman with black hair speaking directly to the camera, wearing a necklace and a white sports jacket, with dynamic and rhythmic hand gestures that complement his speech. Her hands are clearly visible, independent, and unobstructed. Her facial expressions are expressive and full of emotion, enhancing the delivery. Her mouth is open when talking and close when not talking, her teeth are visible. The camera remains steady, capturing sharp, clear movements and a focused, engaging presence, background is green screen." \
  --seed=42 \
  --ulysses_degree=2 \
  --ring_degree=2 \
  --motion_frame=25 \
  --sample_steps=50 \
  --width=512 \
  --height=512 \
  --fsdp_dit \
  --overlap_window_length=10 \
  --sample_text_guide_scale=3.0 \
  --sample_audio_guide_scale=5.0
