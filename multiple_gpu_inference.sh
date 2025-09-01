export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
export WORLD_SIZE=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 inference.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-rec-vec.pt" \
  --pretrained_wav2vec_path="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="TestData/inference/sxy_image.png" \
  --validation_driven_audio_path="TestData/audio/sxy_ss30_60s_16k.WAV" \
  --output_dir="TestData/output/sxy-src" \
  --validation_prompts="A realistic video scene:A female teacher sits on a chair, giving a lecture directly to the camera. Her hand gestures are dynamic and rhythmic, perfectly complementing the content of her lecture as she teaches. When her hands appear in the frame, they are clearly visible, move freely, and remain completely unobstructed. Her facial expressions are vivid, full of emotion, and add much to her delivery. The camera stays steady throughout, capturing every movement with sharp clarity, while the figure exudes a focused and engaging presence that is highly contagious." \
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
