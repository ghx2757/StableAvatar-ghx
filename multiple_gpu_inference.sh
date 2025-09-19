export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
export WORLD_SIZE=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

reference_path=TestData/inference/frame_lmy_f0009_crop.png
audio_path=TestData/audio/lmy_10s.WAV
checkpoint_path_prefix=checkpoints/lmy_f0009_10mins
output_prefix=TestData/output/lmy-f0009-30mins-savelatents-multiGPU-newDecode2
train_steps=18000

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 inference.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path="${checkpoint_path_prefix}/checkpoint-${train_steps}/transformer3d-checkpoint-${train_steps}.pt" \
  --pretrained_wav2vec_path="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path=$reference_path \
  --validation_driven_audio_path=$audio_path \
  --output_dir="${output_prefix}/${train_steps}/" \
  --validation_prompts="A female teacher in a crisp white top sits at a neat desk, speaking to the camera in a warm tone. She has neat shoulder-length black hair, and her slender hands (with neatly trimmed nude nails) move steadily and naturally, syncing with her words—either lifting palms to emphasize points or sweeping side-to-side to explain concepts. She holds no items, and her hands stay visible. When speaking, her white teeth show, her head stays steady, and her expressions are vivid: smiling at anecdotes, focusing on complex ideas, and softening her gaze for the audience. The camera is rock-steady (medium close-up) capturing her movements and expressions. She radiates a focused, earnest aura, with a clean, vibrant solid-color green screen background (even lighting, no distractions)." \
  --seed=42 \
  --ulysses_degree=2 \
  --ring_degree=2 \
  --motion_frame=25 \
  --sample_steps=50 \
  --width=512 \
  --height=512 \
  --fsdp_dit \
  --overlap_window_length=10 \
  --clip_sample_n_frames=81 \
  --sample_text_guide_scale=3.0 \
  --sample_audio_guide_scale=5.0 \
  --lora_path= \
  --rank=128 \
  --network_alpha=64 \
  --overlapping_weight_scheme='log'

ffmpeg -i "${output_prefix}/${train_steps}/video_without_audio.mp4" -i "${audio_path}" -c:v copy -c:a aac -shortest "${output_prefix}/${train_steps}/video.mp4"