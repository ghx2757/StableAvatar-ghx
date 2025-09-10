export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"

reference_path=TestData/inference/frame_lmy_f0009_crop.png
audio_path=TestData/audio/lmy_60s_16k.wav
checkpoint_path_prefix=checkpoints/lmy_f0009_10mins_lora
output_prefix=TestData/output/lmy-f0009-singletrain_temp
train_steps=2000

CUDA_VISIBLE_DEVICES=3 python inference.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path="${checkpoint_path_prefix}/checkpoint-${train_steps}/transformer3d-checkpoint-${train_steps}.pt" \
  --pretrained_wav2vec_path="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path=$reference_path \
  --validation_driven_audio_path=$audio_path \
  --output_dir="${output_prefix}/${train_steps}/" \
  --validation_prompts="一位女教师坐在书桌前，直接对着镜头讲话。她的手势富有动感且节奏鲜明，与话语内容完美呼应。双手清晰可见，活动自如，毫无遮挡。说话时牙齿清晰，头部晃动少。面部表情生动饱满，饱含情感，为表达增色不少。镜头始终保持稳定，精准捕捉每一个清晰的动作，人物全程散发着专注且富有感染力的气场，背景为纯色绿幕。" \
  --seed=42 \
  --ulysses_degree=1 \
  --ring_degree=1 \
  --motion_frame=25 \
  --sample_steps=50 \
  --width=512 \
  --height=512 \
  --overlap_window_length=5 \
  --clip_sample_n_frames=81 \
  --GPU_memory_mode="model_full_load" \
  --sample_text_guide_scale=3.0 \
  --sample_audio_guide_scale=5.0 \
  --lora_path="${checkpoint_path_prefix}/checkpoint-${train_steps}/lora-checkpoint-${train_steps}.pt" \
  --rank=128 \
  --network_alpha=64

ffmpeg -i "${output_prefix}/${train_steps}/video_without_audio.mp4" -i "${audio_path}" -c:v copy -c:a aac -shortest "${output_prefix}/${train_steps}/video.mp4"