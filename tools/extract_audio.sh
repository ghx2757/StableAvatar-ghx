#!/bin/bash

# 设置基础路径
BASE_DIR="lmy_white_72mins/square/speech"

# 计数器
total_count=0
processed_count=0

# 统计总文件数
echo "🔍 正在统计视频文件总数..."
for subdir in "$BASE_DIR"/*; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        echo "  检查目录: $subdir_name"
        for video_file in "$subdir"/*.mp4; do
            if [ -f "$video_file" ]; then
                total_count=$((total_count + 1))
            fi
        done
    fi
done

echo "📊 总共找到 $total_count 个视频文件"
echo "🚀 开始处理..."

# 处理每个子目录
for subdir in "$BASE_DIR"/*; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        echo "📁 处理目录: $subdir_name"
        
        # 处理该目录下的每个mp4文件
        for video_file in "$subdir"/*.mp4; do
            if [ -f "$video_file" ]; then
                processed_count=$((processed_count + 1))
                
                # 获取视频文件名（不含扩展名）
                video_basename=$(basename "$video_file" .mp4)
                
                # 创建输出目录
                output_dir="$(dirname "$video_file")"
                
                echo "🎬 [$processed_count/$total_count] 处理: $video_basename.mp4"
                echo "📂 输出到: $output_dir"
                
                # 1. 提取音频为wav文件
                echo "🎵 正在提取音频..."
                python tools/audio_extractor.py --video_path "$video_file" --saved_audio_path "$output_dir/audio.wav"
                if [ $? -eq 0 ]; then
                    echo "✅ 音频提取完成: $output_dir/audio.wav"
                else
                    echo "❌ 错误: 音频提取失败"
                    continue
                fi
                # 2. 分离人声
                echo "🎤 正在分离人声..."
                python tools/vocal_seperator.py \
                    --audio_separator_model_file="/root/group-shared/digital-human/ghx/StableAvatar/checkpoints/Kim_Vocal_2.onnx" \
                    --audio_file_path="$output_dir/audio.wav" \
                    --saved_vocal_path="$output_dir/vocal.wav"
                if [ $? -eq 0 ]; then
                    echo "✅ 人声分离完成: $output_dir/vocal.wav"
                else
                    echo "❌ 错误: 人声分离失败"
                fi
                
                
                # if [ $? -eq 0 ]; then
                #     frame_count=$(ls "$images_dir"/frame_*.png 2>/dev/null | wc -l)
                #     echo "✅ 成功提取 $frame_count 帧到 $images_dir"
                # else
                #     echo "❌ 错误: 视频帧提取失败"
                # fi
                
                echo "---"
            fi
        done
    fi
done

echo "🎉 处理完成! 总共处理了 $processed_count 个视频文件"
# echo ""
# echo "生成的数据集结构："
# echo "talking_face_data/"
# echo "└── rec/"
# echo "    └── speech/"
# echo "        ├── 00001/"
# echo "        │   ├── sub_clip.mp4"
# echo "        │   ├── audio.wav"
# echo "        │   ├── images/"
# echo "        │   ├── face_masks/"
# echo "        │   └── lip_masks/"
# echo "        ├── 00002/"
# echo "        └── ..."
# echo ""
# echo "注意: face_masks和lip_masks目录已创建，但需要使用专门的工具来生成遮罩图像"
