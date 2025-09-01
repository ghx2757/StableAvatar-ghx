import os
import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_frames_from_video(video_path, output_frames_dir):
    """使用ffmpeg从视频中提取帧"""
    os.makedirs(output_frames_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-i', video_path, '-q:v', '1', '-start_number', '0',
        os.path.join(output_frames_dir, 'frame_%d.png'), '-y'
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg frame extraction failed. Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg and ensure it's in PATH.")


def process_single_video(video_file, output_base_dir, video_index):
    """处理单个视频文件"""
    try:
        print(f"Processing video {video_index}: {video_file}")
        
        # 检查输入视频文件是否存在
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Input video file not found: {video_file}")
        
        # 创建输出目录结构
        video_dir = os.path.join(output_base_dir, "square", "speech", f"{video_index:05d}")
        images_dir = os.path.join(video_dir, "images")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # 复制视频文件到目标位置
        target_video_path = os.path.join(video_dir, "sub_clip.mp4")
        shutil.copy(video_file, target_video_path)
        
        # 提取视频帧
        extract_frames_from_video(video_file, images_dir)
        
        print(f"Successfully processed video {video_index}: {video_file}")
        return True
        
    except Exception as e:
        print(f"Error processing video {video_index} ({video_file}): {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process videos for StableAvatar training data - Extract frames only")
    parser.add_argument("--input_videos", type=str, required=True,
                       help="Input folder path containing video files")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output folder path for organized data")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of worker threads")
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_videos):
        raise ValueError(f"Input directory does not exist: {args.input_videos}")
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Input directory: {args.input_videos}")
    print(f"Output directory: {args.output_path}")
    
    # 获取所有视频文件
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    
    for file in os.listdir(args.input_videos):
        file_path = os.path.join(args.input_videos, file)
        if os.path.isfile(file_path) and Path(file).suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print("No video files found in the input directory.")
        return
    
    video_files.sort()  # 确保处理顺序一致
    print(f"Found {len(video_files)} video files to process")
    
    # 多线程处理视频
    successful_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_video = {}
        for idx, video_file in enumerate(video_files):
            future = executor.submit(
                process_single_video, 
                video_file, 
                args.output_path, 
                idx + 1  # 从1开始编号
            )
            future_to_video[future] = (idx + 1, video_file)
        
        # 等待所有任务完成
        for future in as_completed(future_to_video):
            video_index, video_file = future_to_video[future]
            try:
                success = future.result()
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Exception occurred for video {video_index} ({video_file}): {str(e)}")
                failed_count += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful_count} videos")
    print(f"Failed to process: {failed_count} videos")
    print(f"Total videos: {len(video_files)}")


if __name__ == "__main__":
    main()
