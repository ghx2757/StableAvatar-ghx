import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np

# 全局锁用于线程安全打印
print_lock = Lock()

def process_single_image(image_path, face_save_path, face_mesh, upper_lip_idx, lower_lip_idx):
    """处理单个图像并生成嘴唇mask"""
    try:
        # 检查是否已经存在
        if os.path.exists(face_save_path):
            with print_lock:
                print(f"{face_save_path} already exists!")
            return True

        # 读取并处理图像
        image = cv2.imread(image_path)
        if image is None:
            with print_lock:
                print(f"Failed to load image: {image_path}")
            return False
            
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        mask = np.zeros((h, w), dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                upper_points = np.array([
                    [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                    for i in upper_lip_idx
                ], dtype=np.int32)
                lower_points = np.array([
                    [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                    for i in lower_lip_idx
                ], dtype=np.int32)
                cv2.fillPoly(mask, [upper_points], 255)
                cv2.fillPoly(mask, [lower_points], 255)
        else:
            with print_lock:
                print(f"No face detected in {image_path}. Saving empty mask.")
        
        cv2.imwrite(face_save_path, mask)
        with print_lock:
            print(f"Lip mask saved to {face_save_path}")
        return True
        
    except Exception as e:
        with print_lock:
            print(f"Error processing {image_path}: {e}")
        return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_root", type=str)
    parser.add_argument("--start", type=int, help="Specify the value of start")
    parser.add_argument("--end", type=int, help="Specify the value of end")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker threads")
    args = parser.parse_args()

    folder_root = args.folder_root
    start = args.start
    end = args.end
    max_workers = args.max_workers

    print(f"Starting processing with {max_workers} workers...")

    # 定义嘴唇关键点索引
    upper_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    lower_lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    # 收集所有需要处理的图像任务
    tasks = []
    for idx in range(start, end):
        subfolder = str(idx).zfill(5)
        subfolder_path = os.path.join(folder_root, subfolder)
        images_folder = os.path.join(subfolder_path, "images")
        if os.path.exists(images_folder):
            face_masks_folder = os.path.join(subfolder_path, "lip_masks")
            os.makedirs(face_masks_folder, exist_ok=True)
            for root, dirs, files in os.walk(images_folder):
                for file in files:
                    if file.endswith('.png'):
                        file_name = os.path.splitext(file)[0]
                        image_name = file_name + '.png'
                        image_path = os.path.join(images_folder, image_name)
                        face_save_path = os.path.join(face_masks_folder, file_name + '.png')
                        tasks.append((image_path, face_save_path))
        else:
            print(f"{images_folder} does not exist")

    print(f"Found {len(tasks)} images to process")

    if len(tasks) == 0:
        print("No images found to process")
        exit(0)

    # 使用多线程处理
    successful_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个线程创建独立的MediaPipe实例
        def worker_function(task):
            image_path, face_save_path = task
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10)
            return process_single_image(image_path, face_save_path, face_mesh, upper_lip_idx, lower_lip_idx)
        
        # 提交所有任务
        future_to_task = {executor.submit(worker_function, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as exc:
                failed_count += 1
                with print_lock:
                    print(f'Task {task[0]} generated an exception: {exc}')

    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(tasks)}")