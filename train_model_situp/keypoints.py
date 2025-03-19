import os
import numpy as np
import cv2
import mediapipe as mp

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Đường dẫn thư mục ảnh
folder_situp = "frame/1/"  # Gập bụng
folder_sitdown = "frame/0/"  # Không gập

data = []
labels = []

def extract_keypoints(image):
    """ Trích xuất keypoints từ ảnh bằng Mediapipe Pose và chuẩn hóa tọa độ """
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark]).flatten()
        # Chuẩn hóa tọa độ
        keypoints = (keypoints - np.mean(keypoints)) / np.std(keypoints)
        return keypoints
    return None

# Xử lý ảnh trong thư mục frame/1/ (Gập bụng)
for filename in os.listdir(folder_situp):
    img_path = os.path.join(folder_situp, filename)
    img = cv2.imread(img_path)
    keypoints = extract_keypoints(img)
    if keypoints is not None:
        data.append(keypoints)
        labels.append(1)  # Nhãn 1: Gập bụng

# Xử lý ảnh trong thư mục frame/0/ (Không gập)
for filename in os.listdir(folder_sitdown):
    img_path = os.path.join(folder_sitdown, filename)
    img = cv2.imread(img_path)
    keypoints = extract_keypoints(img)
    if keypoints is not None:
        data.append(keypoints)
        labels.append(0)  # Nhãn 0: Không gập bụng

# Chuyển thành mảng numpy
data = np.array(data)
labels = np.array(labels)

# Lưu dataset
np.save("keypoints/keypoints_data.npy", data)
np.save("keypoints/keypoints_labels.npy", labels)

print(f"Dataset đã lưu: {data.shape}, Labels: {labels.shape}")