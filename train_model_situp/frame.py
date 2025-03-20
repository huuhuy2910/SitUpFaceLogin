import cv2
import os
import numpy as np
import mediapipe as mp

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Thư mục chứa video đầu vào
video_folder = "data/"
output_folder_situp = "frame/1/"  # Gập bụng lên
output_folder_sitdown = "frame/0/"  # Hạ xuống

os.makedirs(output_folder_situp, exist_ok=True)
os.makedirs(output_folder_sitdown, exist_ok=True)

def extract_keypoints(image):
    """ Trích xuất keypoints từ frame bằng Mediapipe Pose """
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
        return keypoints
    return None

def calculate_angle(a, b, c):
    """ Tính góc giữa ba điểm a-b-c """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_situp(keypoints):
    """
    Xác định tư thế sit-up dựa trên vị trí keypoints.
    - Khi gập lên (1): Góc giữa vai - hông - đầu gối nhỏ hơn 90 độ.
    - Khi hạ xuống (0): Góc giữa vai - hông - đầu gối lớn hơn 110 độ.
    """
    if keypoints is None:
        return None
    
    # Lấy các điểm quan trọng
    left_shoulder, left_hip, left_knee = keypoints[11], keypoints[23], keypoints[25]
    right_shoulder, right_hip, right_knee = keypoints[12], keypoints[24], keypoints[26]
    
    # Tính góc giữa vai - hông - đầu gối
    left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    best_angle = min(left_angle, right_angle)  # Chọn góc nhỏ nhất để đảm bảo chính xác
    
    if best_angle < 90:
        return 1  # Gập lên
    elif best_angle > 110:
        return 0  # Hạ xuống
    return None  # Không rõ tư thế

def draw_keypoints(image, keypoints):
    """ Vẽ keypoints lên ảnh để kiểm tra """
    for (x, y) in keypoints:
        x, y = int(x * image.shape[1]), int(y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def process_videos():
    """ Xử lý từng video trong thư mục videos/ """
    for filename in os.listdir(video_folder):
        if filename.endswith((".mp4", ".avi", ".mov")):
            cap = cv2.VideoCapture(os.path.join(video_folder, filename))
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                
                if frame_count % 5 == 0:
                    keypoints = extract_keypoints(frame)
                    label = classify_situp(keypoints)
                    
                    if keypoints is not None:
                        frame = draw_keypoints(frame, keypoints)
                    
                    if label is not None:
                        output_path = os.path.join(
                            output_folder_situp if label == 1 else output_folder_sitdown,
                            f"{filename}_frame{frame_count}.jpg"
                        )
                        cv2.imwrite(output_path, frame)

            cap.release()
    print("Xử lý video hoàn tất!")

# Chạy chương trình
process_videos()
