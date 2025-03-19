import cv2
import face_recognition
import pickle
import numpy as np
import tensorflow as tf
import mediapipe as mp
import collections
import threading
import time
import winsound 
import pyttsx3  
from flask import Flask, render_template, Response, jsonify, request
import mysql.connector
from mysql.connector import Error
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "fitness_tracking"
}

def speak(text):
    """
    Phát giọng nói cho văn bản truyền vào.
    Mỗi lần gọi sẽ tạo một instance mới của pyttsx3 để tránh lỗi 'run loop already started'.
    """
    local_engine = pyttsx3.init()
    local_engine.say(text)
    local_engine.runAndWait()
    local_engine.stop()

def async_speak(text):
    """Gọi hàm speak() trong một luồng riêng để không chặn các hoạt động khác."""
    try:
        threading.Thread(target=speak, args=(text,), daemon=True).start()
    except Exception as e:
        logging.error("Exception in async_speak", exc_info=True)

# --------------------------
# Face Recognition Setup
# --------------------------
with open("face_model.pkl", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]      # List of known face encodings
known_face_names = data["names"]              # Corresponding names

# --------------------------
# Sit-up LSTM Model & Pose Setup
# --------------------------
model = tf.keras.models.load_model("train_model_situp/Model_situp_lstm.h5")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --------------------------
# Global Variables
# --------------------------
# Face recognition
current_user_id = None
current_user_name = "Unknown"
confirmed_user_name = None   # Tên đã được xác nhận (khóa lại sau khi nhấn Confirm)
recognized_face = None       # Luôn cập nhật recognized_face, kể cả khi kết quả là "Unknown"

# Sit-up counting & cooldown
situp_count = 0
previous_state = 0
predictions_queue = collections.deque(maxlen=5)
state_queue = collections.deque(maxlen=5)
down_position = False
ready_to_count = False   # Được set True sau khi cooldown hoàn tất
display_state = "Waiting..."  # "Lying Down", "Sit-Up", hoặc "Success! Completed 1 set"

# Các thông báo bổ sung hiển thị trên giao diện
pose_instruction = ""     # Ví dụ: "Please lie down and show your full body."
cooldown_message = ""     # Ví dụ: "Cooldown: 5s" hoặc "Cooldown Done"

cooldown_duration = 10    # Cooldown 10 giây
cooldown_started = False
cooldown_start_time = None

# Flags
start_counting_flag = False  # Được đặt True sau khi nhấn Start
paused_flag = False

# Flags cho giọng nói (để tránh lặp nhiều lần)
voice_position_announced = False
set_completed_announced = False

# Biến toàn cục theo dõi thời gian thông báo cuối
last_instruction_time = 0

# --------------------------
# Utility Functions
# --------------------------
def extract_keypoints(image):
    """Trích xuất keypoints từ ảnh sử dụng Mediapipe Pose."""
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark]).flatten()
        return results.pose_landmarks, keypoints
    return None, None

def check_full_body(keypoints):
    """Kiểm tra xem toàn bộ cơ thể (các landmark quan trọng) đã được nhận diện hay chưa."""
    required_points = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    if keypoints is None or len(keypoints) < 33 * 2:
        return False
    for idx in required_points:
        if keypoints[idx*2] == 0 or keypoints[idx*2+1] == 0:
            return False
    return True

def is_body_horizontal(keypoints):
    """Kiểm tra xem cơ thể có nằm ngang không dựa vào khoảng cách giữa vai và hông."""
    if keypoints is None or len(keypoints) < 33 * 2:
        return False  
    left_shoulder, right_shoulder = keypoints[11*2], keypoints[12*2]
    left_hip, right_hip = keypoints[23*2], keypoints[24*2]
    return abs(left_shoulder - right_shoulder) < 0.1 and abs(left_hip - right_hip) < 0.1

def predict_action(keypoints):
    """
    Chạy model dự đoán động tác và cập nhật số lần gập bụng.
    """
    global previous_state, situp_count, down_position, display_state
    keypoints = keypoints.reshape(1, 1, -1)
    prediction = model.predict(keypoints)[0][0]
    predictions_queue.append(prediction)
    smoothed_prediction = np.mean(predictions_queue)
    current_state = 1 if smoothed_prediction > 0.95 else 0  

    if current_state == 0:
        down_position = True  

    if down_position and previous_state == 0 and current_state == 1:
        situp_count += 1
        down_position = False  
        winsound.Beep(1000, 200)
        print(f"✅ Sit-up count: {situp_count}")

    previous_state = current_state  
    state_queue.append("Lying Down" if current_state == 0 else "Sit-Up")
    display_state = collections.Counter(state_queue).most_common(1)[0][0]

def countdown_timer():
    """
    Đếm ngược cooldown 10 giây khi cơ thể nằm ngang.
    Cập nhật global cooldown_message.
    """
    global ready_to_count, cooldown_start_time, cooldown_message
    try:
        while True:
            elapsed = time.time() - cooldown_start_time
            remaining = int(cooldown_duration - elapsed)
            cooldown_message = f"Cooldown: {remaining}s"
            print(cooldown_message)
            if remaining <= 0:
                winsound.Beep(1000, 500)
                ready_to_count = True
                cooldown_message = "Cooldown Done"
                async_speak("Start counting!")
                break
            winsound.Beep(1000, 500)
            time.sleep(1)
    except Exception as e:
        logging.error("Exception in countdown_timer", exc_info=True)

def recognize_face(frame):
    """
    Nhận diện khuôn mặt trong frame.
    Hiển thị khung xanh quét mặt liên tục và cập nhật recognized_face.
    Nếu tên không phải "Unknown", cập nhật current_user_id từ DB.
    """
    global current_user_id, current_user_name, recognized_face
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    print(f"Detected {len(face_locations)} face(s)")
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = 0
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = int((1 - face_distances[best_match_index]) * 100)
            if matches[best_match_index] and confidence >= 60:
                name = known_face_names[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence}%)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if name != "Unknown":
            try:
                connection = mysql.connector.connect(**DB_CONFIG)
                cursor = connection.cursor()
                cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
                user = cursor.fetchone()
                if not user:
                    cursor.execute("INSERT INTO users (name, encoding) VALUES (%s, %s)",
                                   (name, face_encoding.tobytes()))
                    connection.commit()
                    cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
                    user = cursor.fetchone()
                if user:
                    current_user_id = user[0]
            except Error as e:
                print(f"Error: {e}")
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()
        recognized_face = {"name": name, "confidence": confidence, "box": (top, right, bottom, left)}
        current_user_name = name
        print(f"Face recognized: {name} ({confidence}%)")
    return

def gen_frames():
    """
    Đọc frame từ webcam.
    Nếu chưa xác nhận người dùng, thực hiện nhận diện khuôn mặt mỗi 30 frame.
    Sau khi nhấn Confirm & Start, nếu nhận diện được toàn bộ cơ thể
    (check_full_body và is_body_horizontal) thì:
      - Nếu cơ thể chưa đúng vị trí, phát lời nhắc "Please lie down and show your full body." mỗi 5 giây.
      - Nếu đúng vị trí và cooldown hoàn tất, gọi predict_action để cập nhật số gập.
    Khi số gập đạt 12, phát lời chúc mừng hoàn thành set và dừng đếm.
    """
    global situp_count, start_counting_flag, recognized_face, cooldown_started, cooldown_start_time, ready_to_count, display_state, pose_instruction, cooldown_message, voice_position_announced, last_instruction_time, set_completed_announced
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        frame = cv2.flip(frame, 1)
        if not start_counting_flag:
            if frame_count % 10 == 0:  # Reduce interval to every 10 frames
                recognize_face(frame)
            if recognized_face is not None:
                top, right, bottom, left = recognized_face["box"]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if start_counting_flag:
            landmarks, keypoints = extract_keypoints(frame)
            if keypoints is not None:
                # Nếu cơ thể không đúng vị trí (không đủ keypoints hoặc không nằm ngang)
                if not (check_full_body(keypoints) and is_body_horizontal(keypoints)):
                    pose_instruction = "Please lie down and show your full body."
                    display_state = "Waiting..."
                    if time.time() - last_instruction_time > 2:  # Reduce interval to every 2 seconds
                        async_speak("Please lie down and show your full body.")
                        last_instruction_time = time.time()
                else:
                    pose_instruction = ""
                    last_instruction_time = 0  # Reset nếu vị trí đúng
                    if not cooldown_started and not ready_to_count:
                        cooldown_started = True
                        cooldown_start_time = time.time()
                        threading.Thread(target=countdown_timer, daemon=True).start()
                if ready_to_count and check_full_body(keypoints):
                    threading.Thread(target=predict_action, args=(keypoints,), daemon=True).start()
                # Kiểm tra nếu số gập đạt 12, phát lời chúc mừng (nếu chưa phát)
                if situp_count >= 12 and not set_completed_announced:
                    async_speak("Congratulations! You have completed one set.")
                    set_completed_announced = True
                    start_counting_flag = False
                    display_state = "Success! Completed 1 set"
            if landmarks is not None:
                mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3))
        else:
            cv2.putText(frame, "Please confirm user first.", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame")
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_count += 1
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Endpoints ---
@app.route('/confirm', methods=['POST'])
def confirm():
    global current_user_name, recognized_face, confirmed_user_name
    if recognized_face is not None and recognized_face["name"].lower() != "unknown":
        current_user_name = recognized_face["name"]
        confirmed_user_name = current_user_name
        async_speak(f"Face recognized, welcome {current_user_name}!")
        return jsonify({"message": f"User confirmed as {current_user_name}. Now click Start to begin the workout."})
    else:
        return jsonify({"message": "Face not recognized. Please try again."})
    
@app.route('/start_counting', methods=['POST'])
def start_counting():
    global start_counting_flag, set_completed_announced, situp_count, voice_position_announced
    set_completed_announced = False
    situp_count = 0
    start_counting_flag = True
    voice_position_announced = False  # Reset flag để đảm bảo lời nhắc được phát
    async_speak("Please lie down and show your full body.")
    return jsonify({"message": "Workout started."})

@app.route('/continue_set', methods=['POST'])
def continue_set():
    global situp_count, previous_state, predictions_queue, state_queue, down_position, ready_to_count, cooldown_started, cooldown_start_time, display_state, pose_instruction, cooldown_message, start_counting_flag
    situp_count = 0
    previous_state = 0
    predictions_queue.clear()
    state_queue.clear()
    down_position = False
    ready_to_count = False
    cooldown_started = False
    cooldown_start_time = None
    display_state = "Waiting..."
    pose_instruction = ""
    cooldown_message = ""
    start_counting_flag = True
    return jsonify({"message": "New set started. Ready to workout!"})

@app.route('/change_user', methods=['POST'])
def change_user():
    global current_user_id, current_user_name, confirmed_user_name, start_counting_flag, situp_count, recognized_face, cooldown_started, ready_to_count, pose_instruction, cooldown_message
    current_user_id = None
    current_user_name = "Unknown"
    confirmed_user_name = None
    start_counting_flag = False
    situp_count = 0
    recognized_face = None
    cooldown_started = False
    ready_to_count = False
    pose_instruction = ""
    cooldown_message = ""
    return jsonify({"message": "User changed."})

@app.route('/save', methods=['POST'])
def save_situp_count():
    global situp_count, current_user_id, start_counting_flag, recognized_face, current_user_name, cooldown_started, ready_to_count
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        result = '1' if situp_count >= 12 else '0'
        cursor.execute("INSERT INTO situp_sessions (user_id, situp_count, result) VALUES (%s, %s, %s)",
                       (current_user_id, situp_count, result))
        connection.commit()
        return jsonify({"message": "Sit-up count saved successfully!"})
    except Error as e:
        print(f"Error: {e}")
        return jsonify({"message": "Failed to save sit-up count."})
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/status')
def status():
    global current_user_name, confirmed_user_name, start_counting_flag, paused_flag
    user_display = confirmed_user_name if confirmed_user_name is not None else current_user_name
    if confirmed_user_name is None:
        status_display = "Waiting for Confirmation"
    else:
        status_display = "Preparing" if not start_counting_flag else ("Paused" if paused_flag else "Counting")
    return jsonify({
        "user": user_display,
        "status": status_display,
        "situp_count": situp_count,
        "result": "Success" if situp_count >= 12 else "Fail",
        "pose": display_state,
        "instruction": pose_instruction,
        "cooldown": cooldown_message,
        "recognized_face": recognized_face
    })

@app.route('/pause', methods=['POST'])
def pause():
    global start_counting_flag, paused_flag
    paused_flag = True
    start_counting_flag = False
    return jsonify({"message": "Sit-up counting paused."})

@app.route('/resume', methods=['POST'])
def resume():
    global start_counting_flag, paused_flag
    paused_flag = False
    start_counting_flag = True
    return jsonify({"message": "Sit-up counting resumed."})

@app.route('/logout', methods=['POST'])
def logout():
    global current_user_id, current_user_name, confirmed_user_name, start_counting_flag, situp_count, recognized_face, cooldown_started, ready_to_count, pose_instruction, cooldown_message, set_completed_announced
    current_user_id = None
    current_user_name = "Unknown"
    confirmed_user_name = None
    start_counting_flag = False
    situp_count = 0
    recognized_face = None
    cooldown_started = False
    ready_to_count = False
    pose_instruction = ""
    cooldown_message = ""
    set_completed_announced = False  # Reset set completed flag
    return jsonify({"message": "Logged out."})

@app.route('/user_history')
def user_history():
    global current_user_id, confirmed_user_name
    if current_user_id is None or confirmed_user_name is None or confirmed_user_name.lower() == "unknown":
        return jsonify({"user_name": "No Confirmed User", "history": []})
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT situp_count, result, session_time 
            FROM situp_sessions 
            WHERE user_id = %s 
            ORDER BY session_time DESC
        """, (current_user_id,))
        history = cursor.fetchall()
        return jsonify({"user_name": confirmed_user_name, "history": history})
    except Error as e:
        print(f"Error: {e}")
        return jsonify({"user_name": confirmed_user_name, "history": []})
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error("Exception in main", exc_info=True)
