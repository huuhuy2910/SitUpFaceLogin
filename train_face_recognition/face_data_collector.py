import cv2
import os
import mysql.connector

def collect_face_data():
    # Tạo thư mục để lưu dữ liệu nếu chưa có
    data_dir = "face_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Nhập tên người cần thu thập dữ liệu
    name = input("Nhập tên người: ")
    person_dir = os.path.join(data_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Thêm người dùng vào cơ sở dữ liệu MySQL
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='1234',
            database='fitness_tracking'
        )
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (name, encoding) VALUES (%s, %s)", (name, b''))
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    # Khởi động webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    #   # Khởi động IP camera
    # ip_camera_url = "http://172.16.1.56:4747/video"
    # cap = cv2.VideoCapture(ip_camera_url)
    
    # face_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    # )
    count = 0
    print("Nhìn vào camera và chờ...")

    while count < 100:  # Thu thập 100 ảnh
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển sang ảnh xám để phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Cắt khuôn mặt
            face_img = frame[y:y+h, x:x+w]
            # Lưu ảnh
            file_name = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_name, face_img)
            print(f"Lưu ảnh: {file_name}")  # Thêm dòng này để kiểm tra việc lưu ảnh
            count += 1
            
            # Vẽ khung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập xong {count} ảnh cho {name}")

if __name__ == "__main__":
    collect_face_data()