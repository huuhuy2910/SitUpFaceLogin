import cv2
import os
import sys
from flask import Flask, request, jsonify

app = Flask(__name__)

def collect_face_data(name):
    data_dir = "face_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    person_dir = os.path.join(data_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    count = 0
    print("Nhìn vào camera và chờ...")

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            file_name = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_name, face_img)
            count += 1
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập xong {count} ảnh cho {name}")

@app.route('/collect_face_data', methods=['POST'])
def collect_face_data_route():
    data = request.get_json()
    name = data.get("name", "Unknown")
    collect_face_data(name)
    return jsonify({"message": f"Collected images for {name}", "success": True})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "Unknown"
    collect_face_data(name)
    app.run(debug=True)