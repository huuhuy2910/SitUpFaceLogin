# **Ứng dụng AI trong xác thực khuôn mặt và LSTM trong nhận diện & đếm động tác gập bụng từ video - SitUpFaceLogin**  
---

<p align="center">
  <img src="image/logo.png" alt="DaiNam University Logo" width="200"/>
  <img src="image/AIoTLab_logo.png" alt="AIoTLab Logo" width="170"/>
</p>
<p align="center">
  <a href="https://www.facebook.com/DNUAIoTLab">
    <img src="https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge" alt="Made by AIoTLab"/>
  </a>
  <a href="https://fitdnu.net/">
    <img src="https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge" alt="Fit DNU"/>
  </a>
  <a href="https://dainam.edu.vn">
    <img src="https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge" alt="DaiNam University"/>
  </a>
</p>

## **Giới thiệu**  
**SitUpFaceLogin** là một dự án sử dụng **nhận diện khuôn mặt** và **nhận diện tư thế** để **đếm số lần gập bụng**. Hệ thống sẽ xác thực người dùng bằng khuôn mặt trước khi theo dõi số lần tập luyện và lưu trữ kết quả vào **MySQL** được hướng dẫn và góp ý bởi 2 giảng viên **LÊ TRUNG HIẾU** và **NGUYỄN VĂN NHÂN** thuộc *KHOA CÔNG NGHỆ THÔNG TIN* - **ĐẠI HỌC ĐẠI NAM "Dainam University"**.  

Tài liệu này hướng dẫn chi tiết cách **cài đặt** và **chạy** hệ thống.  
## **Thành viên tham gia**
| STT | Tên sinh viên         | Mã sinh viên    | Nhóm  | Lớp        |  
|-----|----------------------|---------------|-------|-----------|  
| 1   | Nguyễn Hữu Huy       | 1671020139    | 10    | CNTT 16-01 |  
| 2   | Đặng Lê Hoàng Anh    | 1671020010    | 10    | CNTT 16-01 |  
| 3   | Nguyễn Văn Nguyên    | 1671020229    | 10    | CNTT 16-01 |  


## **Mô hình hoạt động**
![image](https://github.com/user-attachments/assets/5ff27214-e647-402d-8008-d2fa27e15780)

---
💡 **Công nghệ sử dụng:**  
- **Face Recognition**: Nhận diện khuôn mặt  
- **OpenCV**: Xử lý hình ảnh và video  
- **MediaPipe/OpenPose**: Trích xuất keypoints  
- **LSTM (Long Short-Term Memory)**: Nhận diện động tác gập bụng  
- **Flask**: API backend  
- **MySQL**: Lưu trữ dữ liệu  

---
## **Yêu cầu hệ thống**  
- **Python** 3.7 trở lên  
- **MySQL Server**  
- **OpenCV, MediaPipe, TensorFlow**  
- Các thư viện Python cần thiết (**liệt kê trong `requirements.txt`**)  

---

## **Hướng dẫn cài đặt**  

### **1. Cài đặt các thư viện cần thiết**  
Chạy lệnh sau để cài đặt các thư viện Python yêu cầu:  
```sh
pip install -r requirements.txt
```

---

### **2. Thiết lập cơ sở dữ liệu MySQL**  

#### **2.1. Cài đặt MySQL Server**  
- Cài đặt MySQL Server (nếu chưa có).  
- Đảm bảo MySQL đang chạy trên hệ thống.  

#### **2.2. Tạo cơ sở dữ liệu**  
Mở MySQL và chạy lệnh sau để tạo cơ sở dữ liệu **`fitness_tracking`**:  
```sql
CREATE DATABASE fitness_tracking;
```

#### **2.3. Tạo các bảng cần thiết**  
Kết nối đến cơ sở dữ liệu **fitness_tracking** và tạo các bảng sau:  

```sql
USE fitness_tracking;

-- Bảng lưu thông tin người dùng nhận diện khuôn mặt
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    encoding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng lưu lịch sử nhận diện khuôn mặt
CREATE TABLE IF NOT EXISTS face_recognition_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);
-- Bảng lưu lịch sử bài tập gập bụng
CREATE TABLE IF NOT EXISTS situp_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    situp_count INT NOT NULL,
    result TINYINT(1) NOT NULL,  -- 1 means "passed", 0 means "not passed"
    session_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

```

#### **2.4. Cấu hình kết nối MySQL trong `app.py`**  
Mở file **`app.py, face_data_collector.py`** và cập nhật thông tin kết nối MySQL:  
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "fitness_tracking"
}
```
🔹 **Lưu ý:** Thay `your_password` bằng mật khẩu MySQL của bạn.  

---

### **3. Thu thập dữ liệu khuôn mặt**  
Chạy script **`face_data_collector.py`** để thu thập dữ liệu khuôn mặt:  
```sh
python face_data_collector.py
```
Hoặc có thể sử dụng API Flask để thu thập dữ liệu khuôn mặt:  
```sh
curl -X POST http://localhost:5000/collect_face_data -H "Content-Type: application/json" -d '{"name": "TênCủaBạn"}'
```

---

### **4. Huấn luyện mô hình nhận diện khuôn mặt**  
Sau khi thu thập ảnh khuôn mặt, chạy notebook **`train_face.ibynb`** để huấn luyện mô hình nhận diện:  
🔹 Mô hình sau khi huấn luyện sẽ được lưu vào **`face_model.pkl`**.  

---

### **5. Chuẩn bị video làm dataset**  
#### **5.1. Thu thập video gập bụng**  
- Ghi lại các video gập bụng từ 2 góc chính là góc **45 độ** và **góc ngang**.  
- Độ phân giải tối thiểu **720p**, tốc độ khung hình **30 FPS**.  
- Mỗi video có thể kéo dài **15-60 giây**.  

| Góc ngang | Góc 45 độ |
|-----------|----------|
| ![Góc ngang](https://github.com/user-attachments/assets/515fe7ab-f236-494f-bfea-17c7f2130e5f) | ![Góc 45 độ](https://github.com/user-attachments/assets/8281e8f0-6762-4425-a10e-ac1cea4146c5) |


#### **5.2. Lưu video vào thư mục dataset**  
- Tạo thư mục **`dataset/videos`**.  
- Lưu các video vào thư mục này.  
- Định dạng video khuyến nghị: `.mp4` hoặc `.avi`.  

---

### **6. Trích xuất Keypoints từ video**  
Chạy script **`frame.py`** để trích xuất keypoints từ video tập luyện:  
```sh
python frame.py
```
🔹 Hệ thống sử dụng **MediaPipe/OpenPose** để trích xuất keypoints từ video.  

---

### **7. Chuẩn bị dữ liệu keypoints**  
Chạy script **`keypoints.py`** để xử lý dữ liệu keypoints trước khi đưa vào mô hình LSTM:  
```sh
python keypoints.py
```
🔹 Dữ liệu đầu ra sẽ là chuỗi thời gian (**time series**) dùng để huấn luyện mô hình LSTM.  

---

### **8. Huấn luyện mô hình LSTM**  
Mở và chạy notebook **`train_lstm.ipynb`** để huấn luyện mô hình LSTM nhận diện số lần gập bụng.  
Sau khi huấn luyện, mô hình sẽ được lưu dưới dạng:  
```
Model_situp_lstm.h5
```

---

### **9. Chạy ứng dụng**  
Chạy Flask API để khởi động hệ thống:  
```sh
python app.py
```
Sau khi chạy, mở trình duyệt và truy cập:  
```
http://localhost:5000
```
🔹 **Giao diện chính của ứng dụng sẽ hiển thị tại đây.**  

---

## **Các API Endpoint**  
| Endpoint                 | Phương thức | Mô tả |
|--------------------------|------------|-------|
| `/`                      | GET        | Trang chính |
| `/video_feed`            | GET        | Luồng video từ camera |
| `/confirm`               | POST       | Xác nhận người dùng |
| `/start_counting`        | POST       | Bắt đầu đếm số lần gập bụng |
| `/continue_set`          | POST       | Tiếp tục sang set tập mới |
| `/change_user`           | POST       | Đổi người dùng |
| `/save`                  | POST       | Lưu số lần gập bụng vào database |
| `/status`                | GET        | Lấy trạng thái hiện tại |
| `/pause`                 | POST       | Tạm dừng đếm |
| `/resume`                | POST       | Tiếp tục đếm sau khi tạm dừng |
| `/logout`                | POST       | Đăng xuất người dùng |
| `/user_history`          | GET        | Xem lịch sử tập luyện của người dùng |

---

## **Ghi chú quan trọng**  
✅ **Kiểm tra webcam**: Đảm bảo webcam đang hoạt động trước khi chạy hệ thống.  
✅ **Chạy MySQL Server**: Hệ thống cần MySQL để lưu dữ liệu tập luyện.  
✅ **Điều chỉnh tham số**: Có thể thay đổi **thời gian chờ** và các tham số khác trong `app.py` để phù hợp với yêu cầu thực tế.  

---

## **Mô hình tổng quan của hệ thống**  
1️⃣ **Nhận diện người tập luyện** (📷 **Camera**)  
   - **Face Recognition** để xác thực người dùng  
   - Nếu xác thực thành công → Tiếp tục sang bước đếm số lần gập bụng  
   - Nếu thất bại → Cho đếm số lần gập bụng nhưng không lưu vào cơ sở dữ liệu kết quả 

2️⃣ **Nhận diện động tác gập bụng**  
   - Trích xuất **keypoints** từ video bằng **MediaPipe/OpenPose**  
   - Ghép thành dữ liệu chuỗi thời gian (**time series**)  
   - Đưa vào mô hình **LSTM** để đếm số lần gập bụng  

3️⃣ **Lưu kết quả vào MySQL**  
   - Lưu thông tin người tập, số lần gập bụng, thời gian tập luyện  
   - Hiển thị lịch sử tập luyện khi cần  

---
## **Poster**
![Poster_Nhom10](https://github.com/user-attachments/assets/5d03963d-0ab2-458f-8b09-0d8bb33275f1)

**🔥 Chúc bạn triển khai thành công dự án SitUpFaceLogin! 🔥** 🚀
