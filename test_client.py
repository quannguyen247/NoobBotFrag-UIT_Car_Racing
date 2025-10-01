from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np

# Các hằng số
MAX_SPEED = 90 
MIN_SPEED = 30
MAX_ANGLE = 25
# Hệ số PID cơ bản
KP0 = 0.30
KI0 = 0.0015
KD0 = 0.25 
# Tinh chỉnh cho Speed control
SPEED_DECAY = 5
# Lấy mẫu làn đường
Y_THRESHOLD = 92       
NUM_SLICES = 15        
MIN_BLOB_WIDTH = 8     

# Hệ số thích nghi (GIỮ NGUYÊN)
ALPHA = 0.01
BETA = 0.02
GAMMA = 0.01
SMOOTH = 0.9
INTEGRAL_LIMIT = 1000

# Tham số cải tiến Look-Ahead (GIỮ NGUYÊN)
LOOK_AHEAD_DISTANCE_Y = 30 
MIN_LOOK_AHEAD_Y_RATIO = 0.1 
POLYFIT_DEGREE = 2        

# Khoảng cách cơ bản so với lề phải (MARGIN CAO) (GIỮ NGUYÊN)
RIGHT_MARGIN_BASE = 60 
MIN_MARGIN = 50 
MAX_MARGIN = 70 

# Khởi tạo PID (GIỮ NGUYÊN)
integral_error = 0
previous_deviation = 0
Kp, Ki, Kd = KP0, KI0, KD0
deviation = 0.0
curvature = 0.0

# Biến để lưu trữ đường cong từ frame trước
last_poly = None 
last_car_ref_x = None
last_predicted_center = None 
last_curvature = 0.0 
angle = 0.0 

def create_binary_from_segment(segment_image):
    gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return binary

def find_white_blobs(row, min_width=MIN_BLOB_WIDTH):
    blobs = []
    in_blob = False
    start_x = -1
    for x, pixel_val in enumerate(row):
        if pixel_val == 255 and not in_blob:
            start_x = x
            in_blob = True
        elif pixel_val == 0 and in_blob:
            end_x = x
            if (end_x - start_x) >= min_width:
                blobs.append({'start': start_x, 'end': end_x, 'center': (start_x + end_x) / 2, 'width': end_x - start_x})
            in_blob = False
    if in_blob and (len(row) - start_x) >= min_width: 
        blobs.append({'start': start_x, 'end': len(row), 'center': (start_x + len(row)) / 2, 'width': len(row) - start_x})
    return blobs

# HÀM LOGIC CỐT LÕI: Phân tích các blob để chọn ra đường đi thẳng
def analyze_and_select_straight_lane(all_blobs_by_y, y_positions, width, height, last_poly):
    
    straight_lane_points = []
    
    # 1. Phát hiện ngã ba/ngã rẽ TỔNG THỂ (tìm kiếm 2 blob ở nửa gần xe)
    is_fork_detected = any(len([b for b in blobs if b['center'] > width // 2 - 50]) >= 2 
                           for y, blobs in all_blobs_by_y.items() if y > height * 0.7)

    for y in y_positions:
        blobs = [b for b in all_blobs_by_y[y] if b['center'] > width // 2 - 50] # Chỉ xét nửa phải
        
        if not blobs:
            continue
            
        chosen_blob_center = None
        
        # Lấy target_x từ poly trước (dùng làm mỏ neo)
        target_x = (np.polyval(last_poly, y) if last_poly is not None 
                    else width // 2 + RIGHT_MARGIN_BASE)
        TOLERANCE_X = 50 
        
        # 2. XỬ LÝ NGÃ BA: Ưu tiên đường thẳng (blob bên trái nhất)
        if is_fork_detected and len(blobs) >= 2:
            blobs.sort(key=lambda b: b['center'])
            
            # KIỂM TRA LẠI MỎ NEO: Blob trái nhất phải nằm gần đường dự đoán
            if abs(blobs[0]['center'] - target_x) < TOLERANCE_X * 1.5:
                 chosen_blob_center = blobs[0]['center']
            else:
                 # Nếu blob trái nhất quá xa đường dự đoán, sử dụng đường dự đoán cũ (giữ hướng)
                 chosen_blob_center = target_x 
            
        else:
            # 3. XỬ LÝ BÌNH THƯỜNG: Chọn blob gần đường dự đoán (Ưu tiên liên tục)
            min_dist = float('inf')
            best_blob = None
            
            for blob in blobs:
                # Lọc mạnh mẽ: phải nằm gần target_x
                if abs(blob['center'] - target_x) < TOLERANCE_X:
                    dist = abs(blob['center'] - target_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_blob = blob
            
            # Fallback nếu không tìm thấy blob gần target_x
            if best_blob:
                chosen_blob_center = best_blob['center']
            elif blobs:
                 chosen_blob_center = min(b['center'] for b in blobs) # Chọn trái nhất nếu không có mỏ neo

                
        if chosen_blob_center is not None:
            # CHỈ LƯU NHỮNG ĐIỂM NẰM TRONG VỊ TRÍ LỀ PHẢI AN TOÀN
            if chosen_blob_center < width * 0.95: 
                straight_lane_points.append((y, chosen_blob_center))

    return straight_lane_points


if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            segment_image = GetSeg()

            binary_lane = create_binary_from_segment(segment_image)
            cv2.imshow('binary_lane_raw', binary_lane) # Ảnh binary gốc

            height, width = binary_lane.shape[:2]
            
            v_state = state.get("speed", 30)
            y_look_ahead = int(height - 1 - LOOK_AHEAD_DISTANCE_Y * (MAX_SPEED / (v_state + 1e-6)) * 0.5)
            y_look_ahead = max(y_look_ahead, int(height * MIN_LOOK_AHEAD_Y_RATIO)) 
            
            y_positions = np.linspace(Y_THRESHOLD, height - 1, NUM_SLICES, dtype=int)
            
            # 1. GIAI ĐOẠN PHÁT HIỆN BLOB
            all_blobs_by_y = {}
            for y in y_positions:
                row = binary_lane[y, :]
                blobs = find_white_blobs(row) 
                all_blobs_by_y[y] = blobs
            
            # 2. GIAI ĐOẠN RA QUYẾT ĐỊNH & PHÂN TÍCH HƯỚNG LÁI
            lane_centers = analyze_and_select_straight_lane(all_blobs_by_y, y_positions, width, height, last_poly)
            
            # Khối PID và Điều khiển
            if lane_centers:
                xs = [c[1] for c in lane_centers]
                ys = [c[0] for c in lane_centers]

                if len(lane_centers) >= POLYFIT_DEGREE + 1: 
                    
                    # Hồi quy bậc 2
                    poly = np.polyfit(ys, xs, POLYFIT_DEGREE)
                    last_poly = poly 

                    # Cập nhật Curvature
                    curvature = abs(poly[0]) * 1e4 
                    last_curvature = curvature 
                    
                    # 💡 PHẦN MỚI: TÁI TẠO ẢNH DỰA TRÊN HỒI QUY ĐỂ DEBUG
                    clean_binary_image = np.zeros_like(binary_lane)
                    
                    # Vẽ lại đường cong đã hồi quy lên ảnh sạch
                    plot_y = np.linspace(Y_THRESHOLD, height - 1, 100)
                    plot_x = np.polyval(poly, plot_y).astype(int)
                    
                    for i in range(len(plot_y)):
                        y_int = int(plot_y[i])
                        x_center = plot_x[i]
                        # Vẽ đường trắng với độ dày nhất định (10 pixels)
                        cv2.line(clean_binary_image, (x_center, y_int), (x_center, y_int), (255), thickness=10)
                    
                    cv2.imshow('binary_lane_clean', clean_binary_image)
                    # KẾT THÚC PHẦN MỚI
                    
                    # 4. Tính toán car_ref_x (Điểm tham chiếu Look-Ahead)
                    margin_dynamic = RIGHT_MARGIN_BASE + int(curvature * 0.2)
                    margin_dynamic = np.clip(margin_dynamic, MIN_MARGIN, MAX_MARGIN)

                    predicted_lane_center_at_lookahead = np.polyval(poly, y_look_ahead)
                    last_predicted_center = predicted_lane_center_at_lookahead 

                    # car_ref_x đảm bảo bám lề phải
                    car_ref_x = int(predicted_lane_center_at_lookahead - margin_dynamic)
                    car_ref_x = max(car_ref_x, width // 2 - 20) 
                    last_car_ref_x = car_ref_x 

                    deviation = predicted_lane_center_at_lookahead - car_ref_x

                    # PID adaptive (Giữ nguyên)
                    target_Kp = KP0 / (1 + ALPHA * v_state) + min(0.002 * curvature, 0.1) 
                    target_Kp += min(abs(deviation) * 0.0015, 0.1)
                    target_Ki = KI0 / (1 + GAMMA * v_state)
                    target_Kd = KD0 * (1 + BETA * v_state) + min(0.005 * curvature, 0.15) 

                    Kp = SMOOTH * Kp + (1 - SMOOTH) * target_Kp
                    Ki = SMOOTH * Ki + (1 - SMOOTH) * target_Ki
                    Kd = SMOOTH * Kd + (1 - SMOOTH) * target_Kd

                    # PID
                    angle_P = Kp * deviation
                    integral_error += deviation
                    integral_error = np.clip(integral_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                    angle_I = Ki * integral_error
                    derivative_error = deviation - previous_deviation
                    angle_D = Kd * derivative_error

                    angle = angle_P + angle_I + angle_D
                    previous_deviation = deviation

                    angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)

                    # Tối ưu tốc độ (Giữ nguyên)
                    base_speed = MAX_SPEED - SPEED_DECAY * abs(angle)
                    
                    if curvature > 30 or abs(angle) > MAX_ANGLE * 0.5:
                        speed = base_speed * 0.5
                    elif curvature > 15 or abs(angle) > MAX_ANGLE * 0.3:
                        speed = base_speed * 0.7
                    else:
                        if curvature < 5 and abs(deviation) < 10:
                            speed = MAX_SPEED 
                        else:
                            speed = base_speed
                            
                    speed = np.clip(speed, MIN_SPEED, MAX_SPEED)

                else: 
                    # Fallback mềm: không đủ điểm
                    if last_poly is not None:
                        angle = np.polyval(last_poly, height - 1) / (width/2) * MAX_ANGLE * 0.5 
                    else:
                        angle = 0.0 
                    speed = MIN_SPEED
                    integral_error = 0
                    
            else: 
                # Fallback khi mất lane hoàn toàn
                if last_poly is not None and last_car_ref_x is not None:
                    predicted_angle_from_last_poly = (np.polyval(last_poly, y_look_ahead) - last_car_ref_x) / (width/2) * MAX_ANGLE
                    angle = np.clip(predicted_angle_from_last_poly * 0.5, -MAX_ANGLE, MAX_ANGLE) 
                else:
                    angle = 0.0 
                speed = MIN_SPEED
                integral_error = 0
            
            # IN THÔNG TIN I/O ĐIỀU KHIỂN
            print(f"Speed: {speed:.1f} | Angle: {angle:.1f} | Deviation: {deviation:.1f} | Curv: {curvature:.1f}")
            
            AVControl(speed, angle)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()

#       python test_client.py
