from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np

# Các hằng số
MAX_SPEED = 90
MIN_SPEED = 25
MAX_ANGLE = 25
KP0 = 0.32
KI0 = 0.0015
KD0 = 0.3
SPEED_DECAY = 5.5
Y_THRESHOLD = 92
NUM_SLICES = 10

# Hệ số thích nghi
ALPHA = 0.01
BETA = 0.02
GAMMA = 0.01
SMOOTH = 0.9
INTEGRAL_LIMIT = 1000

# Khoi tao PID
integral_error = 0
previous_deviation = 0
Kp, Ki, Kd = KP0, KI0, KD0

# Khoảng cách cơ bản so với lề phải
RIGHT_MARGIN_BASE = 20

def create_binary_from_segment(segment_image):
    gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return binary

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            segment_image = GetSeg()

            # Tạo ảnh binary và hiển thị
            binary_lane = create_binary_from_segment(segment_image)
            cv2.imshow('binary_lane', binary_lane)

            height, width = binary_lane.shape[:2]

            y_positions = np.linspace(Y_THRESHOLD, height - 1, NUM_SLICES, dtype=int)
            lane_centers = []
            for y in y_positions:
                row = binary_lane[y, :]
                white_pixels = np.where(row == 255)[0]
                if white_pixels.size > 0:
                    lane_centers.append((y, np.mean(white_pixels)))

            if lane_centers:
                xs = [c[1] for c in lane_centers]
                mean_x = np.mean(xs)
                filtered = [(y, x) for (y, x) in lane_centers if abs(x - mean_x) < width * 0.25]

                if filtered:
                    xs = [c[1] for c in filtered]

                    # Tính car_ref_x động bám lane phải
                    right_lane_xs = [x for x in xs if x > width // 2]
                    if right_lane_xs:
                        # Margin động dựa trên độ cong
                        if len(filtered) >= 3:
                            ys = [c[0] for c in filtered]
                            poly = np.polyfit(ys, xs, 2)
                            curvature = abs(poly[0]) * 1e4
                        else:
                            curvature = 0
                        margin_dynamic = RIGHT_MARGIN_BASE + int(curvature * 0.2)
                        car_ref_x = int(np.mean(right_lane_xs)) - margin_dynamic
                        car_ref_x = max(car_ref_x, width // 2)
                    else:
                        car_ref_x = width // 2
                        curvature = 0

                    lane_center_x = int(np.mean(xs))
                    deviation = lane_center_x - car_ref_x

                    # PID adaptive
                    v_state = state.get("speed", 30)
                    target_Kp = KP0 / (1 + ALPHA * v_state) + min(0.001 * curvature, 0.05)
                    # tăng Kp khi deviation lớn để quẹo nhanh
                    target_Kp += min(abs(deviation) * 0.0015, 0.1)
                    target_Ki = KI0 / (1 + GAMMA * v_state)
                    target_Kd = KD0 * (1 + BETA * v_state)

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

                    # Tối ưu tốc độ khi cua gấp
                    base_speed = MAX_SPEED - SPEED_DECAY * abs(angle)
                    if curvature > 30 or abs(angle) > MAX_ANGLE * 0.5:
                        speed = base_speed * 0.5
                    elif curvature > 15 or abs(angle) > MAX_ANGLE * 0.3:
                        speed = base_speed * 0.7
                    else:
                        speed = base_speed
                    speed = np.clip(speed, MIN_SPEED, MAX_SPEED)

                else:
                    # Fallback mềm
                    angle *= 0.7
                    speed = MIN_SPEED
                    integral_error = 0
            else:
                # Fallback khi mất lane
                angle *= 0.7
                speed = MIN_SPEED
                integral_error = 0

            AVControl(speed, angle)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()



#       python test_client.py
