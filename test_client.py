from client_lib import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np

# Cac hang so
MAX_SPEED = 90
MIN_SPEED = 30
MAX_ANGLE = 25
KP = 0.32     # He so ti le (Proportional)
KI = 0.0015   # He so tich phan (Integral)
KD = 0.3     # He so dao ham (Derivative)
SPEED_DECAY = 6
Y_THRESHOLD = 95 # Toa do y toi thieu de phan tich anh
NUM_SLICES = 10  # So lat cat ROI

# Khoi tao bien PID
integral_error = 0
previous_deviation = 0

if __name__ == "__main__":
    try:
        while True:
            # Nhan trang thai va hinh anh tu xe
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()

            # Hien thi hinh anh de debug
            print(state)
            cv2.imshow('raw_image', raw_image)
            cv2.imshow('segment_image', segment_image)

            # Lay kich thuoc anh
            height, width = segment_image.shape[:2]
            car_center_x = width // 2

            # Xac dinh cac vi tri lat cat theo chieu doc
            y_positions = np.linspace(Y_THRESHOLD, height - 1, NUM_SLICES, dtype=int)

            lane_centers = []
            for y in y_positions:
                row = segment_image[y, :]
                white_pixels = np.where(row == 255)[0]
                if white_pixels.size > 0:
                    lane_centers.append((y, np.mean(white_pixels)))

            if lane_centers:
                # Lay tat ca toa do x cua cac tam duong
                xs = [c[1] for c in lane_centers]

                # Phat hien ngo re gia: loai bo diem lech nhieu so voi trung binh
                mean_x = np.mean(xs)
                filtered = [(y, x) for (y, x) in lane_centers if abs(x - mean_x) < width * 0.25]

                if filtered:
                    xs = [c[1] for c in filtered]

                    # Trung binh toa do x
                    lane_center_x = int(np.mean(xs))

                    # Do lech so voi tam xe
                    deviation = lane_center_x - car_center_x

                    # PID
                    angle_P = KP * deviation
                    integral_error += deviation
                    angle_I = KI * integral_error
                    derivative_error = deviation - previous_deviation
                    angle_D = KD * derivative_error

                    angle = angle_P + angle_I + angle_D
                    previous_deviation = deviation

                    # Gioi han goc
                    angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)

                    # Tinh toan toc do
                    speed = MAX_SPEED - SPEED_DECAY * abs(angle)
                    speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
                else:
                    # Khong co du lieu dang tin cay
                    angle, speed = 0, MIN_SPEED
                    integral_error = 0
                    previous_deviation = 0
            else:
                # Khong tim thay duong
                angle, speed = 0, MIN_SPEED
                integral_error = 0
                previous_deviation = 0

            # Dieu khien xe
            AVControl(speed, angle)

            # Thoat bang phim q
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()


# python test_client.py
