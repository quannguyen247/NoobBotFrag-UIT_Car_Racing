from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np

MAX_SPEED = 90
MIN_SPEED = 45
MAX_ANGLE = 30
KP0 = 0.3
KI0 = 0.0015
KD0 = 0.3
SPEED_DECAY = 6
Y_THRESHOLD = 92       
NUM_SLICES = 15        
MIN_BLOB_WIDTH = 8     

ALPHA = 0.02
BETA = 0.02
GAMMA = 0.02
SMOOTH = 0.9
INTEGRAL_LIMIT = 1000

LOOK_AHEAD_DISTANCE_Y = 25
MIN_LOOK_AHEAD_Y_RATIO = 0.1 
POLYFIT_DEGREE = 2        

RIGHT_MARGIN_BASE = 60 
MIN_MARGIN = 50 
MAX_MARGIN = 70 

integral_error = 0
previous_deviation = 0
Kp, Ki, Kd = KP0, KI0, KD0

last_poly = None 
last_car_ref_x = None
last_predicted_center = None 
last_curvature = 0.0 
angle = 0.0 
speed = MIN_SPEED 

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

if __name__ == "__main__":
    try:
        frame_count = 0
        while True:
            frame_count += 1
            state = GetStatus()
            segment_image = GetSeg()

            binary_lane = create_binary_from_segment(segment_image)
            cv2.imshow('Binary_Image', binary_lane)

            height, width = binary_lane.shape[:2]
            
            v_state = state.get("speed", 30)
            y_look_ahead = int(height - 1 - LOOK_AHEAD_DISTANCE_Y * (MAX_SPEED / (v_state + 1e-6)) * 0.5)
            y_look_ahead = max(y_look_ahead, int(height * MIN_LOOK_AHEAD_Y_RATIO)) 
            
            y_positions = np.linspace(Y_THRESHOLD, height - 1, NUM_SLICES, dtype=int)
            
            all_blobs_by_y = {}
            for y in y_positions:
                row = binary_lane[y, :]
                blobs = find_white_blobs(row) 
                all_blobs_by_y[y] = blobs
            
            is_fork_detected = any(len([b for b in blobs if b['center'] > width // 2 - 50]) >= 2 
                                   for blobs in all_blobs_by_y.values())
            
            lane_centers = []
            car_center_x = width // 2
            predicted_x_from_last = last_predicted_center if last_predicted_center is not None else car_center_x + RIGHT_MARGIN_BASE
            
            for y in y_positions:
                blobs = all_blobs_by_y[y]
                chosen_blob_center = None
                
                if blobs:
                    if last_poly is not None:
                        target_x = np.polyval(last_poly, y)
                    else:
                        target_x = predicted_x_from_last 
                    
                    TOLERANCE_X = 50 
                    
                    potential_blobs = [b for b in blobs if 
                                       abs(b['center'] - target_x) < TOLERANCE_X and 
                                       b['center'] > width // 2 - 50] 
                    
                    if is_fork_detected: 
                        right_blobs = [b for b in blobs if b['center'] > width // 2 - 50]
                        
                        if len(right_blobs) >= 2:
                            right_blobs.sort(key=lambda b: b['center'])
                            best_blob = right_blobs[0]
                            chosen_blob_center = best_blob['center']
                            lane_centers.append((y, chosen_blob_center))
                            continue
                            
                        elif len(right_blobs) == 1:
                             chosen_blob_center = right_blobs[0]['center']

                    if potential_blobs:
                        min_dist = float('inf')
                        best_blob = None
                        
                        for blob in potential_blobs:
                            dist = abs(blob['center'] - target_x)
                            if dist < min_dist:
                                min_dist = dist
                                best_blob = blob
                        
                        if best_blob:
                            chosen_blob_center = best_blob['center']
                            
                    elif chosen_blob_center is None and blobs:
                        chosen_blob_center = max(blobs, key=lambda b: b['center'])['center']
                
                if chosen_blob_center is not None:
                    lane_centers.append((y, chosen_blob_center))
                    
            if lane_centers:
                xs = [c[1] for c in lane_centers]
                ys = [c[0] for c in lane_centers]

                if len(lane_centers) >= POLYFIT_DEGREE + 1: 
                    
                    poly = np.polyfit(ys, xs, POLYFIT_DEGREE)
                    last_poly = poly 

                    curvature = abs(poly[0]) * 1e4 
                    last_curvature = curvature 
                    
                    predicted_lane_center_at_lookahead = np.polyval(poly, y_look_ahead)
                    last_predicted_center = predicted_lane_center_at_lookahead 

                    deviation = predicted_lane_center_at_lookahead - car_center_x 
                    last_car_ref_x = car_center_x 

                    # PID adaptive 
                    target_Kp = KP0 / (1 + ALPHA * v_state) + min(0.002 * curvature, 0.1) 
                    target_Kp += min(abs(deviation) * 0.0015, 0.1)
                    target_Ki = KI0 / (1 + GAMMA * v_state)
                    target_Kd = KD0 * (1 + BETA * v_state) + min(0.005 * curvature, 0.15) 

                    Kp = SMOOTH * Kp + (1 - SMOOTH) * target_Kp
                    Ki = SMOOTH * Ki + (1 - SMOOTH) * target_Ki
                    Kd = SMOOTH * Kd + (1 - SMOOTH) * target_Kd

                    angle_P = Kp * deviation
                    integral_error += deviation
                    integral_error = np.clip(integral_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                    angle_I = Ki * integral_error
                    derivative_error = deviation - previous_deviation
                    angle_D = Kd * derivative_error

                    angle = angle_P + angle_I + angle_D
                    previous_deviation = deviation

                    angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)

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
                    if last_poly is not None:
                        angle = np.polyval(last_poly, height - 1) / (width/2) * MAX_ANGLE * 0.5 
                    else:
                        angle = 0.0
                    speed = MIN_SPEED
                    integral_error = 0
                    
            else: 
                if last_poly is not None and last_car_ref_x is not None:
                    predicted_lane_center_from_last_poly = np.polyval(last_poly, y_look_ahead)
                    deviation_fallback = predicted_lane_center_from_last_poly - car_center_x
                    angle = np.clip(deviation_fallback / (width/2) * MAX_ANGLE * 0.5, -MAX_ANGLE, MAX_ANGLE) 
                else:
                    angle = 0.0
                speed = MIN_SPEED
                integral_error = 0
                      
            AVControl(speed, angle)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('\nClosing socket')
        CloseSocket()
        cv2.destroyAllWindows()