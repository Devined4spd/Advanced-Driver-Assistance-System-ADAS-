import cv2
import numpy as np
import math
from ultralytics import YOLO

# ------------------------------
# YOLO MODEL
# ------------------------------
model = YOLO("yolov8n.pt")  # small & fast

# COCO vehicle-related classes: car, bus, truck, bike, etc.
VEHICLE_CLASSES = {1, 2, 3, 5, 7}
XM_PER_PIX = 3.7 / 700.0   # approx lane width scale


# ------------------------------
# SIMPLE MULTI-OBJECT VEHICLE TRACKER
# ------------------------------
class VehicleTracker:
    """
    Very simple centroid-based tracker.
    - Assigns IDs to detected vehicles
    - Keeps track of each car across frames
    - Estimates TTC using change in bounding-box height
    """

    def __init__(self, max_distance=80, max_missed=10):
        self.next_id = 1
        self.tracks = {}  # id -> track dict
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections, fps):
        """
        detections: list of dicts {'bbox':(x1,y1,x2,y2), 'cx':cx, 'cy':cy}
        returns: list of track dicts
        """
        updated_tracks = {}
        used_tracks = set()
        used_dets = set()

        # 1. Associate detections to existing tracks (nearest neighbor)
        for det_idx, det in enumerate(detections):
            cx, cy = det['cx'], det['cy']
            best_id = None
            best_dist = float("inf")
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                dist = math.hypot(cx - tr['cx'], cy - tr['cy'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist < self.max_distance:
                # Update this track
                tr = self.tracks[best_id]
                x1, y1, x2, y2 = det['bbox']
                h = max(1.0, y2 - y1)
                inv_h = 1.0 / h
                ttc = None

                # TTC estimation (using change in 1/h, independent of absolute scale)
                if tr['last_inv_h'] is not None and fps > 0:
                    delta_inv = tr['last_inv_h'] - inv_h
                    if delta_inv > 0:
                        ttc = inv_h / (delta_inv * fps)  # seconds
                        if ttc < 0 or ttc > 20:  # ignore crazy values
                            ttc = None

                tr.update({
                    'cx': cx,
                    'cy': cy,
                    'bbox': det['bbox'],
                    'last_inv_h': inv_h,
                    'ttc': ttc,
                    'missed': 0,
                    'age': tr['age'] + 1,
                })
                updated_tracks[best_id] = tr
                used_tracks.add(best_id)
                used_dets.add(det_idx)

        # 2. Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            x1, y1, x2, y2 = det['bbox']
            h = max(1.0, y2 - y1)
            inv_h = 1.0 / h
            tid = self.next_id
            self.next_id += 1
            updated_tracks[tid] = {
                'id': tid,
                'cx': det['cx'],
                'cy': det['cy'],
                'bbox': det['bbox'],
                'last_inv_h': inv_h,
                'ttc': None,
                'age': 1,
                'missed': 0,
            }

        # 3. Increase "missed" count for tracks not updated
        for tid, tr in self.tracks.items():
            if tid not in updated_tracks:
                tr['missed'] += 1
                if tr['missed'] <= self.max_missed:
                    updated_tracks[tid] = tr

        # 4. Remove very old tracks
        self.tracks = {
            tid: tr for tid, tr in updated_tracks.items()
            if tr['missed'] <= self.max_missed
        }

        return list(self.tracks.values())


# ------------------------------
# LANE DETECTION FUNCS
# ------------------------------
def canny_edge_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.4 * width), int(0.6 * height))
    ]])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines_and_offset(image, lines):
    """
    Draw lane lines AND estimate vehicle offset from lane center.
    Returns:
        line_image
        lane_offset_m (+right, -left)
        left_line, right_line
    """
    line_image = np.zeros_like(image)
    lane_offset_m = None
    left_line = None
    right_line = None

    if lines is None:
        return line_image, lane_offset_m, left_line, right_line

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -0.5:
            left_lines.append((slope, intercept))
        elif slope > 0.5:
            right_lines.append((slope, intercept))

    left_line = average_slope_intercept(image, left_lines)
    right_line = average_slope_intercept(image, right_lines)

    height = image.shape[0]
    width = image.shape[1]

    # Draw lane lines
    for line in [left_line, right_line]:
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # Offset calc
    if left_line is not None and right_line is not None:
        left_bottom_x = left_line[0]
        right_bottom_x = right_line[0]

        lane_center_x = (left_bottom_x + right_bottom_x) / 2.0
        vehicle_center_x = width / 2.0

        offset_pixels = vehicle_center_x - lane_center_x
        lane_offset_m = offset_pixels * XM_PER_PIX

        y_bottom = height - 5
        cv2.circle(line_image, (int(vehicle_center_x), y_bottom), 6, (255, 0, 0), -1)
        cv2.circle(line_image, (int(lane_center_x), y_bottom), 6, (0, 255, 255), -1)

    return line_image, lane_offset_m, left_line, right_line


def make_coordinates(image, line_params):
    slope, intercept = line_params
    height = image.shape[0]

    y1 = height
    y2 = int(height * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, line_parameters):
    if len(line_parameters) == 0:
        return None
    line_parameters = np.array(line_parameters)
    slope = np.mean(line_parameters[:, 0])
    intercept = np.mean(line_parameters[:, 1])
    return make_coordinates(image, (slope, intercept))


# ------------------------------
# STEERING ANGLE
# ------------------------------
def compute_steering_angle(image, left_line, right_line):
    height, width, _ = image.shape
    if left_line is None or right_line is None:
        return None

    y_lookahead = int(height * 0.7)

    def line_x_at_y(line):
        x1, y1, x2, y2 = line
        if y2 - y1 == 0:
            return (x1 + x2) / 2
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y_lookahead - y1)

    left_x = line_x_at_y(left_line)
    right_x = line_x_at_y(right_line)
    lane_center_x = (left_x + right_x) / 2.0

    vehicle_center_x = width / 2.0

    dx = lane_center_x - vehicle_center_x
    dy = height - y_lookahead

    if dy == 0:
        return None

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def draw_steering_direction(image, angle_deg):
    if angle_deg is None:
        return image

    height, width, _ = image.shape
    length = 150

    angle_rad = math.radians(angle_deg)
    x_center = width // 2
    y_bottom = height - 10

    x_end = int(x_center + length * math.sin(angle_rad))
    y_end = int(y_bottom - length * math.cos(angle_rad))

    cv2.line(image, (x_center, y_bottom), (x_end, y_end), (255, 165, 0), 4)
    return image


# ------------------------------
# CURVATURE
# ------------------------------
def measure_lane_curvature(cropped_edges):
    ys, xs = np.nonzero(cropped_edges)
    if len(xs) < 500:
        return None

    height, width = cropped_edges.shape
    midpoint = width // 2

    left_mask = xs < midpoint
    right_mask = xs >= midpoint

    left_x = xs[left_mask]
    left_y = ys[left_mask]
    right_x = xs[right_mask]
    right_y = ys[right_mask]

    if len(left_x) < 100 or len(right_x) < 100:
        return None

    ym_per_pix = 30 / 720.0
    xm_per_pix = XM_PER_PIX

    left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)

    y_eval = height * ym_per_pix

    def radius_of_curvature(poly_coeffs, y_eval_m):
        A, B, _ = poly_coeffs
        return (1 + (2 * A * y_eval_m + B) ** 2) ** 1.5 / abs(2 * A)

    left_curverad = radius_of_curvature(left_fit_cr, y_eval)
    right_curverad = radius_of_curvature(right_fit_cr, y_eval)

    curvature_m = (left_curverad + right_curverad) / 2.0
    return curvature_m


# ------------------------------
# ROUGH AVERAGE SPEED (same as before)
# ------------------------------
def estimate_average_speed(prev_centers, curr_centers, fps, xm_per_pix=XM_PER_PIX):
    if fps <= 0 or not prev_centers or not curr_centers:
        return None

    used_prev = set()
    total_speed = 0.0
    matches = 0

    for cx, cy in curr_centers:
        best_idx = -1
        best_dist = float("inf")
        for i, (px, py) in enumerate(prev_centers):
            if i in used_prev:
                continue
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx != -1 and best_dist < 150:
            used_prev.add(best_idx)
            dist_m = best_dist * xm_per_pix
            speed_m_s = dist_m * fps
            speed_kmh = speed_m_s * 3.6
            total_speed += speed_kmh
            matches += 1

    if matches == 0:
        return None

    return total_speed / matches


# ------------------------------
# MAIN
# ------------------------------
def main():
    video_path = "test_video.mp4"
    # Or webcam:
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video. Check path or webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    prev_centers = []
    tracker = VehicleTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        # Lane detection
        edges = canny_edge_detector(frame)
        cropped_edges = region_of_interest(edges)

        lines = cv2.HoughLinesP(
            cropped_edges,
            rho=2,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=150
        )
        if lines is not None:
            lines = lines.reshape(-1, 4)

        line_image, lane_offset_m, left_line, right_line = display_lines_and_offset(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # Curvature
        curvature_m = measure_lane_curvature(cropped_edges)
        if curvature_m is not None:
            if curvature_m > 3000:
                curv_text = "Lane Curvature: Straight"
            else:
                curv_text = f"Lane Curvature: {curvature_m:.1f} m"
        else:
            curv_text = "Lane Curvature: N/A"

        cv2.putText(
            combo_image,
            curv_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Lane Departure Warning
        ldw_text = "Lane Position: N/A"
        ldw_color = (0, 255, 0)

        if lane_offset_m is not None:
            offset_abs = abs(lane_offset_m)
            dir_text = "RIGHT" if lane_offset_m > 0 else "LEFT"
            ldw_text = f"Offset: {offset_abs:.2f} m to {dir_text}"

            if offset_abs > 0.4:
                warn_text = f"LANE DEPARTURE WARNING! ({dir_text})"
                ldw_color = (0, 0, 255)

                overlay = combo_image.copy()
                cv2.rectangle(overlay, (0, 0), (combo_image.shape[1], 80), (0, 0, 255), -1)
                alpha = 0.3
                combo_image = cv2.addWeighted(overlay, alpha, combo_image, 1 - alpha, 0)

                cv2.putText(
                    combo_image,
                    warn_text,
                    (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA
                )

        cv2.putText(
            combo_image,
            ldw_text,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            ldw_color,
            2,
            cv2.LINE_AA
        )

        # Steering angle
        steering_angle = compute_steering_angle(combo_image, left_line, right_line)
        if steering_angle is not None:
            direction = "RIGHT" if steering_angle > 0 else "LEFT"
            steering_text = f"Steering Angle: {abs(steering_angle):.1f}° {direction}"
        else:
            steering_text = "Steering Angle: N/A"

        combo_image = draw_steering_direction(combo_image, steering_angle)

        cv2.putText(
            combo_image,
            steering_text,
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (173, 255, 47),
            2,
            cv2.LINE_AA
        )

        # -------------------------
        # YOLO + ID TRACKING + TTC
        # -------------------------
        results = model(frame, verbose=False)[0]
        detections = []
        centers = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if cls_id in VEHICLE_CLASSES:
                detections.append({'bbox': (x1, y1, x2, y2), 'cx': cx, 'cy': cy})
                centers.append((cx, cy))

        tracks = tracker.update(detections, fps)
        vehicle_count = len(tracks)

        for tr in tracks:
            x1, y1, x2, y2 = tr['bbox']
            tid = tr['id']
            ttc = tr['ttc']

            cv2.rectangle(
                combo_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )

            if ttc is not None:
                ttc_text = f"TTC: {ttc:.1f}s"
            else:
                ttc_text = "TTC: --"

            text = f"ID {tid} | {ttc_text}"
            cv2.putText(
                combo_image,
                text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            combo_image,
            f"Vehicles tracked: {vehicle_count}",
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Average speed (same rough logic as before)
        avg_speed_kmh = estimate_average_speed(prev_centers, centers, fps, XM_PER_PIX)
        prev_centers = centers

        if avg_speed_kmh is not None:
            speed_text = f"Avg vehicle speed: {avg_speed_kmh:.1f} km/h"
        else:
            speed_text = "Avg vehicle speed: N/A"

        cv2.putText(
            combo_image,
            speed_text,
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("ADAS with Tracking & TTC", combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
