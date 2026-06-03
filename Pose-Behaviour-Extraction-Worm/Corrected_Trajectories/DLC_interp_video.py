import os
import sys
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd


# =============================================================================
# 1. ENVIRONMENT & VIDEO METADATA UTILITIES
# =============================================================================

def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    conda_bin = os.path.join(sys.prefix, "bin", "ffmpeg")
    if os.path.isfile(conda_bin):
        return conda_bin
    for guess in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/ffmpeg/bin/ffmpeg"]:
        if os.path.isfile(guess):
            return guess
    raise RuntimeError(
        "ffmpeg not found.\n"
        "Install with:  conda install -c conda-forge ffmpeg -y"
    )


FFMPEG = find_ffmpeg()


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cv2 cannot open video:\n  {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if w == 0 or h == 0:
        raise IOError(f"cv2 reported 0x0 resolution for:\n  {video_path}")
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    return w, h, fps, n


# =============================================================================
# 2. FFMPEG PIPE READER (Bypasses OpenCV Codec Jumps)
# =============================================================================

class FfmpegFrameReader:
    def __init__(self, video_path, start_frame, native_w, native_h):
        self.video_path = video_path
        self.start_frame = start_frame
        self.native_w = native_w
        self.native_h = native_h
        self._proc = None

    def __enter__(self):
        cmd = [FFMPEG, "-loglevel", "error", "-i", self.video_path]
        if self.start_frame > 0:
            cmd += ["-vf", f"select=gte(n\\,{self.start_frame})", "-vsync", "0"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)
        self._frame_size = self.native_w * self.native_h * 3
        return self

    def __iter__(self):
        while True:
            raw = self._proc.stdout.read(self._frame_size)
            if len(raw) < self._frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.native_h, self.native_w, 3))
            yield frame.copy()

    def __exit__(self, *_):
        if self._proc:
            self._proc.stdout.close()
            self._proc.terminate()
            self._proc.wait()


# def get_interpolated_p3(df, idx, good_indices):
#     print()


def find_good_anchors(df, bad_idx, distance_threshold):
    """
    Scans backward and forward from a bad index.
    Returns (start_anchor, end_anchor) frame indices based purely on your distance threshold helper.
    """
    total_frames = len(df)
    idx_a = None
    idx_b = None

    # 1. Scan backward for Start Anchor
    search_back = bad_idx - 1
    while search_back >= 0:
        x2 = df.loc[search_back, ('point_2', 'x')]
        y2 = df.loc[search_back, ('point_2', 'y')]
        x3 = df.loc[search_back, ('point_3', 'x')]
        y3 = df.loc[search_back, ('point_3', 'y')]

        # Use your distance/NaN helper directly
        if is_distance_valid(x2, y2, x3, y3, distance_threshold):
            idx_a = search_back
            break
        search_back -= 1

    # 2. Scan forward for End Anchor
    search_forward = bad_idx + 1
    while search_forward < total_frames:
        x2 = df.loc[search_forward, ('point_2', 'x')]
        y2 = df.loc[search_forward, ('point_2', 'y')]
        x3 = df.loc[search_forward, ('point_3', 'x')]
        y3 = df.loc[search_forward, ('point_3', 'y')]

        # Use your distance/NaN helper directly
        if is_distance_valid(x2, y2, x3, y3, distance_threshold):
            idx_b = search_forward
            break
        search_forward += 1

    return idx_a, idx_b

# def get_interpolated_p3(df, target_idx, idx_a, idx_b):
#     """
#     Interpolates Point 3 relative to Point 2 by blending the distance
#     and global vector angle from the nearest valid anchor frames.
#     """
#     # Edge case safety handling for video boundaries
#     if idx_a is None and idx_b is None:
#         return df.loc[target_idx, ('point_3', 'x')], df.loc[target_idx, ('point_3', 'y')]
#     elif idx_a is None:
#         return df.loc[idx_b, ('point_3', 'x')], df.loc[idx_b, ('point_3', 'y')]
#     elif idx_b is None:
#         return df.loc[idx_a, ('point_3', 'x')], df.loc[idx_a, ('point_3', 'y')]
#
#     # Calculate progress fraction (f) between anchor A and anchor B
#     f = (target_idx - idx_a) / (idx_b - idx_a)
#
#     # 1. Extract vector properties for Start Anchor (A)
#     x2_a = df.loc[idx_a, ('point_2', 'x')]
#     y2_a = df.loc[idx_a, ('point_2', 'y')]
#     x3_a = df.loc[idx_a, ('point_3', 'x')]
#     y3_a = df.loc[idx_a, ('point_3', 'y')]
#
#     r_a = np.hypot(x3_a - x2_a, y3_a - y2_a)          # Start distance
#     theta_a = np.arctan2(y3_a - y2_a, x3_a - x2_a)    # Start angle
#
#     # 2. Extract vector properties for End Anchor (B)
#     x2_b = df.loc[idx_b, ('point_2', 'x')]
#     y2_b = df.loc[idx_b, ('point_2', 'y')]
#     x3_b = df.loc[idx_b, ('point_3', 'x')]
#     y3_b = df.loc[idx_b, ('point_3', 'y')]
#
#     r_b = np.hypot(x3_b - x2_b, y3_b - y2_b)          # End distance
#     theta_b = np.arctan2(y3_b - y2_b, x3_b - x2_b)    # End angle
#
#     # 3. Interpolate distance and angle
#     r_t = r_a + f * (r_b - r_a)
#
#     # Safe angular delta calculation to prevent boundary clipping wraps
#     d_theta = np.arctan2(np.sin(theta_b - theta_a), np.cos(theta_b - theta_a))
#     theta_t = theta_a + f * d_theta
#
#     # 4. Get current frame's baseline P2 coordinates
#     x2_t = df.loc[target_idx, ('point_2', 'x')]
#     y2_t = df.loc[target_idx, ('point_2', 'y')]
#
#     # 5. Project final tracking coordinates out from the current P2
#     interp_x = x2_t + r_t * np.cos(theta_t)
#     interp_y = y2_t + r_t * np.sin(theta_t)
#
#     return interp_x, interp_y


def get_interpolated_p3(df, target_idx, idx_a, idx_b):
    """
    Interpolates Point 3 by tracking the perpendicular distance and local angle
    of the P2 -> P3 segment relative to the P1 -> P2 body axis at both the
    start range (idx_a) and end range (idx_b).
    """
    # Edge case handling for video boundaries
    if idx_a is None and idx_b is None:
        return df[('point_3', 'x')].values[target_idx], df[('point_3', 'y')].values[target_idx]
    elif idx_a is None:
        idx_a = idx_b
    elif idx_b is None:
        idx_b = idx_a

    # Calculate progress fraction (f) between start and end range
    f = (target_idx - idx_a) / (idx_b - idx_a) if idx_b != idx_a else 0.0

    # Internal helper to explicitly measure P2 -> P3 relative to P1 -> P2
    def calculate_relative_geometry(idx):
        x1 = df[('point_1', 'x')].values[idx]
        y1 = df[('point_1', 'y')].values[idx]
        x2 = df[('point_2', 'x')].values[idx]
        y2 = df[('point_2', 'y')].values[idx]
        x3 = df[('point_3', 'x')].values[idx]
        y3 = df[('point_3', 'y')].values[idx]

        # 1. Heading angle of the baseline body axis (P1 -> P2)
        theta_body = np.arctan2(y2 - y1, x2 - x1)

        # 2. Vector and length of the tail segment (P2 -> P3)
        tx, ty = x3 - x2, y3 - y2
        r_tail = np.hypot(tx, ty)
        theta_tail_global = np.arctan2(ty, tx)

        # 3. Calculate local angle of P2 -> P3 relative to the P1 -> P2 direction
        local_angle = theta_tail_global - theta_body
        local_angle = np.arctan2(np.sin(local_angle), np.cos(local_angle)) # Normalize between -pi and +pi

        # 4. Calculate exact perpendicular distance of P3 from the body axis line
        perp_distance = r_tail * np.sin(local_angle)

        return r_tail, local_angle, perp_distance

    # --- STEP 1: Calculate P2 -> P3 properties at START good range (idx_a) ---
    r_a, local_angle_a, perp_dist_a = calculate_relative_geometry(idx_a)


    # --- STEP 2: Calculate P2 -> P3 properties at END good range (idx_b) ---
    r_b, local_angle_b, perp_dist_b = calculate_relative_geometry(idx_b)

    # Safety fallback: If one anchor frame contains NaN parameters, mirror the good one
    if np.isnan(r_a) or np.isnan(local_angle_a):
        r_a, local_angle_a, perp_dist_a = r_b, local_angle_b, perp_dist_b
    if np.isnan(r_b) or np.isnan(local_angle_b):
        r_b, local_angle_b, perp_dist_b = r_a, local_angle_a, perp_dist_a
    if np.isnan(r_a):
        return df[('point_3', 'x')].values[target_idx], df[('point_3', 'y')].values[target_idx]

    # --- STEP 3: Linearly blend properties across the gap ---
    # Linearly blending tail length (r) and local angle preserves the exact
    # perpendicular distance and relative angle smoothly at both boundary frames.
    r_t = r_a + f * (r_b - r_a)

    delta_angle = np.arctan2(np.sin(local_angle_b - local_angle_a), np.cos(local_angle_b - local_angle_a))
    local_angle_t = local_angle_a + f * delta_angle

    # --- STEP 4: Grab the current frame's live body position and heading ---
    x1_t = df[('point_1', 'x')].values[target_idx]
    y1_t = df[('point_1', 'y')].values[target_idx]
    x2_t = df[('point_2', 'x')].values[target_idx]
    y2_t = df[('point_2', 'y')].values[target_idx]

    theta_body_t = np.arctan2(y2_t - y1_t, x2_t - x1_t)

    if np.isnan(theta_body_t) or np.isnan(x2_t):
        return df[('point_3', 'x')].values[target_idx], df[('point_3', 'y')].values[target_idx]

    # --- STEP 5: Reconstruct global tail angle and project outward from current P2 ---
    theta_tail_t = theta_body_t + local_angle_t

    interp_x = x2_t + r_t * np.cos(theta_tail_t)
    interp_y = y2_t + r_t * np.sin(theta_tail_t)

    return interp_x, interp_y

def is_distance_valid(x2, y2, x3, y3, distance_threshold):
    """
    Helper function to check if the physical distance between P2 and P3
    is within the acceptable threshold boundary.
    """
    # If any coordinates are missing (NaN), it's automatically invalid
    if np.isnan(x2) or np.isnan(y2) or np.isnan(x3) or np.isnan(y3):
        return False

    # Calculate direct Euclidean distance
    distance = np.hypot(x3 - x2, y3 - y2)

    # Returns True if within threshold, False if it crosses it
    return (distance <= distance_threshold and distance>=70)
# =============================================================================
# 3. GRAPHICS UTILITIES
# =============================================================================

def draw_frame_label(frame, frame_number, h):
    text = f"Frame: {frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0, y0 = 20, h - 20
    pad = 6
    cv2.rectangle(frame, (x0 - pad, y0 - th - pad), (x0 + tw + pad, y0 + baseline + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, y0), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def update_and_save_dlc_csv(df, df_sliced, start_frame, end_frame, csv_path, output_csv_path):
    """
    Maps the sliced data back to the main DataFrame and saves it,
    preserving the 3-row DeepLabCut structure.
    """
    # 1. Map the modified slice back into the original untouched full dataframe
    df.iloc[start_frame:end_frame] = df_sliced.values

    # 2. Grab the very first line (the 'scorer' row) from the original file
    with open(csv_path, 'r') as f:
        scorer_header = f.readline().strip()

    # 3. Write the first line out, then let pandas append the rest of the df
    with open(output_csv_path, 'w') as f:
        f.write(scorer_header + "\n")
        df.to_csv(f, index=False)

    print(f"[csv] Saved successfully to: {output_csv_path}")

# =============================================================================
# 4. CORE EXECUTOR
# =============================================================================

def generate_raw_labeled_video(root_dir, csv_filename, raw_video_filename, output_filename,
                               start_frame=0, end_frame=None, likelihood_threshold=0.001,distance_threshold = 350.0):
    csv_path = os.path.join(root_dir, csv_filename)
    video_path = os.path.join(root_dir, raw_video_filename)
    out_path = os.path.join(root_dir, output_filename)

    native_w, native_h, fps, total_frames = get_video_info(video_path)
    print(f"[video] {native_w}x{native_h} | {fps:.2f} FPS")

    # Read the multi-level DLC header format
    df = pd.read_csv(csv_path, header=[1, 2])
    if end_frame is None or end_frame > len(df):
        end_frame = len(df)

    df_sliced = df.iloc[start_frame:end_frame].copy().reset_index(drop=True)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (native_w, native_h))

    COLOR_MAP = {'point_1': (150, 0, 150), 'point_2': (180, 255, 130), 'point_3': (0, 0, 255)}
    LABELS = {'point_1': 'P1', 'point_2': 'P2', 'point_3': 'P3'}

    print(f"[render] Writing video output using raw DLC coordinates...")
    with FfmpegFrameReader(video_path, start_frame, native_w, native_h) as reader:
        for idx, frame in enumerate(reader):
            if idx >= len(df_sliced):
                break

            # --- RUN DISTANCE THRESHOLD CHECK FOR THE CURRENT FRAME ---
            x2_val = df_sliced[('point_2', 'x')].values[idx]
            y2_val = df_sliced[('point_2', 'y')].values[idx]
            x3_val = df_sliced[('point_3', 'x')].values[idx]
            y3_val = df_sliced[('point_3', 'y')].values[idx]
            print("x2 val: y2_val", x2_val, y2_val)

            p3_is_valid = is_distance_valid(x2_val, y2_val, x3_val, y3_val, distance_threshold)

            # 1. Collect valid consecutive points to draw skeleton lines
            pts = []
            for bp in ['point_1', 'point_2', 'point_3']:
                xv = df_sliced[(bp, 'x')].values[idx]
                yv = df_sliced[(bp, 'y')].values[idx]
                p_lh = df_sliced[(bp, 'likelihood')].values[idx]

                # 2. Extract specific points
                p1 = (int(df_sliced[('point_1', 'x')].values[idx]), int(df_sliced[('point_1', 'y')].values[idx]))
                p2 = (int(df_sliced[('point_2', 'x')].values[idx]), int(df_sliced[('point_2', 'y')].values[idx]))
                p3 = (int(df_sliced[('point_3', 'x')].values[idx]), int(df_sliced[('point_3', 'y')].values[idx]))

                # 3. Calculate the midpoint of P1 and P2
                mid_p1p2 = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

                # Only draw lines through points above likelihood threshold
                if not (np.isnan(xv) or np.isnan(yv)) and p_lh >= likelihood_threshold:
                    # EXTRA CHECK: If this is point_3 and it failed the distance threshold, skip it
                    if bp == 'point_3' and not p3_is_valid:
                        start_anchor, end_anchor = find_good_anchors(df_sliced, idx, distance_threshold)
                        xv, yv = get_interpolated_p3(df_sliced, idx, start_anchor, end_anchor)
                        # xv = 200.0
                        # yv = 200.0
                        #continue
                    pts.append((int(round(xv)), int(round(yv))))
            # add transparent line
            alpha = 0.5
            line_color = (200, 200, 200)
            for i in range(len(pts) - 1):
                #cv2.line(frame, pts[i], pts[i + 1], (200, 200, 200), 1, cv2.LINE_AA)
                overlay = frame.copy()

                if i==1:
                    mid_p1p2 = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2)
                    cv2.line(overlay, mid_p1p2, pts[i+1] , line_color, 2, cv2.LINE_AA)
                else:
                    cv2.line(overlay, pts[i], pts[i + 1], line_color, 2, cv2.LINE_AA)

                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 2. Draw Individual Tracking Nodes
            for bodypart, color in COLOR_MAP.items():
                x = df_sliced[(bodypart, 'x')].values[idx]
                y = df_sliced[(bodypart, 'y')].values[idx]
                lh = df_sliced[(bodypart, 'likelihood')].values[idx]

                # If DLC missed the point or confidence is too low, skip drawing it
                if np.isnan(x) or np.isnan(y) or lh < likelihood_threshold:
                    continue

                # EXTRA CHECK: If this is point_3 and it failed the distance threshold, skip drawing it
                if bodypart == 'point_3' and not p3_is_valid:
                    start_anchor, end_anchor = find_good_anchors(df_sliced, idx, distance_threshold)
                    x, y = get_interpolated_p3(df_sliced, idx, start_anchor, end_anchor)
                    # UPDATE 1B: Fallback visual mapping assignment
                    # ==========================================
                    df_sliced.loc[idx, ('point_3', 'x')] = x
                    df_sliced.loc[idx, ('point_3', 'y')] = y
                    #continue

                cx, cy = int(round(x)), int(round(y))
                cv2.circle(frame, (cx, cy), 3, color, -1)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)
                cv2.putText(frame, LABELS[bodypart], (cx + 10, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            draw_frame_label(frame, start_frame + idx, native_h)
            out.write(frame)

    out.release()
    print(f"[done] Video execution finished successfully.")

    name_part, extension_part = os.path.splitext(csv_filename)
    output_csv_path = os.path.join(root_dir, f"corrected_{name_part}{extension_part}")
    update_and_save_dlc_csv(df, df_sliced, start_frame, end_frame, csv_path, output_csv_path)


if __name__ == "__main__":
    ROOT_DIR = "/gpfs/soma_fs/home/pyaasa/Desktop/Test_Data"
    CSV_FILE = "20252606-16-11Expnt001DLC_Resnet50_tracking_20252606-16-11Expnt001May22shuffle1_snapshot_best-90.csv"

    generate_raw_labeled_video(
        root_dir=ROOT_DIR,
        csv_filename=CSV_FILE,
        raw_video_filename="20252606-16-11Expnt001.mp4",
        output_filename="interpolated_dlc_tracking_distance.mp4",
        start_frame=0,
        end_frame=8000,
        likelihood_threshold=0.001,
        distance_threshold=250.0

    )