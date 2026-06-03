"""
Worm Tracking — Custom Labeled Video Generator
================================================
Reads raw unlabeled video via ffmpeg pipe (bypasses OpenCV H.264 codec issues).
Fixes Point 3 (Red) glitches using rotation-aware local frame reconstruction.
Saves corrected coordinates to a new CSV preserving the original DLC structure.

Install once if ffmpeg missing:
    conda install -c conda-forge ffmpeg -y

Dependencies:
    pip install opencv-python-headless numpy pandas
"""

import os
import sys
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd


# =============================================================================
# 1. LOCATE FFMPEG
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
print(f"[ffmpeg]  {FFMPEG}")


# =============================================================================
# 2. VIDEO METADATA
# =============================================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cv2 cannot open video:\n  {video_path}")
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if w == 0 or h == 0:
        raise IOError(f"cv2 reported 0x0 resolution for:\n  {video_path}")
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        print("[warn]  FPS reported as 0 — defaulting to 30")
    return w, h, fps, n


# =============================================================================
# 3. FFMPEG PIPE READER
# =============================================================================

class FfmpegFrameReader:
    def __init__(self, video_path, start_frame, native_w, native_h):
        self.video_path  = video_path
        self.start_frame = start_frame
        self.native_w    = native_w
        self.native_h    = native_h
        self._proc       = None

    def __enter__(self):
        cmd = [FFMPEG, "-loglevel", "error", "-i", self.video_path]
        if self.start_frame > 0:
            cmd += ["-vf", f"select=gte(n\\,{self.start_frame})", "-vsync", "0"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self._proc       = subprocess.Popen(cmd, stdout=subprocess.PIPE,
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


# =============================================================================
# 4. POINT 3 GLITCH CORRECTION — rotation-aware local frame
# =============================================================================

def fix_point3(df, likelihood_threshold=0.35, distance_threshold=15.0):
    """
    Correct Point 3 (Red) glitches using the worm head's local coordinate frame.

    Thresholds
    ----------
    likelihood_threshold : DLC confidence below this → frame flagged bad
                           lower = more frames corrected
    distance_threshold   : P2->P3 distance shift above this (px) → flagged bad
                           lower = tighter, catches subtler jumps

    Reconstruction
    --------------
    Good frame:
        head_angle = atan2(y2-y1, x2-x1)
        r          = distance(P2, P3)
        alpha      = atan2(y3-y2, x3-x2) - head_angle

    Bad frame:
        head_angle_now = atan2(y2-y1, x2-x1)
        P3 = P2 + r * [cos(head_angle_now + alpha),
                        sin(head_angle_now + alpha)]
    """
    x1 = df[('point_1', 'x')].values.copy()
    y1 = df[('point_1', 'y')].values.copy()
    x2 = df[('point_2', 'x')].values.copy()
    y2 = df[('point_2', 'y')].values.copy()
    x3 = df[('point_3', 'x')].values.copy()
    y3 = df[('point_3', 'y')].values.copy()
    p3 = df[('point_3', 'likelihood')].values.copy()

    n = len(df)

    def get_head_angle(t):
        dx, dy = x2[t] - x1[t], y2[t] - y1[t]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return np.arctan2(dy, dx)

    def encode(t):
        ha = get_head_angle(t)
        if ha is None:
            return None
        r     = np.hypot(x3[t] - x2[t], y3[t] - y2[t])
        alpha = np.arctan2(y3[t] - y2[t], x3[t] - x2[t]) - ha
        return r, alpha

    def reconstruct(t, r, alpha):
        ha = get_head_angle(t)
        if ha is None:
            return x2[t] + r * np.cos(alpha), y2[t] + r * np.sin(alpha)
        angle_global = ha + alpha
        return x2[t] + r * np.cos(angle_global), y2[t] + r * np.sin(angle_global)

    last_good = None
    for t in range(n):
        if p3[t] >= likelihood_threshold:
            enc = encode(t)
            if enc is not None:
                last_good = enc
                break

    if last_good is None:
        print("[warn]  No reliable seed frame found — fix skipped.")
        return df

    fixed_count = 0
    for t in range(n):
        enc    = encode(t)
        is_bad = p3[t] < likelihood_threshold
        if not is_bad and enc is not None:
            if abs(enc[0] - last_good[0]) > distance_threshold:
                is_bad = True

        if is_bad:
            rx, ry = reconstruct(t, last_good[0], last_good[1])
            x3[t], y3[t], p3[t] = rx, ry, 1.0
            fixed_count += 1
        else:
            if enc is not None:
                last_good = enc

    df[('point_3', 'x')]          = x3
    df[('point_3', 'y')]          = y3
    df[('point_3', 'likelihood')] = p3
    print(f"[fix]    Corrected {fixed_count} / {n} frames for Point 3 (Red)")
    return df


# =============================================================================
# 5. SAVE CORRECTED CSV  (preserves original DLC multi-level header exactly)
# =============================================================================

def save_corrected_csv(original_csv_path, corrected_df, output_csv_path):
    """
    Write the corrected DataFrame back to a CSV that matches the original
    DLC format exactly:

        Row 0  — scorer   row  (e.g. "DLC_Resnet50_...")   copied verbatim
        Row 1  — bodypart row  (point_1, point_2, ...)     from MultiIndex level 0
        Row 2  — coord    row  (x, y, likelihood)          from MultiIndex level 1
        Row 3+ — data rows

    The scorer row is read from the raw original file so the network name
    is preserved precisely, then the corrected data is written underneath.
    """
    # Read the raw original to capture the exact scorer header (row 0)
    raw = pd.read_csv(original_csv_path, header=None, nrows=1)
    scorer_row = raw.iloc[0].tolist()           # e.g. ['scorer', 'DLC_...', 'DLC_...', ...]

    # Rebuild bodypart and coord rows from the corrected DataFrame's MultiIndex
    bodypart_row = ['bodyparts'] + [col[0] for col in corrected_df.columns]
    coord_row    = ['coords']    + [col[1] for col in corrected_df.columns]

    # Build the complete output: 3 header rows + data rows
    # Reset index so the original row numbers become a plain column
    data_rows = corrected_df.reset_index(drop=True)

    with open(output_csv_path, 'w') as f:
        # Header rows
        f.write(','.join(str(v) for v in scorer_row)   + '\n')
        f.write(','.join(str(v) for v in bodypart_row) + '\n')
        f.write(','.join(str(v) for v in coord_row)    + '\n')

        # Data rows — row index matches the original frame numbers
        for i, row in data_rows.iterrows():
            f.write(str(i) + ',' + ','.join(str(v) for v in row.values) + '\n')

    print(f"[csv]    Corrected CSV saved -> {output_csv_path}")


# =============================================================================
# 6. DRAW FRAME LABEL
# =============================================================================

def draw_frame_label(frame, frame_number, h):
    text      = f"Frame: {frame_number}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0, y0 = 20, h - 20
    pad = 6
    cv2.rectangle(frame,
                  (x0 - pad,      y0 - th - pad),
                  (x0 + tw + pad, y0 + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, y0), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def generate_custom_labeled_video(
    root_dir,
    csv_filename,
    raw_video_filename,
    output_filename,
    output_csv_filename,
    start_frame          = 0,
    end_frame            = None,
    likelihood_threshold = 0.5,
    distance_threshold   = 10.0,
):
    csv_path     = os.path.join(root_dir, csv_filename)
    video_path   = os.path.join(root_dir, raw_video_filename)
    out_path     = os.path.join(root_dir, output_filename)
    out_csv_path = os.path.join(root_dir, output_csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found:\n  {csv_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found:\n  {video_path}")

    native_w, native_h, fps, total_frames = get_video_info(video_path)
    print(f"[video]  {native_w}x{native_h}  {fps:.2f} fps  {total_frames} frames")

    df = pd.read_csv(csv_path, header=[1, 2])
    if end_frame is None or end_frame > len(df):
        end_frame = len(df)
    df_sliced = df.iloc[start_frame:end_frame].copy().reset_index(drop=True)
    n_frames  = len(df_sliced)
    print(f"[csv]    frames {start_frame} to {end_frame}  ({n_frames} frames)")

    # Fix glitches
    df_sliced = fix_point3(df_sliced, likelihood_threshold, distance_threshold)

    # Save corrected CSV immediately after fixing
    save_corrected_csv(csv_path, df_sliced, out_csv_path)

    # Render video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(out_path, fourcc, fps, (native_w, native_h))
    if not out.isOpened():
        raise IOError(f"VideoWriter failed:\n  {out_path}")

    COLOR_MAP = {
        'point_1': (150,   0, 150),
        'point_2': (130, 255, 180),
        'point_3': (  0,   0, 255),
    }
    LABELS = {'point_1': 'P1', 'point_2': 'P2', 'point_3': 'P3'}

    print(f"[render] writing -> {out_path}")
    frames_written = 0

    with FfmpegFrameReader(video_path, start_frame, native_w, native_h) as reader:
        for idx, frame in enumerate(reader):
            if idx >= n_frames:
                break

            pts = []
            for bp in ['point_1', 'point_2', 'point_3']:
                xv = df_sliced[(bp, 'x')].values[idx]
                yv = df_sliced[(bp, 'y')].values[idx]
                if not (np.isnan(xv) or np.isnan(yv)):
                    pts.append((int(round(xv)), int(round(yv))))
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], (200, 200, 200), 1, cv2.LINE_AA)

            for bodypart, color in COLOR_MAP.items():
                x = df_sliced[(bodypart, 'x')].values[idx]
                y = df_sliced[(bodypart, 'y')].values[idx]
                if np.isnan(x) or np.isnan(y):
                    continue
                cx, cy = int(round(x)), int(round(y))
                cv2.circle(frame, (cx, cy), 7, color, -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)
                cv2.putText(frame, LABELS[bodypart],
                            (cx + 10, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 1, cv2.LINE_AA)

            draw_frame_label(frame, start_frame + idx, native_h)

            out.write(frame)
            frames_written += 1

            if idx % 200 == 0:
                print(f"  [{100*idx/n_frames:5.1f}%]  frame {idx:>5} / {n_frames}")

    out.release()
    print(f"\n[done]  {frames_written} frames -> {out_path}")
    if frames_written == 0:
        print("  WARNING: zero frames written — check ffmpeg and file paths.")


# =============================================================================
# 8. RUN
# =============================================================================

if __name__ == "__main__":

    ROOT_DIR = "/gpfs/soma_fs/home/pyaasa/Desktop/Test_Data"

    CSV_FILE = (
        "20252606-16-11Expnt001"
        "DLC_Resnet50_tracking_20252606-16-11Expnt001May22shuffle1_snapshot_best-90.csv"
    )

    generate_custom_labeled_video(
        root_dir             = ROOT_DIR,
        csv_filename         = CSV_FILE,
        raw_video_filename   = "20252606-16-11Expnt001.mp4",
        output_filename      = "clean_interpolated_tracking.mp4",
        output_csv_filename  = "clean_interpolated_tracking_corrected.csv",
        start_frame          = 0,
        end_frame            = 8000,
        likelihood_threshold = 0.5,
        distance_threshold   = 10.0,
    )