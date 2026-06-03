import os
import subprocess
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve
from matplotlib.backends.backend_agg import FigureCanvasAgg


# =============================================================================
# VIDEO READER
# =============================================================================
def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    num, den = parts[2].split("/")
    fps = float(num) / float(den)
    n_frames = int(parts[3]) if parts[3].strip() not in ("N/A", "") else -1
    return w, h, fps, n_frames


class FfmpegFrameReader:
    def __init__(self, video_path, start_frame, native_w, native_h):
        self.video_path = video_path
        self.start_frame = start_frame
        self.native_w = native_w
        self.native_h = native_h
        self._proc = None

    def __enter__(self):
        cmd = ["ffmpeg", "-loglevel", "error", "-i", self.video_path]
        if self.start_frame > 0: cmd += ["-vf", f"select=gte(n\\,{self.start_frame})", "-vsync", "0"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._frame_size = self.native_w * self.native_h * 3
        return self

    def __iter__(self):
        while True:
            raw = self._proc.stdout.read(self._frame_size)
            if len(raw) < self._frame_size: break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.native_h, self.native_w, 3))
            yield frame.copy()

    def __exit__(self, *_):
        if self._proc:
            self._proc.stdout.close()
            self._proc.terminate()
            self._proc.wait()

def powersmooth(data_in, order=2, weight=500.0):
    weight = float(weight) * 5e-14
    data_in = np.asarray(data_in, dtype=float)
    N = len(data_in)
    data_out = np.full(N, np.nan)
    is_good = np.isfinite(data_in)
    data_valid = data_in[is_good]
    n = len(data_valid)
    if n <= order: return data_out
    d = spdiags(np.ones(n), 0, n, n) - spdiags(np.ones(n-1), -1, n, n)
    p = np.concatenate((np.zeros(order), np.ones(n - order)))
    dk = spdiags(p, 0, n, n)
    for _ in range(order): dk = dk @ d
    effective_weight = weight * (n ** (2 * order))
    A = eye(n) + effective_weight * (dk.T @ dk)
    data_out[is_good] = spsolve(A.tocsr(), data_valid)
    return data_out

def draw_frame_label(frame, frame_number, h):
    text = f"Frame: {frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x0 = 20
    y0 = h - baseline - pad - 10  # FIX: fully visible inside frame
    cv2.rectangle(frame, (x0 - pad, y0 - th - pad), (x0 + tw + pad, y0 + baseline + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, y0), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def process_video_distance(root_dir, csv_filename, video_filename, output_filename, start_frame=0, end_frame=None,
                           panel_width=800, panel_height=600):
    csv_filepath = os.path.join(root_dir, csv_filename)
    video_filepath = os.path.join(root_dir, video_filename)
    output_filepath = os.path.join(root_dir, output_filename)

    native_w = 1544
    native_h = 1032
    panel_width = 750
    panel_height = 500
    fps = 30

    # Read CSV
    df = pd.read_csv(csv_filepath, header=[1, 2])
    if end_frame is None or end_frame > len(df): end_frame = len(df)
    df_sliced = df.iloc[start_frame:end_frame].copy()
    n_frames = len(df_sliced)

    # Calculate Distance
    x1, y1 = df_sliced[('point_1', 'x')].values, df_sliced[('point_1', 'y')].values
    x2, y2 = df_sliced[('point_2', 'x')].values, df_sliced[('point_2', 'y')].values
    x3, y3 = df_sliced[('point_3', 'x')].values, df_sliced[('point_3', 'y')].values
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #distance_vel = derivative(distance, x=np.arange(n_frames)) * fps
    distance_smooth = powersmooth(distance, weight=500)
    distance = distance_smooth
    time_seconds = np.arange(n_frames) / fps
    padding = 60

    # -------------------------------------------------------------------------
    # CHANGE 1: Compute fixed crop window once across ALL frames and ALL points.
    # Previously the bounding box was recalculated per-frame, causing the crop
    # window to jump and resize every frame — very disorienting to watch.
    # Now we take the global min/max of all 3 points over the entire clip and
    # add padding once, so the view is perfectly stable for the whole video.
    # -------------------------------------------------------------------------
    all_x = np.concatenate([x1, x2, x3])
    all_y = np.concatenate([y1, y2, y3])

    fixed_min_x = max(0, int(np.min(all_x) - padding))
    fixed_max_x = min(native_w, int(np.max(all_x) + padding))
    fixed_min_y = max(0, int(np.min(all_y) - padding))
    fixed_max_y = min(native_h, int(np.max(all_y) + padding))

    fixed_crop_w = fixed_max_x - fixed_min_x
    fixed_crop_h = fixed_max_y - fixed_min_y

    # CHANGE 2: Pre-compute letterbox scale and offsets once (reused every frame)
    scale = min(panel_width / fixed_crop_w, panel_height / fixed_crop_h)
    new_w = int(fixed_crop_w * scale)
    new_h = int(fixed_crop_h * scale)
    x_offset = (panel_width - new_w) // 2
    y_offset = (panel_height - new_h) // 2
    # -------------------------------------------------------------------------

    # Setup Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(panel_width / 100, panel_height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)

    d_min, d_max = np.min(distance) - 2, np.max(distance) + 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (panel_width * 2, panel_height))

    with FfmpegFrameReader(video_filepath, start_frame, native_w, native_h) as reader:
        for idx, frame in tqdm(enumerate(reader), total=n_frames, desc="Processing"):
            if idx >= n_frames: break
            if distance[idx] < 22:
                print(f"frame {idx} has spike and distance is {distance[idx]}")

            # -----------------------------------------------------------------
            # CHANGE 3: Replaced per-frame dynamic crop with fixed crop window.
            # No more per-frame min_x/max_x/crop_w/ratio_x/ratio_y calculations.
            # scale, x_offset, y_offset are all pre-computed constants now.
            # -----------------------------------------------------------------
            cropped_frame = frame[fixed_min_y:fixed_max_y, fixed_min_x:fixed_max_x]
            resized = cv2.resize(cropped_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frame_left = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            frame_left[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            pt1_x = int((x1[idx] - fixed_min_x) * scale) + x_offset
            pt1_y = int((y1[idx] - fixed_min_y) * scale) + y_offset
            pt2_x = int((x2[idx] - fixed_min_x) * scale) + x_offset
            pt2_y = int((y2[idx] - fixed_min_y) * scale) + y_offset

            # -----------------------------------------------------------------

            # Right: Distance Plot
            ax.clear()
            ax.plot(time_seconds[:idx + 1], distance[:idx + 1], color='#00ff88', linewidth=2)
            ax.set_title('Euclidian Distance between P1 and P2')
            ax.set_ylabel('Distance (pixels)')
            ax.set_xlabel('Time (s)')
            ax.set_ylim(d_min, d_max)
            ax.set_xlim(0, time_seconds[-1])
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            draw_frame_label(frame_left, start_frame + idx, panel_height)
            canvas.draw()
            plot_frame = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

            out.write(np.hstack((frame_left, cv2.resize(plot_frame, (panel_width, panel_height)))))

    out.release()
    plt.close(fig)
    print(f"Done! Saved to {output_filepath}")


if __name__ == "__main__":
    process_video_distance(
        root_dir="/gpfs/soma_fs/home/pyaasa/Desktop/Test_Data",
        csv_filename="corrected_20252606-16-11Expnt001DLC_Resnet50_tracking_20252606-16-11Expnt001May22shuffle1_snapshot_best-90.csv",
        video_filename="interpolated_dlc_tracking_distance.mp4",
        output_filename="distance_plot_video.mp4",
        start_frame=0,
        end_frame=8000
    )