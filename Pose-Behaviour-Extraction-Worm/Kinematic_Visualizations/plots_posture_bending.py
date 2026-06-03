import os
import subprocess
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve


def derivative(y, x=None):
    if x is None: x = np.arange(len(y))
    dy = np.zeros(y.shape)
    if y.ndim == 2:
        dy[0, :] = (y[1, :] - y[0, :]) / (x[1] - x[0])
        for i in range(1, y.shape[0] - 1):
            dy[i, :] = ((y[i+1, :] - y[i, :]) / (x[i+1] - x[i]) + (y[i, :] - y[i-1, :]) / (x[i] - x[i-1])) / 2
        dy[-1, :] = (y[-1, :] - y[-2, :]) / (x[-1] - x[-2])
    else:
        dy[0] = (y[1] - y[0]) / (x[1] - x[0])
        for i in range(1, y.shape[0] - 1):
            dy[i] = ((y[i+1] - y[i]) / (x[i+1] - x[i]) + (y[i] - y[i-1]) / (x[i] - x[i-1])) / 2
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy

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

def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0", video_path]
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
            self._proc.stdout.close(); self._proc.terminate(); self._proc.wait()

def process_video_and_extract_frames(root_dir, csv_filename, video_filename, output_filename, start_frame=0, end_frame=None, fps=30, panel_width=800, panel_height=600):
    csv_filepath = os.path.join(root_dir, csv_filename)
    video_filepath = os.path.join(root_dir, video_filename)
    output_filepath = output_filename if os.path.isabs(output_filename) else os.path.join(root_dir, output_filename)

    native_w, native_h, _, _ = get_video_info(video_filepath)
    scale_x, scale_y = panel_width / native_w, panel_height / native_h

    df = pd.read_csv(csv_filepath, header=[1, 2])
    if end_frame is None or end_frame > len(df): end_frame = len(df)
    df_sliced = df.iloc[start_frame:end_frame].copy()
    n_frames = len(df_sliced)

    x1, y1 = df_sliced[('point_1', 'x')].values, df_sliced[('point_1', 'y')].values
    x2, y2 = df_sliced[('point_2', 'x')].values, df_sliced[('point_2', 'y')].values
    x3, y3 = df_sliced[('point_3', 'x')].values, df_sliced[('point_3', 'y')].values
    x_mid, y_mid = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # Kinematics
    vec_neck = np.array([x2 - x1, -(y2 - y1)])
    posture_deg = np.degrees(np.unwrap(np.arctan2(vec_neck[1], vec_neck[0])))
    posture_smooth = powersmooth(posture_deg, weight=500)
    posture_vel = derivative(posture_smooth, x=np.arange(n_frames)) * fps
    vec_head = np.array([x3 - x_mid, -(y3 - y_mid)])
    norm_neck, norm_head = np.linalg.norm(vec_neck, axis=0), np.linalg.norm(vec_head, axis=0)
    cos_theta = np.clip(np.sum(vec_neck * vec_head, axis=0) / (norm_neck * norm_head), -1.0, 1.0)
    bending_mag = np.degrees(np.arccos(cos_theta)) - np.mean(np.degrees(np.arccos(cos_theta))[:5])
    bending_smooth = powersmooth(bending_mag, weight=500)
    bending_vel = derivative(bending_smooth, x=np.arange(n_frames)) * fps

    time_seconds = np.arange(start_frame, end_frame) / fps
    t_min, t_max = time_seconds[0], time_seconds[-1]

    # --- INITIALIZE PLOT ONCE ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(panel_width / 100, panel_height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)

    #ax_p_vel, ax_b_vel = axes[0].twinx(), axes[1].twinx()
    ax_p_angle, ax_b_angle = axes[0].twinx(), axes[1].twinx()

    line_p, = ax_p_angle.plot([], [], color='#ffcc00', linewidth=2, zorder=5)
    line_pv, = axes[0].plot([], [], color='#00ff88', linewidth=2, zorder=10)
    line_b, = ax_b_angle.plot([], [], color='#ff2a5f', linewidth=2, zorder=2)
    line_bv, = axes[1].plot([], [], color='#00bfff', linewidth=2, zorder=1)

    # --- ADD LABELS HERE (Outside the loop) ---
    axes[0].set_title('POSTURE', fontsize=14)
    ax_p_angle.set_ylabel('Angle (°)', color='#ffcc00', fontsize=12)
    axes[0].set_ylabel('Vel (°/s)', color='#00ff88', fontsize=12)


    axes[1].set_title('BENDING', fontsize=14)
    ax_b_angle.set_ylabel('Angle (°)', color='#ff2a5f', fontsize=12)
    axes[1].set_ylabel('Vel (°/s)', color='#00bfff', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=11)

    # Set limits once
    for ax in [axes[0], axes[1]]: ax.set_xlim(t_min, t_max)
    ax_p_angle.set_ylim(np.nanmin(posture_smooth)-2, np.nanmax(posture_smooth)+2)
    axes[0].set_ylim(np.nanmin(posture_vel) - 10, np.nanmax(posture_vel) + 10)
    ax_b_angle.set_ylim(np.nanmin(bending_smooth)-2, np.nanmax(bending_smooth)+2)
    axes[1].set_ylim(np.nanmin(bending_vel) - 10, np.nanmax(bending_vel) + 10)

    fig.tight_layout(pad=1.5)

    # Video Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps*2, (panel_width * 2, panel_height))

    with FfmpegFrameReader(video_filepath, start_frame, native_w, native_h) as reader:
        for idx, frame in tqdm(enumerate(reader), total=n_frames, desc="Processing Video"):
            if idx >= n_frames: break

            # --- Draw Left Panel ---
            frame_left = cv2.resize(frame, (panel_width, panel_height))

            # --- UPDATE PLOT LINES ---
            line_p.set_data(time_seconds[:idx+1], posture_smooth[:idx+1])
            line_pv.set_data(time_seconds[:idx+1], posture_vel[:idx+1])
            line_b.set_data(time_seconds[:idx+1], bending_smooth[:idx+1])
            line_bv.set_data(time_seconds[:idx+1], bending_vel[:idx+1])

            # Live readout overlay
            overlay_items = [

                (f"Posture Angle : {posture_deg[idx]:+.1f} deg", (0, 204, 255), 30),

                (f"Posture Vel   : {posture_vel[idx]:+.1f} deg/s", (136, 255, 0), 60),

                (f"Bending Angle : {bending_mag[idx]:+.1f} deg", (95, 42, 255), 90),

                (f"Bending Vel   : {bending_vel[idx]:+.1f} deg/s", (255, 191, 0), 120),

                (f"t = {time_seconds[idx]:.2f} s", (200, 200, 200), 150),

            ]

            for text, colour, y_pos in overlay_items:
                cv2.putText(frame_left, text,

                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,

                            0.7, colour, 2, cv2.LINE_AA)

            canvas.draw()
            plot_frame = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            out.write(np.hstack((frame_left, cv2.resize(plot_frame, (panel_width, panel_height)))))

    out.release()
    plt.close(fig)
    print(f"\n[Done] Saved to: {output_filepath}")

if __name__ == "__main__":
    process_video_and_extract_frames(
        root_dir="/gpfs/soma_fs/home/pyaasa/Desktop/Test_Data",
        csv_filename="corrected_20252606-16-11Expnt001DLC_Resnet50_tracking_20252606-16-11Expnt001May22shuffle1_snapshot_best-90.csv",
        video_filename="interpolated_dlc_tracking_distance.mp4",
        output_filename="synchronized_velocity_yy_plot.mp4",
        end_frame=8000
    )