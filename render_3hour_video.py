"""
3-Hour Black Hole Plunge & Interior Video Stream Exporter
========================================================
Generates a long-form MP4 video or stream of an observer plunging
through a black hole's event horizon and traveling inside its interior for 3 hours.

Memory-efficient chunked streaming writer ensures low RAM consumption.
Supports time-warp speed acceleration (1x real-time to 3600x warp).

Usage:
    # Quick 10-second preview of the 3-hour travel:
    python render_3hour_video.py --preview

    # Full 3-hour journey video (time-warped or real-time stream):
    python render_3hour_video.py --duration 10800 --time-warp 60 --output plunge_3hour.mp4
"""

import argparse
import math
import os
import sys
import time
import numpy as np
from PIL import Image

try:
    import imageio
except ImportError:
    imageio = None

from grrt_plunge import render_plunge_frame, get_camera_state


def render_3hour_plunge_video(
    duration_sec=10800,
    time_warp=60.0,
    fps=30,
    width=800,
    height=450,
    output_filename="black_hole_3hour_plunge.mp4",
    preview=False
):
    """
    Renders a long-form video of plunging into and traveling inside a black hole.

    Args:
        duration_sec: Simulated travel time inside/through black hole (seconds). Default 10800s (3 hours).
        time_warp: Speed multiplier (60x means 3 simulated hours take 3 minutes video time).
        fps: Video frame rate.
        width, height: Resolution in pixels.
        output_filename: MP4 output file path.
        preview: If True, renders a fast 10-second preview clip.
    """
    if imageio is None:
        print("Error: imageio library is required. Install with: pip install imageio[ffmpeg]")
        return False

    M = 1.0
    r_isco = 6.0 * M
    r_out = 26.0 * M
    cam_start_r = 40.0 * M
    max_steps = 1800

    if preview:
        effective_duration = 10.0  # 10 second video preview
        total_frames = int(effective_duration * fps)
        t_sim_max = 120.0  # covers full plunge + extended interior travel
        print(f"[PREVIEW MODE] Rendering 10s video clip ({total_frames} frames) representing 3-hour plunge...")
    else:
        video_duration_sec = duration_sec / time_warp
        total_frames = int(video_duration_sec * fps)
        t_sim_max = duration_sec / 30.0  # normalized simulation timeline
        print("=" * 68)
        print("  3-HOUR BLACK HOLE INTERIOR PLUNGE VIDEO RENDERER  ")
        print("=" * 68)
        print(f"Simulated Interior Journey: {duration_sec / 3600:.1f} Hours ({duration_sec:.0f} seconds)")
        print(f"Time-Warp Speed Factor:     {time_warp}x")
        print(f"Output Video Duration:      {video_duration_sec:.1f} seconds ({total_frames} frames @ {fps} fps)")
        print(f"Target Resolution:          {width} x {height}")
        print(f"Output File:                {output_filename}")
        print("=" * 68)

    t_start = time.time()
    writer = imageio.get_writer(
        output_filename,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2
    )

    horizon_crossed_frame = None

    for idx in range(total_frames):
        fraction = idx / max(1, total_frames - 1)
        t_sim = fraction * t_sim_max

        # Compute current camera state & render frame
        img, r_cam, is_inside = render_plunge_frame(
            width, height, 45.0, t_sim, cam_start_r, M, r_isco, r_out, max_steps
        )
        rgb = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        writer.append_data(rgb)

        if is_inside and horizon_crossed_frame is None:
            horizon_crossed_frame = idx
            print(f"  ---> EVENT HORIZON CROSSED at Frame {idx+1}/{total_frames} (r = {r_cam:.3f}M)!")

        # Progress reporting
        if (idx + 1) % max(1, total_frames // 20) == 0 or idx == total_frames - 1:
            elapsed = time.time() - t_start
            fps_perf = (idx + 1) / elapsed
            rem_sec = (total_frames - (idx + 1)) / (fps_perf + 1e-6)
            sim_time_hours = (t_sim / t_sim_max) * (duration_sec / 3600.0)
            status = "INSIDE HORIZON (TRAVELING INTERIOR)" if is_inside else "APPROACHING EVENT HORIZON"

            print(f"  Frame {idx+1:5d}/{total_frames} | r = {r_cam:6.3f}M | Sim Time: {sim_time_hours:4.2f}h | {fps_perf:4.1f} fps | ETA: {rem_sec:4.0f}s | {status}")

    writer.close()
    t_end = time.time()
    total_render_time = t_end - t_start

    print("\n" + "=" * 68)
    print("  3-HOUR INTERIOR JOURNEY VIDEO EXPORT COMPLETE  ")
    print("=" * 68)
    print(f"Saved video to:          {output_filename}")
    print(f"Total Frames Rendered:   {total_frames}")
    print(f"Total Render Time:       {total_render_time:.2f} seconds ({total_frames / total_render_time:.1f} fps)")
    print("=" * 68 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="3-Hour Black Hole Interior Plunge Video Renderer")
    parser.add_argument("--duration", type=float, default=10800.0, help="Simulated interior journey duration in seconds (default: 10800 = 3 hours)")
    parser.add_argument("--time-warp", type=float, default=60.0, help="Time warp speed multiplier (default: 60.0x)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=800, help="Video width in pixels (default: 800)")
    parser.add_argument("--height", type=int, default=450, help="Video height in pixels (default: 450)")
    parser.add_argument("--output", type=str, default="black_hole_3hour_plunge.mp4", help="Output MP4 filename")
    parser.add_argument("--preview", action="store_true", help="Generate a quick 10-second preview clip")

    args = parser.parse_args()

    render_3hour_plunge_video(
        duration_sec=args.duration,
        time_warp=getattr(args, "time_warp", 60.0),
        fps=args.fps,
        width=args.width,
        height=args.height,
        output_filename=args.output,
        preview=args.preview
    )


if __name__ == "__main__":
    main()
