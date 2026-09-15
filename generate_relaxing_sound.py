"""
Black Hole Relaxing Ambient Audio Generator & Video Audio Muxer
================================================================
Synthesizes a deeply relaxing, soothing cosmic black hole ambient soundtrack:
- Sub-bass gravitational wave hum (43.2 Hz & 86.4 Hz warm sine waves)
- Deep space brownian noise drone (filtered interstellar gas & radiation hum)
- 432 Hz harmonic ambient pad overtones (108 Hz, 216 Hz, 432 Hz warm tones)
- Gentle 0.05 Hz slow breathing LFO pulse for deep relaxation, sleep & focus

Can also combine (mux) the relaxing audio seamlessly into black hole video loops.

Usage:
    # Synthesize 60-second seamless looping relaxing audio:
    python generate_relaxing_sound.py --output black_hole_relaxing_ambient.wav --duration 60

    # Add relaxing sound to a black hole video loop:
    python generate_relaxing_sound.py --add-to-video black_hole_simulation_video.mp4 --output black_hole_video_with_relaxing_sound.mp4
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np
from scipy.io import wavfile

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"


def generate_relaxing_black_hole_audio(duration_sec=60.0, sample_rate=44100, output_wav="black_hole_relaxing_sound.wav"):
    """
    Synthesizes a high-fidelity relaxing cosmic black hole ambient soundscape.

    Args:
        duration_sec: Length of audio track in seconds (default: 60s, seamlessly loopable).
        sample_rate: Audio sampling rate (default: 44100 Hz).
        output_wav: Output WAV file path.
    """
    print("=" * 68)
    print("  SYNTHESIZING BLACK HOLE RELAXING COSMIC AMBIENCE  ")
    print("=" * 68)
    print(f"Duration:     {duration_sec:.1f} seconds")
    print(f"Sample Rate:  {sample_rate} Hz")
    print(f"Output File:  {output_wav}")

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    n_samples = len(t)

    # 1. Sub-Bass Gravitational Wave Oscillators (43.2 Hz & 86.4 Hz)
    sub_bass_43 = 0.35 * np.sin(2.0 * np.pi * 43.2 * t)
    sub_bass_86 = 0.20 * np.sin(2.0 * np.pi * 86.4 * t)

    # Gentle 0.05 Hz (20s period) breathing LFO modulation
    lfo_breath = 0.65 + 0.35 * np.sin(2.0 * np.pi * 0.05 * t)
    sub_track = (sub_bass_43 + sub_bass_86) * lfo_breath

    # 2. Cosmic Space Drone (Filtered Brownian/Pink Noise)
    # Generate brown noise by integrating white noise
    white_noise = np.random.normal(0, 1, n_samples)
    brown_noise = np.cumsum(white_noise)
    brown_noise = brown_noise - np.mean(brown_noise)
    brown_noise = brown_noise / (np.max(np.abs(brown_noise)) + 1e-6)

    # Simple IIR Low-Pass Filter (cutoff around 150 Hz)
    alpha = 0.02
    filtered_drone = np.zeros(n_samples, dtype=np.float64)
    for i in range(1, n_samples):
        filtered_drone[i] = filtered_drone[i-1] + alpha * (brown_noise[i] - filtered_drone[i-1])

    drone_track = 0.25 * filtered_drone * (0.8 + 0.2 * np.sin(2.0 * np.pi * 0.025 * t))

    # 3. 432 Hz Harmonically Relaxing Ambient Pads (108 Hz, 216 Hz, 432 Hz)
    pad_108 = 0.12 * np.sin(2.0 * np.pi * 108.0 * t + 0.2 * np.sin(2.0 * np.pi * 0.1 * t))
    pad_216 = 0.08 * np.sin(2.0 * np.pi * 216.0 * t + 0.4 * np.sin(2.0 * np.pi * 0.08 * t))
    pad_432 = 0.04 * np.sin(2.0 * np.pi * 432.0 * t + 0.6 * np.sin(2.0 * np.pi * 0.04 * t))
    pad_track = (pad_108 + pad_216 + pad_432) * (0.7 + 0.3 * np.cos(2.0 * np.pi * 0.033 * t))

    # 4. Combine Tracks & Normalize
    mix = sub_track + drone_track + pad_track
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val * 0.85  # Master peak headroom at -1.4 dB

    # 5. Apply Smooth Fade In / Fade Out for Seamless Looping
    fade_len = int(sample_rate * 1.5)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    mix[:fade_len] *= fade_in
    mix[-fade_len:] *= fade_out

    # Convert to 16-bit PCM WAV
    audio_pcm = (mix * 32767).astype(np.int16)

    # Save stereo WAV (duplicate left and right with slight phase shift for spatial width)
    right_channel = (np.roll(mix, int(sample_rate * 0.005)) * 32767).astype(np.int16)
    stereo_pcm = np.column_stack((audio_pcm, right_channel))

    wavfile.write(output_wav, sample_rate, stereo_pcm)
    print(f"[OK] Generated relaxing black hole audio: {output_wav}\n")
    return output_wav


def add_audio_to_video(video_path, audio_path, output_path):
    """Muxes relaxing audio track onto black hole video file using FFmpeg."""
    print("=" * 68)
    print("  MUXING RELAXING BLACK HOLE AUDIO INTO VIDEO  ")
    print("=" * 68)
    print(f"Input Video:  {video_path}")
    print(f"Input Audio:  {audio_path}")
    print(f"Output Video: {output_path}")

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-stream_loop", "-1",       # Loop audio if video is longer
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",             # Copy video codec without re-encoding
        "-c:a", "aac",              # High quality AAC audio
        "-b:a", "192k",
        "-shortest",                # Match shortest length
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[OK] Successfully saved video with relaxing black hole audio: {output_path}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode('utf-8', errors='ignore')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Black Hole Relaxing Sound Synthesizer & Video Audio Muxer")
    parser.add_argument("--output", type=str, default="black_hole_relaxing_sound.wav", help="Output audio WAV filename")
    parser.add_argument("--duration", type=float, default=120.0, help="Audio duration in seconds (default: 120s)")
    parser.add_argument("--add-to-video", type=str, help="Path to input black hole video file to add sound to")
    parser.add_argument("--video-output", type=str, help="Path to final output video with sound")

    args = parser.parse_args()

    # Generate relaxing sound WAV
    audio_wav = generate_relaxing_black_hole_audio(duration_sec=args.duration, output_wav=args.output)

    # Mux to video if requested
    if args.add_to_video:
        video_in = args.add_to_video
        video_out = args.video_output if args.video_output else os.path.splitext(video_in)[0] + "_relaxing_sound.mp4"
        add_audio_to_video(video_in, audio_wav, video_out)


if __name__ == "__main__":
    main()
