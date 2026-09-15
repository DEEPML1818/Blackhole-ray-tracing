"""
run_all.py -- Single Entry Point for the Complete Black Hole Simulation Suite
=============================================================================

Pipeline Steps:
  Step 1 -- simulation.py         : Construct puncture initial data (numerical relativity)
  Step 2 -- plot_simulation.py    : Plot & save 3D solution surface to PNG
  Step 3 -- grrt_black_hole.py    : General Relativistic Ray-Tracing (static render)
  Step 4 -- grrt_plunge.py        : Horizon Plunge & Interior Spacetime Simulation
  Step 5 -- render_3hour_video.py : 3-Hour Interior Journey MP4 Exporter / Web Simulator

Usage:
    python run_all.py                  # Run complete standard pipeline
    python run_all.py --web            # Launch Interactive Web 3D Simulator in browser
    python run_all.py --plunge         # Run Horizon Plunge test & preview video
    python run_all.py --video-3h       # Render 3-Hour Interior Plunge MP4 video
"""

import argparse
import os
import sys
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)


def launch_web_server(port=8000):
    """Launches local HTTP server and opens browser for interactive web simulator."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    url = f"http://localhost:{port}/index.html"
    print("\n" + "=" * 68)
    print("  LAUNCHING INTERACTIVE BLACK HOLE 3D WEB SIMULATOR  ")
    print("=" * 68)
    print(f"Server Running at: {url}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 68 + "\n")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


def run_pipeline(args):
    # =========================================================================
    # STEP 1 -- Numerical Relativity: Puncture Initial Data
    # =========================================================================
    print("\n" + "=" * 68)
    print("  STEP 1 -- Puncture Initial Data (simulation.py)")
    print("=" * 68)

    from simulation import Puncture

    bh_location = (0.0, 0.0, 0.0)
    linear_momentum = (1.0, 0.0, 0.0)
    grid_dim = 16
    boundary = 4.0
    tol = 1.0e-12
    it_max = 50

    t0 = time.time()
    bh = Puncture(bh_location, linear_momentum, grid_dim, boundary)
    bh.construct_solution(tol, it_max)
    bh.write_to_file()
    t1 = time.time()

    data_file = f"simulation_data_{grid_dim}_{boundary}.data"
    print(f"[OK] Puncture simulation done in {t1 - t0:.2f}s --> {data_file}")

    # =========================================================================
    # STEP 2 -- Plot Solution Surface
    # =========================================================================
    print("\n" + "=" * 68)
    print("  STEP 2 -- Plot Solution Surface (plot_simulation.py)")
    print("=" * 68)

    from plot_simulation import puncture_plot
    import matplotlib
    matplotlib.use("Agg")

    plot_output = "simulation_plot_output.png"
    t0 = time.time()
    puncture_plot(data_file, plot_file=plot_output)
    t1 = time.time()
    print(f"[OK] Plot saved in {t1 - t0:.2f}s --> {plot_output}")

    # =========================================================================
    # STEP 3 -- GRRT Static Render
    # =========================================================================
    print("\n" + "=" * 68)
    print("  STEP 3 -- Static GRRT Render (grrt_black_hole.py)")
    print("=" * 68)

    import numpy as np
    from PIL import Image
    from grrt_black_hole import precompute_geodesics, shade_frame

    M = 1.0
    r_isco = 6.0 * M
    r_out = 26.0 * M
    width, height = 985, 568
    fov, dist, incl = 36.0, 40.0, 85.0
    max_steps = 2200

    t0 = time.time()
    hits, r1, p1, g1, r2, p2, g2 = precompute_geodesics(
        width, height, fov, dist, incl, M, r_isco, r_out, max_steps
    )
    img = shade_frame(hits, r1, p1, g1, r2, p2, g2, M, r_isco, r_out, 0.0)
    rgb = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    grrt_output = "grrt_render_output.png"
    Image.fromarray(rgb).save(grrt_output)
    t1 = time.time()
    print(f"[OK] GRRT static render saved in {t1 - t0:.2f}s --> {grrt_output}")

    # =========================================================================
    # STEP 4 -- Horizon Plunge Integration (grrt_plunge.py)
    # =========================================================================
    print("\n" + "=" * 68)
    print("  STEP 4 -- Horizon Plunge Integration (grrt_plunge.py)")
    print("=" * 68)

    from grrt_plunge import render_plunge_frame
    t0 = time.time()
    img_plunge, r_cam, is_inside = render_plunge_frame(800, 450, 45.0, 32.0, 40.0, 1.0, 6.0, 26.0, 1800)
    rgb_plunge = (np.clip(img_plunge, 0.0, 1.0) * 255).astype(np.uint8)
    plunge_output = "horizon_plunge_render.png"
    Image.fromarray(rgb_plunge).save(plunge_output)
    t1 = time.time()
    print(f"[OK] Horizon crossing render (r={r_cam:.2f}M) saved in {t1 - t0:.2f}s --> {plunge_output}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 68)
    print("  COMPLETE SIMULATION PIPELINE SUCCESSFUL  ")
    print("=" * 68)
    print(f"  Puncture Data:     {data_file}")
    print(f"  3D Solution Plot:  {plot_output}")
    print(f"  Static Raytrace:   {grrt_output}")
    print(f"  Horizon Plunge:    {plunge_output}")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Black Hole Simulation Pipeline & 3-Hour Plunge Suite")
    parser.add_argument("--web", action="store_true", help="Launch interactive 3D Web Simulator in browser")
    parser.add_argument("--plunge", action="store_true", help="Run horizon plunge integration test")
    parser.add_argument("--video-3h", action="store_true", help="Render 3-hour interior plunge MP4 video")
    parser.add_argument("--preview", action="store_true", help="Render 10-second plunge video preview")
    parser.add_argument("--audio", action="store_true", help="Synthesize relaxing black hole cosmic sound & mux to video")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")

    args = parser.parse_args()

    if args.web:
        launch_web_server(args.port)
    elif args.audio:
        from generate_relaxing_sound import generate_relaxing_black_hole_audio, add_audio_to_video
        audio_wav = generate_relaxing_black_hole_audio(duration_sec=120.0, output_wav="black_hole_relaxing_sound.wav")
        if os.path.exists("black_hole_simulation_video.mp4"):
            add_audio_to_video("black_hole_simulation_video.mp4", audio_wav, "black_hole_simulation_relaxing.mp4")
    elif args.plunge:
        from grrt_plunge import main as plunge_main
        sys.argv = ["grrt_plunge.py", "--test-horizon-cross", "--image"]
        plunge_main()
    elif args.video_3h or args.preview:
        from render_3hour_video import render_3hour_plunge_video
        render_3hour_plunge_video(preview=args.preview)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
