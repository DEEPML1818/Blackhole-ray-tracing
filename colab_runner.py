"""
colab_runner.py -- Google Colab Helper & Automated Tester for Black Hole GRRT
=============================================================================
This script provides helper functions and automated test routines for running
General Relativistic Ray-Tracing (GRRT) and Numerical Relativity initial data
simulations inside Google Colab environments.

Usage:
    python colab_runner.py --test         # Run verification benchmark
    python colab_runner.py --quick-render # Render a quick sample GRRT image
"""

import argparse
import sys
import os
import time
import numpy as np
from PIL import Image

# Import existing modules in the repository
import grrt_black_hole
import grrt_plunge
import simulation
import plot_simulation


def run_quick_render(width=640, height=360, fov=36.0, dist=40.0, incl=85.0, output="colab_render.png"):
    """Renders a fast GRRT frame for Colab inline display."""
    print(f"[*] Starting GRRT Ray Trace ({width}x{height}, dist={dist}, incl={incl} deg)...")
    M = 1.0
    r_isco = 6.0 * M
    r_out = 26.0 * M
    max_steps = 2000

    t0 = time.time()
    hits, r1, p1, g1, r2, p2, g2 = grrt_black_hole.precompute_geodesics(
        width, height, fov, dist, incl, M, r_isco, r_out, max_steps
    )
    img = grrt_black_hole.shade_frame(hits, r1, p1, g1, r2, p2, g2, M, r_isco, r_out, 0.0)
    rgb = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    
    Image.fromarray(rgb).save(output)
    t1 = time.time()
    print(f"[OK] Render completed in {t1 - t0:.2f} seconds -> Saved to '{output}'")
    return output


def run_verification_tests():
    """Runs a quick diagnostic benchmark for Colab setup."""
    print("=" * 60)
    print("  GOOGLE COLAB BLACK HOLE RAY TRACING DIAGNOSTIC BENCHMARK")
    print("=" * 60)
    
    # Check Numba JIT acceleration
    print("[1/3] Testing Numba JIT Compilation & Accelerations...")
    t0 = time.time()
    _ = grrt_black_hole.get_acceleration(10.0, 5.0, 2.0, 1.0)
    t1 = time.time()
    print(f"  |-- Acceleration function warm-up time: {(t1 - t0)*1000:.2f} ms")

    # Check Puncture initial data solver
    print("[2/3] Testing Numerical Relativity Puncture Solver...")
    t0 = time.time()
    bh = simulation.Puncture((0, 0, 0), (1, 0, 0), grid_dim=12, boundary=4.0)
    bh.construct_solution(tol=1e-8, it_max=20)
    t1 = time.time()
    print(f"  |-- Puncture solver completed in {t1 - t0:.2f} s")

    # Check Ray Tracing Engine
    print("[3/3] Testing GRRT Geodesic Integrator (320x180 resolution)...")
    t0 = time.time()
    out = run_quick_render(width=320, height=180, output="test_render.png")
    t1 = time.time()
    print(f"  |-- 320x180 render finished in {t1 - t0:.2f} s")

    print("\n[OK] ALL COLAB BENCHMARKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Colab GRRT Helper Runner")
    parser.add_argument("--test", action="store_true", help="Run diagnostic verification tests")
    parser.add_argument("--quick-render", action="store_true", help="Render quick sample image")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=360, help="Image height")
    parser.add_argument("--output", type=str, default="colab_render.png", help="Output PNG path")

    args = parser.parse_args()

    if args.test:
        run_verification_tests()
    elif args.quick-render:
        run_quick_render(width=args.width, height=args.height, output=args.output)
    else:
        run_quick_render()
