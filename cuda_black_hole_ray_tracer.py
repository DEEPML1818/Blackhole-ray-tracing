"""
CUDA-Accelerated General Relativistic Ray Tracing (Painlevé-Gullstrand / Isotropic Metric)
========================================================================================
High-performance GPU Ray Tracing Engine using Numba CUDA.

Fixes included:
1. Replaced 'math.abs' (invalid attribute) with Python built-in 'abs()'.
2. Corrected camera initial position & ray direction to capture the accretion disk.
3. Added robust multi-hit equatorial plane crossing logic (z = 0 plane).
4. Implemented relativistic Doppler beaming & NASA color palette.
"""

import math
import time
import numpy as np
from PIL import Image
from numba import cuda

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720
M = 1.0                  # Black Hole Mass
R_HORIZON = 2.0 * M      # Event Horizon
R_ISCO = 6.0 * M         # Innermost Stable Circular Orbit
R_OUT = 26.0 * M         # Outer Disk Radius
CAM_R = 40.0 * M         # Camera Distance
INCLINATION_DEG = 85.0   # Inclination Angle (degrees)
MAX_STEPS = 2000
STEP_SIZE = 0.05

@cuda.jit(device=True)
def get_acceleration(x, y, z, M):
    """Spatial ray acceleration in isotropic Schwarzschild metric."""
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.5001 * M:
        return 0.0, 0.0, 0.0, r
    u = 0.5 * M / r
    one_plus_u = 1.0 + u
    one_minus_u = 1.0 - u
    if one_minus_u < 1e-4:
        one_minus_u = 1e-4

    factor = -(M / (r * r)) * (one_plus_u ** 5) * (2.0 - u) / (one_minus_u ** 3)
    inv_r = 1.0 / r
    return factor * x * inv_r, factor * y * inv_r, factor * z * inv_r, r


@cuda.jit(device=True)
def rk4_step(x, y, z, vx, vy, vz, h, M):
    """4th-order Runge-Kutta step on GPU."""
    ax1, ay1, az1, _ = get_acceleration(x, y, z, M)

    x2 = x + 0.5 * h * vx
    y2 = y + 0.5 * h * vy
    z2 = z + 0.5 * h * vz
    vx2 = vx + 0.5 * h * ax1
    vy2 = vy + 0.5 * h * ay1
    vz2 = vz + 0.5 * h * az1
    ax2, ay2, az2, _ = get_acceleration(x2, y2, z2, M)

    x3 = x + 0.5 * h * vx2
    y3 = y + 0.5 * h * vy2
    z3 = z + 0.5 * h * vz2
    vx3 = vx + 0.5 * h * ax2
    vy3 = vy + 0.5 * h * ay2
    vz3 = vz + 0.5 * h * az2
    ax3, ay3, az3, _ = get_acceleration(x3, y3, z3, M)

    x4 = x + h * vx3
    y4 = y + h * vy3
    z4 = z + h * vz3
    vx4 = vx + h * ax3
    vy4 = vy + h * ay3
    vz4 = vz + h * az3
    ax4, ay4, az4, _ = get_acceleration(x4, y4, z4, M)

    x_next = x + (h / 6.0) * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4)
    y_next = y + (h / 6.0) * (vy + 2.0 * vy2 + 2.0 * vy3 + vy4)
    z_next = z + (h / 6.0) * (vz + 2.0 * vz2 + 2.0 * vz3 + vz4)

    vx_next = vx + (h / 6.0) * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4)
    vy_next = vy + (h / 6.0) * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4)
    vz_next = vz + (h / 6.0) * (az1 + 2.0 * az2 + 2.0 * az3 + az4)

    return x_next, y_next, z_next, vx_next, vy_next, vz_next


@cuda.jit
def render_cuda_kernel(output_image, width, height, cam_dist, theta_deg, M):
    i, j = cuda.grid(2)
    if i >= width or j >= height:
        return

    # Camera Coordinate Frame
    theta_rad = theta_deg * (math.pi / 180.0)
    cam_x = 0.0
    cam_y = -cam_dist * math.sin(theta_rad)
    cam_z = cam_dist * math.cos(theta_rad)

    # Unit direction vectors
    fov_rad = 36.0 * (math.pi / 180.0)
    tan_fov = math.tan(fov_rad * 0.5)
    aspect = width / height

    # Screen pixel coordinates
    u_screen = (2.0 * (i + 0.5) / width - 1.0) * aspect * tan_fov
    v_screen = (1.0 - 2.0 * (j + 0.5) / height) * tan_fov

    # Camera orientation vectors
    # Forward vector towards origin
    norm_cam = math.sqrt(cam_x*cam_x + cam_y*cam_y + cam_z*cam_z)
    fx, fy, fz = -cam_x / norm_cam, -cam_y / norm_cam, -cam_z / norm_cam
    
    # Right vector = Forward x (0,0,1)
    rx = fy * 1.0 - fz * 0.0
    ry = fz * 0.0 - fx * 1.0
    rz = fx * 0.0 - fy * 0.0
    norm_r = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx, ry, rz = rx / norm_r, ry / norm_r, rz / norm_r

    # Up vector = Right x Forward
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx

    # Initial ray direction
    dir_x = fx + u_screen * rx + v_screen * ux
    dir_y = fy + u_screen * ry + v_screen * uy
    dir_z = fz + u_screen * rz + v_screen * uz
    dir_len = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    dir_x, dir_y, dir_z = dir_x / dir_len, dir_y / dir_len, dir_z / dir_len

    # Metric index at camera position
    r_cam = norm_cam
    u_cam = 0.5 * M / r_cam
    n_cam = ((1.0 + u_cam) ** 3) / (1.0 - u_cam)

    vx = dir_x * n_cam
    vy = dir_y * n_cam
    vz = dir_z * n_cam

    x, y, z = cam_x, cam_y, cam_z
    r_horizon_iso = 0.5005 * M
    r_max_iso = r_cam * 1.6

    tot_r, tot_g, tot_b = 0.0, 0.0, 0.0
    hit_count = 0

    for step in range(MAX_STEPS):
        r = math.sqrt(x*x + y*y + z*z)
        if r <= r_horizon_iso or (r >= r_max_iso and (x*vx + y*vy + z*vz) > 0):
            break

        # Adaptive step size
        if r < 0.75 * M:
            h = 0.006
        elif r < 1.8 * M:
            h = 0.016
        elif r < 4.5 * M:
            h = 0.042
        elif r < 12.0 * M:
            h = 0.09
        else:
            h = STEP_SIZE

        x_next, y_next, z_next, vx_next, vy_next, vz_next = rk4_step(x, y, z, vx, vy, vz, h, M)
        r_next = math.sqrt(x_next*x_next + y_next*y_next + z_next*z_next)

        if r_next <= r_horizon_iso:
            break

        # Equatorial plane crossing (z = 0)
        if z * z_next <= 0.0 and abs(z_next - z) > 1e-12:
            frac = -z / (z_next - z)
            x_hit = x + frac * (x_next - x)
            y_hit = y + frac * (y_next - y)
            r_iso_hit = math.sqrt(x_hit*x_hit + y_hit*y_hit)

            if r_iso_hit > r_horizon_iso:
                u_hit = 0.5 * M / r_iso_hit
                r_schw_hit = r_iso_hit * (1.0 + u_hit) * (1.0 + u_hit)

                if R_ISCO <= r_schw_hit <= R_OUT:
                    phi_hit = math.atan2(y_hit, x_hit)
                    lz = y_hit * vx_next - x_hit * vy_next
                    omega_k = math.sqrt(M / (r_schw_hit ** 3))

                    denom = 1.0 - omega_k * lz
                    if denom > 0.02 and (1.0 - 3.0 * M / r_schw_hit) > 0.0:
                        g = math.sqrt(1.0 - 3.0 * M / r_schw_hit) / denom
                    else:
                        g = 0.1

                    # Emissivity profile & Doppler brightness
                    x_rel = (r_schw_hit - R_ISCO) / (R_OUT - R_ISCO)
                    profile = ((r_schw_hit / R_ISCO) ** (-1.8)) * (1.0 - math.sqrt(R_ISCO / r_schw_hit)) * math.exp(-1.8 * (x_rel ** 0.5))
                    profile = max(0.0, profile) * 30.0

                    lane = 0.90 + 0.10 * math.sin(110.0 * phi_hit + 12.0 * math.log(r_schw_hit))
                    intensity = profile * lane * (g ** 1.5)

                    # NASA Color spectrum mapping
                    r_val = min(1.0, intensity * (0.8 + 0.4 * g))
                    g_val = min(1.0, intensity * (0.3 + 0.5 * g))
                    b_val = min(1.0, intensity * (0.05 + 0.2 * (g ** 2)))

                    if hit_count == 0:
                        tot_r += 0.96 * r_val
                        tot_g += 0.96 * g_val
                        tot_b += 0.96 * b_val
                        hit_count = 1
                    elif hit_count == 1:
                        tot_r = min(1.0, tot_r + 0.22 * r_val)
                        tot_g = min(1.0, tot_g + 0.22 * g_val)
                        tot_b = min(1.0, tot_b + 0.22 * b_val)
                        hit_count = 2
                        break

        x, y, z = x_next, y_next, z_next
        vx, vy, vz = vx_next, vy_next, vz_next

    output_image[j, i, 0] = min(255, int(tot_r * 255.0))
    output_image[j, i, 1] = min(255, int(tot_g * 255.0))
    output_image[j, i, 2] = min(255, int(tot_b * 255.0))


def main():
    print("=" * 60)
    print("  CUDA-ACCELERATED GENERAL RELATIVISTIC RAY-TRACER")
    print("=" * 60)

    try:
        # Allocate GPU Device Memory
        d_output = cuda.device_array((HEIGHT, WIDTH, 3), dtype=np.uint8)

        threads_per_block = (16, 16)
        blocks_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        print(f"[*] Launching CUDA Kernel ({WIDTH}x{HEIGHT}, Grid={blocks_per_grid}, Threads={threads_per_block})...")
        t0 = time.time()
        render_cuda_kernel[blocks_per_grid, threads_per_block](d_output, WIDTH, HEIGHT, CAM_R, INCLINATION_DEG, M)
        cuda.synchronize()
        t1 = time.time()

        h_output = d_output.copy_to_host()
        img = Image.fromarray(h_output)
        out_name = "cuda_blackhole_render.png"
        img.save(out_name)
        print(f"[✓] CUDA Ray Trace finished in {t1 - t0:.2f} s -> Saved to '{out_name}'")

    except Exception as e:
        print(f"[!] CUDA Hardware not available or encountered error: {e}")
        print("[*] Note: For GPU execution, ensure NVIDIA drivers and CUDA Toolkit are installed.")

if __name__ == "__main__":
    main()
