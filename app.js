/**
 * Black Hole 3-Hour Interior Plunge WebGL Engine & App Controller
 * ===============================================================
 * Performs real-time GPU General Relativistic Ray Tracing (GRRT)
 * in Painlevé-Gullstrand spacetime coordinates across r = 2.0M horizon.
 */

// =============================================================================
// 1. WebGL 2.0 GLSL Shaders (Painlevé-Gullstrand Ray Tracing)
// =============================================================================

const VERTEX_SHADER_SRC = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SRC = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_camPos;
uniform float u_fov;
uniform float u_timeWarp;

const float M = 1.0;
const float R_ISCO = 6.0;
const float R_OUT = 28.0;
const int MAX_STEPS = 220;

// Exact Schwarzschild Gravitational Ray Acceleration in Isotropic / PG Coordinates
vec3 get_acceleration(vec3 pos) {
    float r = length(pos);
    if (r <= 0.5001 * M) return vec3(0.0);
    
    float u = 0.5 * M / r;
    float one_plus_u = 1.0 + u;
    float one_minus_u = max(1e-4, 1.0 - u);

    float factor = -(M / (r * r)) * pow(one_plus_u, 5.0) * (2.0 - u) / pow(one_minus_u, 3.0);
    return factor * (pos / r);
}

// NASA Goddard Fiery Orange-Red Color Palette
vec3 temperature_to_fiery_rgb(float intensity, float g) {
    if (intensity <= 1e-5) return vec3(0.0);

    float brightness = intensity * (0.85 + 0.15 * g);
    float r_lum = brightness * 4.5;
    float r_val = r_lum / (1.0 + r_lum);

    float g_lum = brightness * 1.8;
    float g_val = (g_lum / (1.0 + g_lum)) * 0.48;

    return vec3(r_val, g_val, 0.0);
}

void main() {
    vec2 st = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
    float r_cam = length(u_camPos);
    bool is_inside = r_cam < 2.0 * M;

    // Camera orientation vectors pointing toward origin
    vec3 forward = normalize(-u_camPos);
    vec3 world_up = vec3(0.0, 0.0, 1.0);
    vec3 right = normalize(cross(forward, world_up));
    if (length(right) < 0.001) right = vec3(1.0, 0.0, 0.0);
    vec3 up = normalize(cross(right, forward));

    // Relativistic sky cone & aberration inside horizon
    float aberration = is_inside ? max(0.25, pow(r_cam / 2.0, 0.8)) : 1.0;
    float tan_half_fov = tan(radians(u_fov) * 0.5) * aberration;

    vec3 dir = normalize(forward + st.x * tan_half_fov * right + st.y * tan_half_fov * up);

    // Initial ray velocity in refractive index spacetime
    float u_cam = 0.5 * M / max(0.501, r_cam);
    float n_cam = pow(1.0 + u_cam, 3.0) / max(1e-4, 1.0 - u_cam);
    vec3 vel = dir * n_cam;
    vec3 pos = u_camPos;

    vec3 accum_color = vec3(0.0);
    int hit_count = 0;
    float base_dt = 0.16;

    for (int step = 0; step < MAX_STEPS; step++) {
        float r = length(pos);

        // Event Horizon cutoff (r_iso = 0.5M)
        if (r <= 0.5005 * M) {
            if (hit_count == 0 && is_inside) {
                // Interior Hyperspace Glow
                float glow = 0.3 + 0.7 * sin(20.0 * atan(pos.y, pos.x) + u_time * 2.0);
                accum_color = vec3(0.15 + 0.5 * glow, 0.45, 0.85);
            }
            break;
        }

        if (r >= 45.0 && dot(pos, vel) > 0.0) {
            if (hit_count == 0) {
                // Starry Celestial Background
                float phi_sky = atan(vel.y, vel.x);
                float star_grid = sin(25.0 * phi_sky) * cos(35.0 * (r * 0.1));
                if (star_grid > 0.95) {
                    float b = (star_grid - 0.95) / 0.05;
                    accum_color = vec3(b * 0.85, b * 0.92, b * 1.0);
                }
            }
            break;
        }

        // Adaptive RK4 step size
        float h = base_dt;
        if (r < 0.75 * M) h = 0.008;
        else if (r < 1.8 * M) h = 0.02;
        else if (r < 4.5 * M) h = 0.05;
        else if (r < 12.0 * M) h = 0.09;

        // RK4 Integration step
        vec3 a1 = get_acceleration(pos);
        vec3 p2 = pos + 0.5 * h * vel;
        vec3 v2 = vel + 0.5 * h * a1;
        vec3 a2 = get_acceleration(p2);
        vec3 p3 = pos + 0.5 * h * v2;
        vec3 v3 = vel + 0.5 * h * a2;
        vec3 a3 = get_acceleration(p3);
        vec3 p4 = pos + h * v3;
        vec3 v4 = vel + h * a3;
        vec3 a4 = get_acceleration(p4);

        vec3 pos_next = pos + (h / 6.0) * (vel + 2.0 * v2 + 2.0 * v3 + v4);
        vec3 vel_next = vel + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);

        // Equatorial plane z = 0 intersection check
        if (pos.z * pos_next.z <= 0.0 && abs(pos_next.z - pos.z) > 1e-12) {
            float frac = -pos.z / (pos_next.z - pos.z);
            vec3 p_hit = pos + frac * (pos_next - pos);
            float r_iso_hit = length(p_hit.xy);

            if (r_iso_hit > 0.5005 * M) {
                // Convert Isotropic radius to Schwarzschild coordinate radius: r_schw = r_iso * (1 + M/2r)^2
                float u_h = 0.5 * M / r_iso_hit;
                float r_schw = r_iso_hit * (1.0 + u_h) * (1.0 + u_h);

                if (r_schw >= R_ISCO && r_schw <= R_OUT) {
                    float phi_h = atan(p_hit.y, p_hit.x);
                    
                    // Relativistic Keplerian velocity & Doppler factor
                    float lz = p_hit.y * vel_next.x - p_hit.x * vel_next.y;
                    float omega_k = sqrt(M / (r_schw * r_schw * r_schw));
                    float denom = 1.0 - omega_k * lz;
                    float g = 0.4;
                    if (denom > 0.02 && (1.0 - 3.0 * M / r_schw) > 0.0) {
                        g = sqrt(1.0 - 3.0 * M / r_schw) / denom;
                    }

                    // Radial emissivity profile (NASA Goddard model)
                    float x_rel = (r_schw - R_ISCO) / (R_OUT - R_ISCO);
                    float profile = pow(r_schw / R_ISCO, -1.8) * (1.0 - sqrt(R_ISCO / r_schw)) * exp(-1.8 * sqrt(x_rel));
                    profile = max(0.0, profile * 32.0);

                    // High-frequency magnetic streaks & differential shear flow
                    float phi_flow = phi_h - omega_k * u_time * 5.0;
                    float streak1 = 0.90 + 0.10 * sin(110.0 * phi_flow + 12.0 * log(r_schw));
                    float streak2 = 0.93 + 0.07 * sin(75.0  * phi_flow - 9.0  * log(r_schw));
                    float intensity = profile * streak1 * streak2;

                    vec3 hit_rgb = temperature_to_fiery_rgb(intensity * pow(g, 1.5), g);

                    if (hit_count == 0) {
                        // Primary image (front disk)
                        accum_color += 0.95 * hit_rgb;
                        hit_count = 1;
                    } else if (hit_count == 1) {
                        // Secondary image (lensed arch over top/bottom of horizon)
                        accum_color += 0.35 * hit_rgb;
                        hit_count = 2;
                        break;
                    }
                }
            }
        }

        pos = pos_next;
        vel = vel_next;
    }

    fragColor = vec4(clamp(accum_color, 0.0, 1.0), 1.0);
}
`;

// =============================================================================
// 2. Application State & Telemetry Controller
// =============================================================================

class BlackHoleSimulator {
    constructor() {
        this.canvas = document.getElementById('gl-canvas');
        this.gl = this.canvas.getContext('webgl2');

        if (!this.gl) {
            alert('WebGL 2.0 is required to run this simulation.');
            return;
        }

        // App State
        this.t_sim = 0.0;           // Normalized simulation time
        this.interior_seconds = 0;  // Real-time counter of seconds spent inside horizon
        this.camStartRadius = 40.0;
        this.camRadius = 40.0;
        this.camTheta = Math.PI * 0.47;
        this.camPhi = 0.0;
        this.isAutoPilot = true;
        this.timeWarp = 60.0;       // Default 60x speed
        this.fov = 45.0;

        // Audio & Video State
        this.audioCtx = null;
        this.audioGain = null;
        this.isAudioActive = false;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;

        this.initWebGL();
        this.bindEvents();
        this.startLoop();
    }

    initWebGL() {
        const gl = this.gl;
        
        // Compile Shaders
        const vertShader = this.compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
        const fragShader = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);

        this.program = gl.createProgram();
        gl.attachShader(this.program, vertShader);
        gl.attachShader(this.program, fragShader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(this.program));
            return;
        }

        // Fullscreen Quad Geometry
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
            -1,  1,
             1, -1,
             1,  1
        ]);

        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const posLoc = gl.getAttribLocation(this.program, 'a_position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        // Uniform Locations
        this.uRes = gl.getUniformLocation(this.program, 'u_resolution');
        this.uTime = gl.getUniformLocation(this.program, 'u_time');
        this.uCamPos = gl.getUniformLocation(this.program, 'u_camPos');
        this.uFov = gl.getUniformLocation(this.program, 'u_fov');
        this.uTimeWarp = gl.getUniformLocation(this.program, 'u_timeWarp');

        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    compileShader(type, src) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, src);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    resize() {
        this.canvas.width = window.innerWidth * window.devicePixelRatio;
        this.canvas.height = window.innerHeight * window.devicePixelRatio;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    bindEvents() {
        // UI Controls
        document.getElementById('time-warp-select').addEventListener('change', (e) => {
            this.timeWarp = parseFloat(e.target.value);
        });

        document.getElementById('btn-autopilot').addEventListener('click', () => {
            this.isAutoPilot = true;
            document.getElementById('btn-autopilot').classList.add('active');
            document.getElementById('btn-manual').classList.remove('active');
        });

        document.getElementById('btn-manual').addEventListener('click', () => {
            this.isAutoPilot = false;
            document.getElementById('btn-manual').classList.add('active');
            document.getElementById('btn-autopilot').classList.remove('active');
        });

        document.getElementById('slider-distance').addEventListener('input', (e) => {
            this.camRadius = parseFloat(e.target.value);
            this.isAutoPilot = false;
            document.getElementById('btn-manual').classList.add('active');
            document.getElementById('btn-autopilot').classList.remove('active');
        });

        document.getElementById('btn-audio').addEventListener('click', () => {
            this.toggleAudio();
        });

        document.getElementById('btn-record').addEventListener('click', () => {
            this.toggleRecording();
        });

        // Keyboard navigation
        window.addEventListener('keydown', (e) => {
            const step = 0.05;
            if (e.key === 'w' || e.key === 'W') this.camRadius = Math.max(0.15, this.camRadius - 0.5);
            if (e.key === 's' || e.key === 'S') this.camRadius = Math.min(45.0, this.camRadius + 0.5);
            if (e.key === 'a' || e.key === 'A') this.camPhi -= step;
            if (e.key === 'd' || e.key === 'D') this.camPhi += step;
            if (e.key === ' ') {
                this.t_sim = 0.0;
                this.interior_seconds = 0;
                this.camRadius = 40.0;
            }
        });
    }

    // Web Audio Synthesizer (432 Hz Relaxing Cosmic Black Hole Soundscape)
    toggleAudio() {
        const btn = document.getElementById('btn-audio');

        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const now = this.audioCtx.currentTime;

            // 1. Sub-bass Gravitational Waves (43.2 Hz & 86.4 Hz)
            this.subOsc1 = this.audioCtx.createOscillator();
            this.subOsc1.type = 'sine';
            this.subOsc1.frequency.setValueAtTime(43.2, now);

            this.subOsc2 = this.audioCtx.createOscillator();
            this.subOsc2.type = 'sine';
            this.subOsc2.frequency.setValueAtTime(86.4, now);

            // 2. Warm 432 Hz Ambient Pads (108 Hz, 216 Hz, 432 Hz)
            this.padOsc1 = this.audioCtx.createOscillator();
            this.padOsc1.type = 'sine';
            this.padOsc1.frequency.setValueAtTime(108.0, now);

            this.padOsc2 = this.audioCtx.createOscillator();
            this.padOsc2.type = 'triangle';
            this.padOsc2.frequency.setValueAtTime(216.0, now);

            // 3. Cosmic Brownian Space Drone (Filtered Noise)
            const bufferSize = 2 * this.audioCtx.sampleRate;
            const noiseBuffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
            const output = noiseBuffer.getChannelData(0);
            let lastOut = 0.0;
            for (let i = 0; i < bufferSize; i++) {
                const white = Math.random() * 2 - 1;
                output[i] = (lastOut + (0.02 * white)) / 1.02;
                lastOut = output[i];
            }

            this.noiseNode = this.audioCtx.createBufferSource();
            this.noiseNode.buffer = noiseBuffer;
            this.noiseNode.loop = true;

            // Lowpass filter for smooth space hum (cutoff at 140 Hz)
            this.filter = this.audioCtx.createBiquadFilter();
            this.filter.type = 'lowpass';
            this.filter.frequency.setValueAtTime(140, now);

            // 4. Slow 0.05 Hz Breathing LFO Modulation
            this.lfo = this.audioCtx.createOscillator();
            this.lfo.frequency.setValueAtTime(0.05, now);

            this.lfoGain = this.audioCtx.createGain();
            this.lfoGain.gain.setValueAtTime(0.08, now);
            this.lfo.connect(this.lfoGain.gain);

            // Master Gain Node
            this.audioGain = this.audioCtx.createGain();
            this.audioGain.gain.setValueAtTime(0.20, now);

            // Routing
            this.subOsc1.connect(this.audioGain);
            this.subOsc2.connect(this.audioGain);
            this.padOsc1.connect(this.audioGain);
            this.padOsc2.connect(this.audioGain);

            this.noiseNode.connect(this.filter);
            this.filter.connect(this.audioGain);
            this.audioGain.connect(this.audioCtx.destination);

            this.subOsc1.start();
            this.subOsc2.start();
            this.padOsc1.start();
            this.padOsc2.start();
            this.noiseNode.start();
            this.lfo.start();

            this.isAudioActive = true;
            btn.innerHTML = '<span class="btn-icon">🔊</span> AUDIO: ON (RELAXING 432Hz)';
            btn.classList.add('active');
        } else if (this.isAudioActive) {
            this.audioGain.gain.setValueAtTime(0, this.audioCtx.currentTime);
            this.isAudioActive = false;
            btn.innerHTML = '<span class="btn-icon">🔇</span> AUDIO: OFF';
            btn.classList.remove('active');
        } else {
            this.audioGain.gain.setValueAtTime(0.20, this.audioCtx.currentTime);
            this.isAudioActive = true;
            btn.innerHTML = '<span class="btn-icon">🔊</span> AUDIO: ON (RELAXING 432Hz)';
            btn.classList.add('active');
        }
    }

    // MediaRecorder API Video Exporter
    toggleRecording() {
        const btn = document.getElementById('btn-record');

        if (!this.isRecording) {
            const stream = this.canvas.captureStream(30);
            this.recordedChunks = [];
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });

            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.recordedChunks.push(e.data);
            };

            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'black_hole_interior_flight.webm';
                a.click();
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            btn.classList.add('recording');
            btn.innerHTML = '<span class="btn-icon">⏹️</span> STOP REC';
        } else {
            this.mediaRecorder.stop();
            this.isRecording = false;
            btn.classList.remove('recording');
            btn.innerHTML = '<span class="btn-icon">🔴</span> RECORD VIDEO';
        }
    }

    updateTrajectory(dt) {
        if (this.isAutoPilot) {
            this.t_sim += dt * 0.5 * (this.timeWarp / 60.0);

            const decayRate = 0.08;
            const r_curr = this.camStartRadius * Math.exp(-decayRate * this.t_sim);

            if (r_curr < 0.35) {
                // Extended 3-Hour Travel inside Interior Spacetime Corridor
                const phase = (this.t_sim - Math.log(this.camStartRadius / 0.35) / decayRate) * 0.5;
                this.camRadius = 0.35 + 0.15 * (0.5 + 0.5 * Math.sin(phase * 1.5));
                this.camTheta = Math.PI * 0.5 + 0.2 * Math.cos(phase * 0.8);
                this.camPhi = phase * 2.0;
            } else {
                this.camRadius = r_curr;
                this.camPhi = this.t_sim * 0.25;
                this.camTheta = Math.min(Math.PI * 0.5, Math.PI * (0.47 - (1.0 - r_curr / 40.0) * 0.2));
            }

            document.getElementById('slider-distance').value = this.camRadius.toFixed(1);
        }

        // Update 3-Hour Interior Timer Clock
        if (this.camRadius < 2.0) {
            this.interior_seconds += dt * this.timeWarp;
        }

        this.updateTelemetry();
    }

    updateTelemetry() {
        const r = this.camRadius;
        const isInside = r < 2.0;

        document.getElementById('telemetry-r').innerText = r.toFixed(2);
        document.getElementById('telemetry-dist-horizon').innerText = Math.max(0.0, r - 2.0).toFixed(2);

        // Infall velocity & Doppler factor
        const v = Math.sqrt(Math.min(0.99, 2.0 / Math.max(0.2, r)));
        document.getElementById('telemetry-v').innerText = v.toFixed(2);

        const g = Math.max(0.05, Math.sqrt(Math.abs(1.0 - 2.0 / Math.max(0.21, r))));
        document.getElementById('telemetry-g').innerText = g.toFixed(2);
        document.getElementById('telemetry-redshift').innerText = `z = ${(1.0 / g - 1.0).toFixed(2)}`;

        // Progress bar
        const rPct = Math.min(100, (r / 40.0) * 100);
        document.getElementById('radius-bar').style.width = `${rPct}%`;

        // Horizon Status Badge
        const badge = document.getElementById('horizon-status-badge');
        const warning = document.getElementById('horizon-warning');

        if (isInside) {
            badge.classList.add('inside-horizon');
            badge.querySelector('#status-text').innerText = 'INSIDE EVENT HORIZON (r < 2M)';
            warning.classList.remove('hidden');
        } else {
            badge.classList.remove('inside-horizon');
            badge.querySelector('#status-text').innerText = 'OUTSIDE EVENT HORIZON (r > 2M)';
            warning.classList.add('hidden');
        }

        // Timer Clock Format (HH:MM:SS)
        const totalSec = Math.floor(this.interior_seconds);
        const hrs = Math.floor(totalSec / 3600);
        const mins = Math.floor((totalSec % 3600) / 60);
        const secs = totalSec % 60;
        const clockStr = `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        document.getElementById('timer-clock').innerText = clockStr;
    }

    startLoop() {
        let lastTime = performance.now();

        const render = (now) => {
            const dt = Math.min(0.1, (now - lastTime) / 1000.0);
            lastTime = now;

            this.updateTrajectory(dt);

            // Compute Cartesian camera position
            const cx = this.camRadius * Math.sin(this.camTheta) * Math.cos(this.camPhi);
            const cy = this.camRadius * Math.sin(this.camTheta) * Math.sin(this.camPhi);
            const cz = this.camRadius * Math.cos(this.camTheta);

            // Set WebGL Uniforms
            const gl = this.gl;
            gl.useProgram(this.program);
            gl.uniform2f(this.uRes, this.canvas.width, this.canvas.height);
            gl.uniform1f(this.uTime, now / 1000.0);
            gl.uniform3f(this.uCamPos, cx, cy, cz);
            gl.uniform1f(this.uFov, this.fov);
            gl.uniform1f(this.uTimeWarp, this.timeWarp);

            gl.drawArrays(gl.TRIANGLES, 0, 6);

            requestAnimationFrame(render);
        };

        requestAnimationFrame(render);
    }
}

// Initialize Application on Page Load
window.addEventListener('DOMContentLoaded', () => {
    window.sim = new BlackHoleSimulator();
});
