import pandas as pd
import numpy as np
import os
import json
import sys
import uuid
from scipy import ndimage
from scipy.interpolate import griddata
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import threading
import time
import shutil
import zipfile
import glob

app = Flask(__name__,static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Ensure folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Processing status tracking
processing_jobs = {}

# --- DEFAULT SETTINGS ---
DEFAULT_SETTINGS = {
    'input_file': '',
    'base_output_name': 'gpr_iso',
    
    # Column indices (0-based)
    'use_column_indices': True,
    'col_idx_x': 0,
    'col_idx_y': 1,
    'col_idx_z': 7,
    'col_idx_amplitude': 8,
    
    # Filtering
    'threshold_percentile': 0.95,
    'iso_bins': 5,
    
    # Depth offset
    'depth_offset_per_level': 0.05,
    
    # VR settings
    'vr_point_size': 0.015,
    
    # Coordinate settings
    'invert_depth': True,
    'center_coordinates': True,
    
    # Surface settings
    'generate_surface': True,
    'surface_resolution': 100,
    'surface_depth_slices': 5,
    'surface_opacity': 0.6,
    'generate_amplitude_surface': True,
    
    # Downsampling
    'max_points_per_layer': 500000
}

def write_ply_fast(filename, points, colors):
    """Write points and colors to a PLY file"""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    data = np.column_stack([points, colors.astype(np.uint8)])
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header)
        np.savetxt(f, data, fmt='%.6f %.6f %.6f %d %d %d')

def write_obj_mesh(filename, vertices, faces, vertex_colors=None):
    """Write mesh to OBJ file with optional vertex colors"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# OBJ file with {len(vertices)} vertices and {len(faces)} faces\n")
        
        # Write vertices (with colors as comments for reference)
        for i, v in enumerate(vertices):
            if vertex_colors is not None:
                c = vertex_colors[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def generate_surface_mesh(df, x_col, y_col, z_col, amp_col, resolution=100):
    """Generate a surface mesh from GPR data"""
    print("  Generating surface mesh...")
    
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    amp = df[amp_col].values
    
    # Create regular grid
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate Z values (use max amplitude depth at each point)
    print("    Interpolating surface...")
    
    # For each grid cell, find the depth with maximum amplitude
    zi_grid = np.zeros_like(xi_grid)
    amp_grid = np.zeros_like(xi_grid)
    
    # Use griddata for interpolation
    points = np.column_stack((x, y))
    
    try:
        zi_grid = griddata(points, z, (xi_grid, yi_grid), method='linear', fill_value=z.mean())
        amp_grid = griddata(points, amp, (xi_grid, yi_grid), method='linear', fill_value=0)
    except Exception as e:
        print(f"    Warning: Interpolation issue - {e}")
        zi_grid = np.full_like(xi_grid, z.mean())
        amp_grid = np.full_like(xi_grid, amp.mean())
    
    # Smooth the surface
    zi_grid = ndimage.gaussian_filter(zi_grid, sigma=1)
    amp_grid = ndimage.gaussian_filter(amp_grid, sigma=1)
    
    # Create vertices
    vertices = []
    vertex_colors = []
    
    # Normalize amplitude for coloring
    amp_min, amp_max = amp_grid.min(), amp_grid.max()
    if amp_max > amp_min:
        amp_norm = (amp_grid - amp_min) / (amp_max - amp_min)
    else:
        amp_norm = np.zeros_like(amp_grid)
    
    for i in range(resolution):
        for j in range(resolution):
            vertices.append([xi_grid[i, j], yi_grid[i, j], zi_grid[i, j]])
            
            # Color based on amplitude (red = high, blue = low)
            amp_val = amp_norm[i, j]
            r = amp_val
            g = 0.2
            b = 1.0 - amp_val
            vertex_colors.append([r, g, b])
    
    vertices = np.array(vertices)
    vertex_colors = np.array(vertex_colors)
    
    # Create faces (triangles)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # Two triangles per grid cell
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    
    faces = np.array(faces)
    
    print(f"    Created mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    return vertices, faces, vertex_colors, {
        'x_range': [float(x.min()), float(x.max())],
        'y_range': [float(y.min()), float(y.max())],
        'z_range': [float(zi_grid.min()), float(zi_grid.max())],
        'resolution': resolution
    }

def generate_depth_slices(df, x_col, y_col, z_col, amp_col, num_slices=5, resolution=50):
    """Generate horizontal slice surfaces at different depths"""
    print(f"  Generating {num_slices} depth slices...")
    
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    amp = np.abs(df[amp_col].values)
    
    z_min, z_max = z.min(), z.max()
    slice_depths = np.linspace(z_max, z_min, num_slices + 2)[1:-1]  # Exclude top and bottom
    
    slices = []
    
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    for depth in slice_depths:
        # Find points near this depth
        depth_tolerance = (z_max - z_min) / (num_slices * 2)
        mask = np.abs(z - depth) < depth_tolerance
        
        if mask.sum() < 10:
            continue
        
        # Interpolate amplitude at this depth
        points = np.column_stack((x[mask], y[mask]))
        amp_slice = amp[mask]
        
        try:
            amp_grid = griddata(points, amp_slice, (xi_grid, yi_grid), method='linear', fill_value=0)
            amp_grid = ndimage.gaussian_filter(amp_grid, sigma=1)
        except:
            amp_grid = np.zeros_like(xi_grid)
        
        # Normalize for coloring
        amp_max = amp_grid.max()
        if amp_max > 0:
            amp_norm = amp_grid / amp_max
        else:
            amp_norm = np.zeros_like(amp_grid)
        
        # Create vertices
        vertices = []
        vertex_colors = []
        
        for i in range(resolution):
            for j in range(resolution):
                vertices.append([xi_grid[i, j], yi_grid[i, j], depth])
                
                amp_val = amp_norm[i, j]
                r = amp_val
                g = 0.3 * (1 - amp_val)
                b = 1.0 - amp_val
                vertex_colors.append([r, g, b])
        
        # Create faces
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                faces.append([idx, idx + 1, idx + resolution])
                faces.append([idx + 1, idx + resolution + 1, idx + resolution])
        
        slices.append({
            'depth': float(depth),
            'vertices': np.array(vertices),
            'faces': np.array(faces),
            'colors': np.array(vertex_colors)
        })
        
        print(f"    Slice at depth {depth:.3f}: {len(vertices)} vertices")
    
    return slices

def create_iso_colormap(iso_level, total_levels):
    """Create distinct colors for different ISO levels"""
    color_palette = [
        [255, 0, 0],      # Red
        [255, 165, 0],    # Orange
        [255, 255, 0],    # Yellow
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
    ]
    return color_palette[iso_level % len(color_palette)]

def generate_layer_loaders(ply_files, amplitude_ranges, output_dir, job_id):
    """Generate JavaScript code to load PLY layers"""
    loaders = []
    for i, ply_file in enumerate(ply_files):
        amp_min, amp_max = amplitude_ranges[i]
        filename = os.path.basename(ply_file)
        loaders.append(f'''
        layerPromises.push(
            new Promise((resolve) => {{
                plyLoader.load('/files/{job_id}/{filename}', (geometry) => {{
                    // Don't center the geometry - keep it in world coordinates
                    const material = new THREE.PointsMaterial({{
                        size: pointSize,
                        vertexColors: true,
                        sizeAttenuation: true
                    }});
                    
                    const points = new THREE.Points(geometry, material);
                    points.userData.layerIndex = {i};
                    points.userData.amplitudeMin = {amp_min};
                    points.userData.amplitudeMax = {amp_max};
                    pointCloudGroup.add(points);
                    layers.push(points);
                    loadedCount++;
                    updateLoadingProgress((loadedCount / totalFiles) * 100, 'Loaded layer {i+1}');
                    resolve();
                }},
                undefined,
                (error) => {{
                    console.error('Error loading layer {i+1}:', error);
                    loadedCount++;
                    resolve();
                }});
            }})
        );''')
    return '\n'.join(loaders)
def create_vr_viewer(ply_files, layer_info, output_dir, settings, data_info, job_id,
                     has_surface=False, surface_info=None, num_slices=0, total_files=0):
    """Create a WebXR VR viewer with Surface, 6-DoF Floor interaction, Data Axes, and Compass HUD"""
    
    # Fix for potential missing total_files argument
    if total_files == 0:
        total_files = len(ply_files)

    amplitude_ranges = []
    for i in range(len(ply_files)):
        amp_min = data_info['amp_min'] + (i / len(ply_files)) * (data_info['amp_max'] - data_info['amp_min'])
        amp_max = data_info['amp_min'] + ((i + 1) / len(ply_files)) * (data_info['amp_max'] - data_info['amp_min'])
        amplitude_ranges.append((amp_min, amp_max))
    
    layer_loaders_js = generate_layer_loaders(ply_files, amplitude_ranges, output_dir, job_id)
    
    # Surface loading code
    surface_loader_js = ""
    if has_surface:
        surface_loader_js = f'''
        objLoader.load('/files/{job_id}/surface_amplitude.obj', (object) => {{
            object.traverse((child) => {{
                if (child.isMesh) {{
                    child.material = new THREE.MeshStandardMaterial({{
                        vertexColors: true, side: THREE.DoubleSide, transparent: true,
                        opacity: surfaceOpacity, metalness: 0.1, roughness: 0.8
                    }});
                    child.userData.isSurface = true; surfaces.push(child);
                }}
            }});
            surfaceGroup.add(object);
            console.log('Loaded amplitude surface');
        }}, undefined, (error) => {{ console.log('No amplitude surface found'); }});
        '''
    
    # Slice loading code
    slice_loader_js = ""
    if num_slices > 0:
        for i in range(num_slices):
            slice_loader_js += f'''
            objLoader.load('/files/{job_id}/slice_{i+1}.obj', (object) => {{
                object.traverse((child) => {{
                    if (child.isMesh) {{
                        child.material = new THREE.MeshStandardMaterial({{
                            vertexColors: true, side: THREE.DoubleSide, transparent: true,
                            opacity: sliceOpacity, metalness: 0.1, roughness: 0.8
                        }});
                        child.userData.isSlice = true; child.userData.sliceIndex = {i}; slices.push(child);
                    }}
                }});
                sliceGroup.add(object);
                console.log('Loaded slice {i+1}');
            }}, undefined, (error) => {{ console.log('Slice {i+1} not found'); }});
            '''
    
    x_len = data_info['x_max'] - data_info['x_min']
    y_len = data_info['y_max'] - data_info['y_min']
    z_len = abs(data_info['z_max'] - data_info['z_min'])
    
    # Determine ground plane size (make it 2x larger than data area)
    ground_size = max(x_len, y_len) * 3
    if ground_size < 50: ground_size = 50

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPR VR Explorer - {data_info['original_filename']}</title>
    
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
        }}
    }}
    </script>
    
    <style>
        body {{ margin: 0; padding: 0; background: #1a1a1a; color: white; font-family: Arial, sans-serif; overflow: hidden; }}
        #container {{ position: relative; width: 100vw; height: 100vh; }}
        
        /* UI OVERLAYS */
        #info {{
            position: absolute; top: 20px; left: 20px;
            background: rgba(0,0,0,0.9); padding: 20px;
            border-radius: 10px; z-index: 100;
            max-width: 400px; max-height: 90vh; overflow-y: auto;
        }}
        
        /* DESKTOP COMPASS STYLE */
        #compass-ui {{
            position: absolute; top: 80px; right: 20px;
            width: 120px; height: 120px; z-index: 100;
            pointer-events: none; /* Allows clicking through to 3D scene */
        }}
        
        #vr-button {{
            position: absolute; bottom: 20px; left: 50%;
            transform: translateX(-50%); padding: 15px 30px;
            font-size: 18px; background: #4CAF50; color: white;
            border: none; border-radius: 10px; cursor: pointer; z-index: 100;
        }}
        #vr-button:hover {{ background: #45a049; }}
        #vr-button:disabled {{ background: #666; cursor: not-allowed; }}
        
        .slider-container {{ margin: 12px 0; }}
        .slider-label {{ display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 13px; }}
        input[type="range"] {{ width: 100%; }}
        .toggle-container {{ display: flex; align-items: center; margin: 8px 0; font-size: 13px; }}
        .toggle-container input {{ margin-right: 10px; }}
        #layer-list {{ max-height: 100px; overflow-y: auto; font-size: 11px; margin-top: 5px; }}
        .section-header {{ background: rgba(255,255,255,0.1); padding: 8px; margin: 10px -10px; font-weight: bold; font-size: 13px; }}
        
        #loading {{
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.95); display: flex; flex-direction: column;
            justify-content: center; align-items: center; z-index: 1000;
        }}
        #loading-text {{ font-size: 24px; margin-bottom: 20px; }}
        #loading-progress {{ width: 300px; height: 20px; background: #333; border-radius: 10px; overflow: hidden; }}
        #loading-bar {{ width: 0%; height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); transition: width 0.3s; }}
        #loading-details {{ margin-top: 10px; font-size: 14px; color: #aaa; }}
        
        #debug {{ position: absolute; bottom: 80px; left: 20px; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px; font-size: 11px; font-family: monospace; z-index: 100; }}
        #data-info {{ background: rgba(0,100,200,0.2); padding: 8px; border-radius: 5px; margin-bottom: 10px; font-size: 11px; }}
        .control-group {{ background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .control-group-title {{ font-weight: bold; margin-bottom: 8px; color: #4CAF50; font-size: 12px; }}
        .file-info {{ font-size: 10px; color: #aaa; margin-top: 5px; border-top: 1px solid #333; padding-top: 5px; }}
    </style>
</head>
<body>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="/static/logo.jpeg" alt="GPR VR Viewer Logo" style="max-width: 350px; height: 70px; float:right">
    </div>
    
    <div id="container"></div>
    
    <div id="compass-ui">
        <canvas id="compassCanvas" width="120" height="120"></canvas>
    </div>
    
    <div id="loading">
        <div id="loading-text">Loading GPR Data...</div>
        <div id="loading-progress"><div id="loading-bar"></div></div>
        <div id="loading-details">Preparing {data_info['total_points']:,} points + surfaces...</div>
    </div>
    
    <div id="info">
        <h3 style="margin-top: 0;">GPR VR Explorer</h3>
        
        <div id="data-info">
            <strong>File:</strong> {data_info['original_filename']}<br>
            <strong>Dimensions:</strong> {x_len:.2f}m x {y_len:.2f}m x {z_len:.2f}m<br>
        </div>
        
        <div class="control-group">
            <div class="control-group-title">POINT CLOUD</div>
            <div class="toggle-container">
                <input type="checkbox" id="showPoints" checked>
                <label for="showPoints">Show Points</label>
            </div>
            <div class="slider-container">
                <div class="slider-label"><span>Visible Layers:</span><span id="layerValue">{len(ply_files)}</span></div>
                <input type="range" id="layerSlider" min="0" max="{len(ply_files)}" value="{len(ply_files)}">
            </div>
            <div class="slider-container">
                <div class="slider-label"><span>Point Size:</span><span id="sizeValue">{settings['vr_point_size']}</span></div>
                <input type="range" id="sizeSlider" min="0.002" max="0.05" step="0.002" value="{settings['vr_point_size']}">
            </div>
            <div id="layer-list">{layer_info}</div>
        </div>
        
        <div class="control-group">
            <div class="control-group-title">SURFACES</div>
            <div class="toggle-container">
                <input type="checkbox" id="showSurface" checked>
                <label for="showSurface">Show Amplitude Surface</label>
            </div>
            <div class="slider-container">
                <div class="slider-label"><span>Surface Opacity:</span><span id="surfaceOpacityValue">{settings['surface_opacity']}</span></div>
                <input type="range" id="surfaceOpacitySlider" min="0.1" max="1" step="0.1" value="{settings['surface_opacity']}">
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showSlices" checked>
                <label for="showSlices">Show Depth Slices ({num_slices})</label>
            </div>
            <div class="slider-container">
                <div class="slider-label"><span>Slice Opacity:</span><span id="sliceOpacityValue">0.5</span></div>
                <input type="range" id="sliceOpacitySlider" min="0.1" max="1" step="0.1" value="0.5">
            </div>
        </div>
        
        <div class="control-group">
            <div class="control-group-title">VIEW</div>
            <div class="slider-container">
                <div class="slider-label"><span>Scale:</span><span id="scaleValue">1.0</span></div>
                <input type="range" id="scaleSlider" min="0.1" max="5" step="0.1" value="1">
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showGround" checked>
                <label for="showGround">Show Ground Image</label>
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showAxes" checked>
                <label for="showAxes">Show Data Axes</label>
            </div>
            <div class="toggle-container">
                <button onclick="resetPosition()">Reset Position to Floor</button>
            </div>
        </div>
        
        <div class="control-group" style="font-size: 10px;">
            <div class="control-group-title">VR INSTRUCTIONS</div>
            <strong>GRIP BUTTON:</strong> Grab and rotate the model (6-DoF).<br>
            <strong>TWO HANDS:</strong> Pull apart to zoom/scale.<br>
            <strong>LEFT TRIGGER:</strong> Cycle Layers | <strong>RIGHT TRIGGER:</strong> Reset.
        </div>
        
        <div class="file-info">
            Processed on: {data_info['processing_date']}<br>
            <a href="/" style="color: #4CAF50;">← Process another file</a>
        </div>
    </div>
    
    <div id="debug">Ready</div>
    <button id="vr-button" disabled>Loading...</button>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ XRControllerModelFactory }} from 'three/addons/webxr/XRControllerModelFactory.js';
        import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';
        import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';

        // --- FIXED COMPASS DRAWING FUNCTION ---
        // Arrows now rotate WITH the compass
        function drawCompassOnContext(ctx, angle, size) {{
            const cx = size/2, cy = size/2, r = size * 0.42;
            
            ctx.clearRect(0, 0, size, size);
            
            // 1. GLOBAL ROTATION (Applies to Ring and Needle)
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(angle); // Rotate based on camera angle
            
            // Outer Ring
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(60, 60, 60, 0.9)'; // Dark Grey
            ctx.fill();
            ctx.lineWidth = size * 0.04;
            ctx.strokeStyle = '#888';
            ctx.stroke();
            
            // Inner Circle Background
            ctx.beginPath();
            ctx.arc(0, 0, r * 0.9, 0, Math.PI * 2);
            ctx.fillStyle = '#222';
            ctx.fill();

            // NEEDLE DRAWING (Rotating with the context)
            const needleLen = r * 0.75;
            const needleWide = r * 0.15;

            // North Pointer (Red)
            ctx.beginPath();
            ctx.moveTo(0, -needleLen); // Tip
            ctx.lineTo(needleWide, 0); // Right Base
            ctx.lineTo(-needleWide, 0); // Left Base
            ctx.closePath();
            ctx.fillStyle = '#FF3333';
            ctx.fill();

            // South Pointer (White)
            ctx.beginPath();
            ctx.moveTo(0, needleLen); // Tip
            ctx.lineTo(needleWide, 0); // Right Base
            ctx.lineTo(-needleWide, 0); // Left Base
            ctx.closePath();
            ctx.fillStyle = '#EEEEEE';
            ctx.fill();
            
            // "N" Label (Fixed to North pointer)
            ctx.save();
            ctx.translate(0, -r + (size * 0.1)); 
            ctx.rotate(0); // Counter-rotate if you wanted text upright, but here we want it attached to North
            ctx.fillStyle = 'white';
            ctx.font = 'bold ' + (size * 0.2) + 'px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.shadowColor="black"; ctx.shadowBlur=4;
            ctx.fillText('N', 0, -5);
            ctx.restore();
            
            // Center Pin
            ctx.beginPath();
            ctx.arc(0, 0, size * 0.05, 0, Math.PI * 2);
            ctx.fillStyle = '#GOLD';
            ctx.fill();

            ctx.restore(); // END ROTATION
            
            // 2. STATIC FRAME (Optional crosshairs or decoration outside rotation)
            ctx.save();
            ctx.translate(cx, cy);
            ctx.strokeStyle = "rgba(255,255,255,0.3)";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, -r); ctx.lineTo(0, -r-5); ctx.stroke(); // Tick mark top
            ctx.restore();
        }}

        const compassCanvas = document.getElementById('compassCanvas');
        const compassCtx = compassCanvas.getContext('2d');
        // Initial Draw
        drawCompassOnContext(compassCtx, 0, 120);

        // --- THREE JS SETUP ---
        const debugEl = document.getElementById('debug');
        const loadingBar = document.getElementById('loading-bar');
        const loadingDetails = document.getElementById('loading-details');
        
        function debug(msg) {{ debugEl.textContent = msg; console.log(msg); }}
        function updateLoadingProgress(pct, txt) {{
            loadingBar.style.width = pct + '%';
            if (txt) loadingDetails.textContent = txt;
        }}

        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 1000);
        camera.position.set(0, 1.7, 2);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.xr.enabled = true;
        renderer.shadowMap.enabled = true;
        document.getElementById('container').appendChild(renderer.domElement);
        
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 0, 0);
        controls.maxPolarAngle = Math.PI; 
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 5);
        dirLight.castShadow = true;
        scene.add(dirLight);
        
        // --- GROUND SURFACE IMAGE ---
        // Replaced GridHelper with Textured Plane
        const texLoader = new THREE.TextureLoader();
        const groundGroup = new THREE.Group();
        scene.add(groundGroup);

        texLoader.load(
            '/static/ground.jpg', // UPDATED LOCATION
            function (texture) {{
                const geometry = new THREE.PlaneGeometry({ground_size}, {ground_size});
                const material = new THREE.MeshBasicMaterial({{ 
                    map: texture, 
                    side: THREE.DoubleSide,
                    color: 0xffffff
                }});
                const plane = new THREE.Mesh(geometry, material);
                plane.rotation.x = -Math.PI / 2;
                plane.position.y = -0.05; // Slightly below data to prevent z-fighting
                groundGroup.add(plane);
                console.log("Ground image loaded");
            }},
            undefined,
            function (err) {{
                console.log("Ground image not found at /static/ground.jpeg, using backup grid");
                // Backup Grid if image fails
                const grid = new THREE.GridHelper(50, 50, 0x666666, 0x444444);
                groundGroup.add(grid);
            }}
        );
        
        // --- DATA CONTAINER ---
        const mainGroup = new THREE.Group();
        scene.add(mainGroup);
        
        // Floor Position (y=0), Rotate x -90deg
        mainGroup.position.set(0, 0.05, 0); 
        mainGroup.rotation.x = -Math.PI / 2;
        
        const initialTransform = {{
            position: mainGroup.position.clone(),
            rotation: mainGroup.rotation.clone(),
            scale: mainGroup.scale.clone()
        }};
        
        window.resetPosition = function() {{
            mainGroup.position.copy(initialTransform.position);
            mainGroup.rotation.copy(initialTransform.rotation);
            mainGroup.scale.copy(initialTransform.scale);
            document.getElementById('scaleSlider').value = 1;
            document.getElementById('scaleValue').textContent = "1.0";
            controls.reset();
        }};
        
        const pointCloudGroup = new THREE.Group();
        mainGroup.add(pointCloudGroup);
        const surfaceGroup = new THREE.Group();
        mainGroup.add(surfaceGroup);
        const sliceGroup = new THREE.Group();
        mainGroup.add(sliceGroup);
        const axesGroup = new THREE.Group();
        mainGroup.add(axesGroup);
        
        // Settings
        const layers = [];
        const surfaces = [];
        const slices = [];
        let pointSize = {settings['vr_point_size']};
        let visibleLayerCount = {len(ply_files)};
        let surfaceOpacity = {settings['surface_opacity']};
        let sliceOpacity = 0.5;

        // --- CUSTOM AXIS GENERATOR ---
        function createTextSprite(text, color) {{
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 128; canvas.height = 64;
            ctx.fillStyle = color;
            ctx.font = "Bold 40px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(text, 0, 32);
            
            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({{ map: texture, depthTest: false }});
            const sprite = new THREE.Sprite(material);
            sprite.scale.set(0.5, 0.25, 1);
            return sprite;
        }}

        function buildDataAxes() {{
            const xLen = {x_len}, yLen = {y_len}, zLen = {z_len};
            const xMin = -xLen / 2, yMin = -yLen / 2, zMin = -zLen / 2;
            
            // X Axis (Red)
            const xMat = new THREE.LineBasicMaterial({{ color: 0xff0000, linewidth: 2 }});
            const xPoints = [new THREE.Vector3(xMin, yMin, zMin), new THREE.Vector3(xMin + xLen, yMin, zMin)];
            axesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(xPoints), xMat));
            const xLabel = createTextSprite("X: " + xLen.toFixed(2) + "m", "#ff0000"); xLabel.position.set(0, yMin - 0.2, zMin); axesGroup.add(xLabel);

            // Y Axis (Green)
            const yMat = new THREE.LineBasicMaterial({{ color: 0x00ff00, linewidth: 2 }});
            const yPoints = [new THREE.Vector3(xMin, yMin, zMin), new THREE.Vector3(xMin, yMin + yLen, zMin)];
            axesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(yPoints), yMat));
            const yLabel = createTextSprite("Y: " + yLen.toFixed(2) + "m", "#00ff00"); yLabel.position.set(xMin - 0.3, 0, zMin); axesGroup.add(yLabel);

            // Z Axis (Blue)
            const zMat = new THREE.LineBasicMaterial({{ color: 0x0088ff, linewidth: 2 }});
            const zPoints = [new THREE.Vector3(xMin, yMin, zMin), new THREE.Vector3(xMin, yMin, zMin + zLen)];
            axesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(zPoints), zMat));
            const zLabel = createTextSprite("Z: " + zLen.toFixed(2) + "m", "#0088ff"); zLabel.position.set(xMin, yMin - 0.2, 0); axesGroup.add(zLabel);
            
            // Bounding box
            const boxGeo = new THREE.BoxGeometry(xLen, yLen, zLen);
            const boxWire = new THREE.WireframeGeometry(boxGeo);
            const boxLine = new THREE.LineSegments(boxWire);
            boxLine.material.depthTest = false; boxLine.material.opacity = 0.2; boxLine.material.transparent = true;
            axesGroup.add(boxLine);
        }}
        buildDataAxes();

        // --- VR HUD SETUP (Floating 3D Compass) ---
        // 1. Create off-screen canvas for VR texture
        const vrHudCanvas = document.createElement('canvas');
        vrHudCanvas.width = 256; vrHudCanvas.height = 256;
        const vrHudCtx = vrHudCanvas.getContext('2d');
        const vrHudTexture = new THREE.CanvasTexture(vrHudCanvas);
        
        // 2. Create Sprite
        const vrHudMaterial = new THREE.SpriteMaterial({{ map: vrHudTexture, depthTest: false, depthWrite: false }});
        const vrHudSprite = new THREE.Sprite(vrHudMaterial);
        
        // 3. Position: Top right relative to camera, slightly forward
        // Note: We add this to the camera when VR starts
        vrHudSprite.position.set(0.15, 0.15, -0.5); 
        vrHudSprite.scale.set(0.15, 0.15, 1);

        // Loaders
        const plyLoader = new PLYLoader();
        const objLoader = new OBJLoader();
        let loadedCount = 0;
        const totalFiles = {total_files};
        const layerPromises = [];
        
        {layer_loaders_js}
        {surface_loader_js}
        {slice_loader_js}
        
        Promise.all(layerPromises).then(() => {{
            document.getElementById('loading').style.display = 'none';
            debug('Ready');
        }});

        // UI Event Listeners
        document.getElementById('layerSlider').addEventListener('input', (e) => {{
            visibleLayerCount = parseInt(e.target.value);
            document.getElementById('layerValue').textContent = visibleLayerCount;
            updateLayerVisibility();
        }});
        document.getElementById('sizeSlider').addEventListener('input', (e) => {{
            pointSize = parseFloat(e.target.value);
            document.getElementById('sizeValue').textContent = pointSize.toFixed(3);
            layers.forEach(l => l.material.size = pointSize);
        }});
        document.getElementById('scaleSlider').addEventListener('input', (e) => {{
            const val = parseFloat(e.target.value);
            document.getElementById('scaleValue').textContent = val.toFixed(1);
            mainGroup.scale.setScalar(val);
        }});
        document.getElementById('showPoints').addEventListener('change', (e) => pointCloudGroup.visible = e.target.checked);
        document.getElementById('showGround').addEventListener('change', (e) => groundGroup.visible = e.target.checked);
        document.getElementById('showAxes').addEventListener('change', (e) => axesGroup.visible = e.target.checked);
        document.getElementById('surfaceOpacitySlider').addEventListener('input', (e) => {{
            surfaceOpacity = parseFloat(e.target.value);
            document.getElementById('surfaceOpacityValue').textContent = surfaceOpacity;
            surfaces.forEach(s => s.material.opacity = surfaceOpacity);
        }});
        document.getElementById('sliceOpacitySlider').addEventListener('input', (e) => {{
            sliceOpacity = parseFloat(e.target.value);
            document.getElementById('sliceOpacityValue').textContent = sliceOpacity;
            slices.forEach(s => s.material.opacity = sliceOpacity);
        }});
        document.getElementById('showSurface').addEventListener('change', (e) => surfaceGroup.visible = e.target.checked);
        document.getElementById('showSlices').addEventListener('change', (e) => sliceGroup.visible = e.target.checked);

        function updateLayerVisibility() {{
            layers.forEach((layer, index) => {{
                layer.visible = index < visibleLayerCount;
            }});
        }}

        // VR Setup
        const controllerModelFactory = new XRControllerModelFactory();
        const cameraRig = new THREE.Group();
        scene.add(cameraRig);
        
        const controller0 = renderer.xr.getController(0);
        const controller1 = renderer.xr.getController(1);
        cameraRig.add(controller0);
        cameraRig.add(controller1);
        
        [controller0, controller1].forEach(c => {{
            const grp = renderer.xr.getControllerGrip(c === controller0 ? 0 : 1);
            grp.add(controllerModelFactory.createControllerModel(grp));
            cameraRig.add(grp);
            // Ray for pointing
            c.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,-1)]),
                new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.5 }})
            ));
        }});

        // --- 6-DOF INTERACTION LOGIC ---
        const grabbingControllers = new Set();
        const previousTransforms = new Map();
        
        function onSqueezeStart(event) {{
            const controller = event.target;
            grabbingControllers.add(controller);
            const controllerInv = controller.matrixWorld.clone().invert();
            const relativeMatrix = new THREE.Matrix4().multiplyMatrices(controllerInv, mainGroup.matrixWorld);
            previousTransforms.set(controller, {{
                relativeMatrix: relativeMatrix,
                startDist: grabbingControllers.size === 2 ? 
                    controller0.position.distanceTo(controller1.position) : 0,
                startScale: mainGroup.scale.x
            }});
        }}
        
        function onSqueezeEnd(event) {{
            const releasedController = event.target;
            grabbingControllers.delete(releasedController);
            previousTransforms.delete(releasedController);
            
            // --- FIX START: RE-BIND REMAINING CONTROLLER ---
            // If we drop from 2 hands (scaling) to 1 hand, we must
            // update the remaining controller's reference to the 
            // CURRENT (zoomed) state so it doesn't snap back.
            if (grabbingControllers.size === 1) {{
                const remainingController = grabbingControllers.values().next().value;
                const controllerInv = remainingController.matrixWorld.clone().invert();
                const relativeMatrix = new THREE.Matrix4().multiplyMatrices(controllerInv, mainGroup.matrixWorld);
                
                previousTransforms.set(remainingController, {{
                    relativeMatrix: relativeMatrix,
                    startDist: 0, 
                    startScale: mainGroup.scale.x // Capture the NEW zoomed scale
                }});
            }}
            // --- FIX END ---
        }}
        
        controller0.addEventListener('squeezestart', onSqueezeStart);
        controller0.addEventListener('squeezeend', onSqueezeEnd);
        controller1.addEventListener('squeezestart', onSqueezeStart);
        controller1.addEventListener('squeezeend', onSqueezeEnd);
        
        controller0.addEventListener('selectstart', () => {{
            visibleLayerCount = (visibleLayerCount % {len(ply_files)}) + 1;
            updateLayerVisibility();
            document.getElementById('layerSlider').value = visibleLayerCount;
            document.getElementById('layerValue').textContent = visibleLayerCount;
        }});
        
        controller1.addEventListener('selectstart', resetPosition);

        // VR Button
        const vrButton = document.getElementById('vr-button');
        if ('xr' in navigator) {{
            navigator.xr.isSessionSupported('immersive-vr').then(ok => {{
                if (ok) {{
                    vrButton.disabled = false; vrButton.textContent = 'Enter VR';
                    vrButton.onclick = async () => {{
                        const session = await navigator.xr.requestSession('immersive-vr', {{
                            optionalFeatures: ['local-floor', 'bounded-floor']
                        }});
                        renderer.xr.setSession(session);
                        // Add Floating Compass HUD to Camera
                        camera.add(vrHudSprite);
                        
                        vrButton.textContent = 'VR Active'; vrButton.disabled = true;
                        session.addEventListener('end', () => {{ 
                            vrButton.textContent = 'Enter VR'; vrButton.disabled = false;
                            camera.remove(vrHudSprite); // Remove HUD when exiting VR
                        }});
                    }};
                }} else vrButton.textContent = 'VR Not Supported';
            }});
        }} else vrButton.textContent = 'WebXR Not Available';

        // --- RENDER LOOP ---
        renderer.setAnimationLoop(() => {{
            controls.update();
            
            // 6-DoF Update
            if (grabbingControllers.size === 1) {{
                const controller = grabbingControllers.values().next().value;
                const data = previousTransforms.get(controller);
                if (data) {{
                    const newMatrix = controller.matrixWorld.clone().multiply(data.relativeMatrix);
                    const pos = new THREE.Vector3(); const quat = new THREE.Quaternion(); const scale = new THREE.Vector3();
                    newMatrix.decompose(pos, quat, scale);
                    mainGroup.position.copy(pos); mainGroup.quaternion.copy(quat); mainGroup.scale.setScalar(data.startScale); 
                }}
            }} else if (grabbingControllers.size === 2) {{
                const dist = controller0.position.distanceTo(controller1.position);
                const data0 = previousTransforms.get(controller0);
                if (data0 && data0.startDist > 0) {{
                    const ratio = dist / data0.startDist;
                    mainGroup.scale.setScalar(data0.startScale * ratio);
                }}
            }}
            
            // Compass / HUD Update
            let azimuth;
            if (renderer.xr.isPresenting) {{
                // VR Mode: Use Camera World Direction
                const camVec = new THREE.Vector3();
                camera.getWorldDirection(camVec);
                azimuth = Math.atan2(camVec.x, camVec.z);
                // Update the VR HUD Texture
                drawCompassOnContext(vrHudCtx, azimuth, 256);
                vrHudTexture.needsUpdate = true;
            }} else {{
                // Desktop Mode: Use OrbitControls angle
                azimuth = controls.getAzimuthalAngle();
                drawCompassOnContext(compassCtx, azimuth, 120);
            }}
            
            renderer.render(scene, camera);
        }});
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>'''

  

    





 
   
    
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def process_gpr_data(job_id, filepath, settings, original_filename):
    """Process GPR data in a separate thread"""
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['message'] = 'Loading CSV file...'
        
        print(f"Processing job {job_id}: {original_filename}")
        
        # Create output directory
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load data with encoding handling
        processing_jobs[job_id]['message'] = 'Detecting file encoding...'
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'utf-16', 'ascii']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Successfully read with {encoding} encoding")
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                print(f"Failed with {encoding}: {e}")
                continue
        
        if df is None:
            # Try with error handling
            try:
                df = pd.read_csv(filepath, encoding_errors='ignore')
                print("Read with encoding errors ignored")
            except Exception as e:
                processing_jobs[job_id]['status'] = 'error'
                processing_jobs[job_id]['message'] = f'Failed to read CSV file: {str(e)}'
                return
        
        processing_jobs[job_id]['message'] = f'Found {len(df):,} rows, processing...'
        
        # Check if columns exist
        if len(df.columns) <= max(settings['col_idx_x'], settings['col_idx_y'], 
                                 settings['col_idx_z'], settings['col_idx_amplitude']):
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['message'] = f'CSV file has only {len(df.columns)} columns, but need column index {max(settings["col_idx_x"], settings["col_idx_y"], settings["col_idx_z"], settings["col_idx_amplitude"])}'
            return
        
        # Extract data
        raw_x = pd.to_numeric(df.iloc[:, settings['col_idx_x']], errors='coerce')
        raw_y = pd.to_numeric(df.iloc[:, settings['col_idx_y']], errors='coerce')
        raw_z = pd.to_numeric(df.iloc[:, settings['col_idx_z']], errors='coerce')
        raw_amp = pd.to_numeric(df.iloc[:, settings['col_idx_amplitude']], errors='coerce')
        
        # Create dataframe
        data = pd.DataFrame({
            'x': raw_x, 'y': raw_y, 'z': raw_z, 'amp': raw_amp
        }).dropna()
        
        if len(data) == 0:
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['message'] = 'No valid numeric data found in specified columns'
            return
        
        # Process coordinates
        if settings['invert_depth']:
            data['z'] = -data['z'].abs()
        
        if settings['center_coordinates']:
            x_c, y_c = data['x'].mean(), data['y'].mean()
            data['x'] -= x_c
            data['y'] -= y_c
        
        # Scale large coordinates
        max_range = max(data['x'].max()-data['x'].min(), data['y'].max()-data['y'].min())
        if max_range > 50:
            sf = 10 / max_range
            data['x'] *= sf
            data['y'] *= sf
            data['z'] *= sf
        
        # Filter by amplitude
        data['abs_amp'] = data['amp'].abs()
        threshold = data['abs_amp'].quantile(settings['threshold_percentile'])
        df_filtered = data[data['abs_amp'] > threshold].copy()
        
        if len(df_filtered) == 0:
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['message'] = 'No points after filtering! Try lowering the percentile.'
            return
        
        amp_min = df_filtered['abs_amp'].min()
        amp_max = df_filtered['abs_amp'].max()
        
        data_bounds = {
            'x_min': float(df_filtered['x'].min()),
            'x_max': float(df_filtered['x'].max()),
            'y_min': float(df_filtered['y'].min()),
            'y_max': float(df_filtered['y'].max()),
            'z_min': float(df_filtered['z'].min()),
            'z_max': float(df_filtered['z'].max())
        }
        
        # Generate surfaces
        surface_info = None
        num_slices = 0
        
        if settings['generate_surface']:
            processing_jobs[job_id]['message'] = 'Generating surfaces...'
            
            try:
                if settings['generate_amplitude_surface']:
                    vertices, faces, colors, surface_info = generate_surface_mesh(
                        df_filtered, 'x', 'y', 'z', 'abs_amp', settings['surface_resolution']
                    )
                    write_obj_mesh(
                        os.path.join(output_dir, 'surface_amplitude.obj'),
                        vertices, faces, colors
                    )
                
                # Generate depth slices
                if settings['surface_depth_slices'] > 0:
                    slices = generate_depth_slices(
                        df_filtered, 'x', 'y', 'z', 'abs_amp', 
                        settings['surface_depth_slices'], resolution=50
                    )
                    num_slices = len(slices)
                    
                    for i, slice_data in enumerate(slices):
                        write_obj_mesh(
                            os.path.join(output_dir, f'slice_{i+1}.obj'),
                            slice_data['vertices'],
                            slice_data['faces'],
                            slice_data['colors']
                        )
                    
            except Exception as e:
                print(f"Surface generation error: {e}")
        
        # Create layers - IMPORTANT: Keep coordinates as they are!
        processing_jobs[job_id]['message'] = 'Creating amplitude layers...'
        try:
            df_filtered['iso_range'] = pd.qcut(df_filtered['abs_amp'], settings['iso_bins'], labels=False, duplicates='drop')
        except:
            # If qcut fails, use manual binning
            df_filtered['iso_range'] = pd.cut(df_filtered['abs_amp'], bins=settings['iso_bins'], labels=False)
        
        actual_bins = df_filtered['iso_range'].nunique()
        
        ply_files = []
        layer_info_html = ""
        amplitude_ranges = []
        total_output_points = 0
        
        for iso_level in range(actual_bins):
            iso_data = df_filtered[df_filtered['iso_range'] == iso_level]
            if len(iso_data) == 0:
                continue
            
            if len(iso_data) > settings['max_points_per_layer']:
                iso_data = iso_data.sample(n=settings['max_points_per_layer'], random_state=42)
            
            # Use the actual coordinates - don't apply depth offset for visualization
            x = iso_data['x'].values
            y = iso_data['y'].values
            z = iso_data['z'].values  # Keep as is - already negative for depth
            
            color = create_iso_colormap(iso_level, actual_bins)
            colors = np.full((len(iso_data), 3), color)
            points = np.column_stack((x, y, z))
            
            iso_min, iso_max = iso_data['abs_amp'].min(), iso_data['abs_amp'].max()
            filename = f"layer_{iso_level+1}.ply"
            filepath_ply = os.path.join(output_dir, filename)
            
            write_ply_fast(filepath_ply, points, colors)
            ply_files.append(filepath_ply)
            amplitude_ranges.append((float(iso_min), float(iso_max)))
            total_output_points += len(iso_data)
            
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
            layer_info_html += f'<div><span class="color-swatch" style="background:{color_hex}"></span>L{iso_level+1}: {iso_min:.0f}-{iso_max:.0f} mV</div>'
        
        # Create viewer
        processing_jobs[job_id]['message'] = 'Creating VR viewer...'
        
        data_info = {
            'original_filename': original_filename,
            'total_points': total_output_points,
            'x_min': data_bounds['x_min'],
            'x_max': data_bounds['x_max'],
            'y_min': data_bounds['y_min'],
            'y_max': data_bounds['y_max'],
            'z_min': data_bounds['z_min'],
            'z_max': data_bounds['z_max'],
            'amp_min': float(amp_min),
            'amp_max': float(amp_max),
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate total files for loading
        total_files = len(ply_files)
        if settings['generate_surface']:
            total_files += 1  # surface_amplitude.obj
        if num_slices > 0:
            total_files += num_slices  # depth slices
        
        create_vr_viewer(
            ply_files, layer_info_html, output_dir, settings, data_info, job_id,
            has_surface=settings['generate_surface'], 
            surface_info=surface_info, 
            num_slices=num_slices,
            total_files=total_files
        )
        
        # Save info
        info_data = {
            'original_filename': original_filename,
            'total_points': total_output_points,
            'layers': actual_bins,
            'has_surface': settings['generate_surface'],
            'num_slices': num_slices,
            'data_bounds': data_bounds,
            'settings': settings
        }
        with open(os.path.join(output_dir, 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2)
        
        processing_jobs[job_id]['status'] = 'completed'
        processing_jobs[job_id]['message'] = 'Processing complete!'
        processing_jobs[job_id]['output_dir'] = job_id
        processing_jobs[job_id]['data_info'] = data_info
        
        print(f"Job {job_id} completed successfully")
        
    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['message'] = f'Error: {str(e)}'

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html', default_settings=DEFAULT_SETTINGS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(filepath)
    
    # Get settings from form
    settings = DEFAULT_SETTINGS.copy()
    for key in settings.keys():
        if key in request.form:
            val = request.form[key]
            if val.lower() in ('true', 'false'):
                settings[key] = val.lower() == 'true'
            elif '.' in val:
                try:
                    settings[key] = float(val)
                except:
                    settings[key] = val
            else:
                try:
                    settings[key] = int(val)
                except:
                    settings[key] = val
    
    # Initialize job tracking
    processing_jobs[job_id] = {
        'status': 'pending',
        'message': 'Waiting to start...',
        'filename': filename,
        'settings': settings
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_gpr_data,
        args=(job_id, filepath, settings, filename)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id, 'filename': filename})

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_jobs[job_id])

@app.route('/view/<job_id>')
def view_result(job_id):
    output_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return "Job not found", 404
    
    index_path = os.path.join(output_dir, 'index.html')
    
    if not os.path.exists(index_path):
        return "Viewer not found", 404
    
    return send_file(index_path)

@app.route('/files/<job_id>/<path:filename>')
def serve_file(job_id, filename):
    """Serve files from the processed folder"""
    output_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return "Job not found", 404
    
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(file_path)

@app.route('/download/<job_id>')
def download_result(job_id):
    output_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
    
    if not os.path.exists(output_dir):
        return "Job not found", 404
    
    # Create zip file
    import zipfile
    zip_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{job_id}.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    
    return send_file(zip_path, as_attachment=True, download_name=f'gpr_vr_{job_id}.zip')

@app.route('/cleanup/<job_id>')
def cleanup_job(job_id):
    if job_id in processing_jobs:
        # Clean up files
        upload_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_*")
        import glob
        for f in glob.glob(upload_file):
            try:
                os.remove(f)
            except:
                pass
        
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except:
                pass
        
        zip_file = os.path.join(app.config['PROCESSED_FOLDER'], f'{job_id}.zip')
        if os.path.exists(zip_file):
            try:
                os.remove(zip_file)
            except:
                pass
        
        del processing_jobs[job_id]
    
    return jsonify({'success': True})

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create HTML template
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPR VR Viewer - Upload</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: #1a1a2e;
            color: white;
            padding: 30px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        header p {
            color: #aaa;
            margin: 10px 0 0;
        }
        .content {
            display: flex;
            min-height: 600px;
        }
        .upload-section {
            flex: 1;
            padding: 40px;
            background: #f8f9fa;
        }
        .settings-section {
            flex: 1;
            padding: 40px;
            background: white;
            border-left: 1px solid #ddd;
            overflow-y: auto;
            max-height: 600px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        input[type="file"] {
            width: 100%;
            padding: 15px;
            border: 2px dashed #667eea;
            border-radius: 5px;
            background: white;
            cursor: pointer;
        }
        input[type="number"], input[type="text"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .checkbox-group input[type="checkbox"] {
            width: auto;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
            width: 100%;
            margin-top: 20px;
        }
        .btn:hover {
            background: #5a67d8;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status-area {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            display: none;
        }
        #progressBar {
            width: 100%;
            height: 20px;
            background: #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        #progressFill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }
        .setting-category {
            background: #f1f3f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .setting-category h3 {
            margin-top: 0;
            color: #495057;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }
        .setting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .setting-label {
            flex: 1;
            font-size: 14px;
        }
        .setting-control {
            flex: 1;
        }
        .small-input {
            width: 80px;
        }
        #fileInfo {
            background: #e7f3ff;
            padding: 10px;
            border-radius = 5px;
            margin-top: 10px;
            font-size: 14px;
        }
        .help-text {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .completed-actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .completed-actions .btn {
            flex: 1;
            margin-top: 0;
        }
        .btn-view {
            background: #10b981;
        }
        .btn-view:hover {
            background: #0da271;
        }
        .btn-download {
            background: #3b82f6;
        }
        .btn-download:hover {
            background: #2563eb;
        }
        .instructions {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-top: 30px;
        }
        .instructions h3 {
            margin-top: 0;
            color: #856404;
        }
        .instructions ul {
            padding-left: 20px;
        }
        .instructions li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
<header>
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; padding: 10px;">
        <img src="/static/logo.jpeg" alt="GPR VR Viewer Logo" style="height: 70px; width: auto; border-radius: 5px;">
        <div style="text-align: left;">
            <h1 style="margin: 0; font-size: 2.2em;">GPR VR Viewer</h1>
            <p style="margin: 5px 0 0; color: #aaa;">Upload GPR CSV data and generate interactive 3D VR visualizations</p>
        </div>
    </div>
</header>
        
        <div class="content">
            <div class="upload-section">
                <h2>1. Upload CSV File</h2>
                <div class="form-group">
                    <label for="fileInput">GPR Data File (.csv)</label>
                    <input type="file" id="fileInput" accept=".csv,.txt">
                    <div id="fileInfo" style="display: none;"></div>
                </div>
                
                <div class="instructions">
                    <h3>File Requirements:</h3>
                    <ul>
                        <li>CSV or text format</li>
                        <li>Default column indices: X(0), Y(1), Z(7), Amplitude(8)</li>
                        <li>Adjust indices in settings if needed</li>
                        <li>Supports large files (up to 1GB)</li>
                        <li>Auto-detects encoding (UTF-8, Latin-1, etc.)</li>
                    </ul>
                </div>
                
                <button class="btn" id="processBtn" onclick="processFile()" disabled>Process File</button>
                
                <div class="status-area" id="statusArea">
                    <h3>Processing Status</h3>
                    <div id="statusMessage">Waiting to start...</div>
                    <div id="progressBar">
                        <div id="progressFill"></div>
                    </div>
                    <div id="progressText">0%</div>
                    
                    <div id="completedActions" class="completed-actions" style="display: none;">
                        <button class="btn btn-view" onclick="viewResult()">View in 3D</button>
                        <button class="btn btn-download" onclick="downloadResult()">Download All Files</button>
                        <button class="btn" onclick="newFile()" style="background: #6c757d;">Process Another</button>
                    </div>
                </div>
            </div>
            
            <div class="settings-section">
                <h2>2. Processing Settings</h2>
                
                <div class="setting-category">
                    <h3>Data Columns (0-based indices)</h3>
                    <div class="setting-row">
                        <div class="setting-label">X Column:</div>
                        <div class="setting-control">
                            <input type="number" id="colIdxX" value="{{ default_settings.col_idx_x }}" min="0" max="100" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Y Column:</div>
                        <div class="setting-control">
                            <input type="number" id="colIdxY" value="{{ default_settings.col_idx_y }}" min="0" max="100" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Z (Depth) Column:</div>
                        <div class="setting-control">
                            <input type="number" id="colIdxZ" value="{{ default_settings.col_idx_z }}" min="0" max="100" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Amplitude Column:</div>
                        <div class="setting-control">
                            <input type="number" id="colIdxAmplitude" value="{{ default_settings.col_idx_amplitude }}" min="0" max="100" class="small-input">
                        </div>
                    </div>
                </div>
                
                <div class="setting-category">
                    <h3>Processing Settings</h3>
                    <div class="setting-row">
                        <div class="setting-label">Amplitude Percentile:</div>
                        <div class="setting-control">
                            <input type="number" id="thresholdPercentile" value="{{ default_settings.threshold_percentile }}" min="0.5" max="1" step="0.01" class="small-input">
                            <div class="help-text">Higher = fewer but stronger points</div>
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Amplitude Layers:</div>
                        <div class="setting-control">
                            <input type="number" id="isoBins" value="{{ default_settings.iso_bins }}" min="1" max="10" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Max Points per Layer:</div>
                        <div class="setting-control">
                            <input type="number" id="maxPointsPerLayer" value="{{ default_settings.max_points_per_layer }}" min="1000" max="1000000" step="1000" class="small-input">
                        </div>
                    </div>
                </div>
                
                <div class="setting-category">
                    <h3>Surface Generation</h3>
                    <div class="checkbox-group">
                        <input type="checkbox" id="generateSurface" checked>
                        <label for="generateSurface">Generate 3D Surfaces</label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Surface Resolution:</div>
                        <div class="setting-control">
                            <input type="number" id="surfaceResolution" value="{{ default_settings.surface_resolution }}" min="10" max="200" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Depth Slices:</div>
                        <div class="setting-control">
                            <input type="number" id="surfaceDepthSlices" value="{{ default_settings.surface_depth_slices }}" min="0" max="20" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Surface Opacity:</div>
                        <div class="setting-control">
                            <input type="number" id="surfaceOpacity" value="{{ default_settings.surface_opacity }}" min="0.1" max="1" step="0.1" class="small-input">
                        </div>
                    </div>
                </div>
                
                <div class="setting-category">
                    <h3>VR Settings</h3>
                    <div class="setting-row">
                        <div class="setting-label">Point Size:</div>
                        <div class="setting-control">
                            <input type="number" id="vrPointSize" value="{{ default_settings.vr_point_size }}" min="0.001" max="0.1" step="0.001" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Depth Offset per Level:</div>
                        <div class="setting-control">
                            <input type="number" id="depthOffsetPerLevel" value="{{ default_settings.depth_offset_per_level }}" min="0" max="0.5" step="0.01" class="small-input">
                        </div>
                    </div>
                </div>
                
                <div class="setting-category">
                    <h3>Coordinate Settings</h3>
                    <div class="checkbox-group">
                        <input type="checkbox" id="invertDepth" checked>
                        <label for="invertDepth">Invert Depth (positive down)</label>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="centerCoordinates" checked>
                        <label for="centerCoordinates">Center Coordinates</label>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let checkInterval = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('fileInfo').innerHTML = `
                    <strong>Selected:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB
                `;
                document.getElementById('processBtn').disabled = false;
            }
        });
        
        function processFile() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert('Please select a file first');
                return;
            }
            
            // Disable button and show status
            document.getElementById('processBtn').disabled = true;
            document.getElementById('statusArea').style.display = 'block';
            document.getElementById('statusMessage').textContent = 'Uploading file...';
            
            // Create FormData
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Add all settings to FormData
            const settings = {
                'col_idx_x': document.getElementById('colIdxX').value,
                'col_idx_y': document.getElementById('colIdxY').value,
                'col_idx_z': document.getElementById('colIdxZ').value,
                'col_idx_amplitude': document.getElementById('colIdxAmplitude').value,
                'threshold_percentile': document.getElementById('thresholdPercentile').value,
                'iso_bins': document.getElementById('isoBins').value,
                'max_points_per_layer': document.getElementById('maxPointsPerLayer').value,
                'generate_surface': document.getElementById('generateSurface').checked,
                'surface_resolution': document.getElementById('surfaceResolution').value,
                'surface_depth_slices': document.getElementById('surfaceDepthSlices').value,
                'surface_opacity': document.getElementById('surfaceOpacity').value,
                'vr_point_size': document.getElementById('vrPointSize').value,
                'depth_offset_per_level': document.getElementById('depthOffsetPerLevel').value,
                'invert_depth': document.getElementById('invertDepth').checked,
                'center_coordinates': document.getElementById('centerCoordinates').checked,
                'generate_amplitude_surface': true
            };
            
            for (const key in settings) {
                formData.append(key, settings[key]);
            }
            
            // Upload file
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    document.getElementById('processBtn').disabled = false;
                    return;
                }
                
                currentJobId = data.job_id;
                document.getElementById('statusMessage').textContent = 'File uploaded, processing...';
                
                // Start checking status
                checkInterval = setInterval(checkStatus, 2000);
            })
            .catch(error => {
                alert('Upload failed: ' + error);
                document.getElementById('processBtn').disabled = false;
            });
        }
        
        function checkStatus() {
            if (!currentJobId) return;
            
            fetch(`/status/${currentJobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        clearInterval(checkInterval);
                        document.getElementById('statusMessage').textContent = 'Error: ' + data.error;
                        document.getElementById('processBtn').disabled = false;
                        return;
                    }
                    
                    document.getElementById('statusMessage').textContent = data.message;
                    
                    // Update progress based on status
                    let progress = 0;
                    if (data.status === 'processing') {
                        progress = 50;
                    } else if (data.status === 'completed') {
                        progress = 100;
                        clearInterval(checkInterval);
                        document.getElementById('progressText').textContent = '100% - Complete!';
                        document.getElementById('completedActions').style.display = 'flex';
                    } else if (data.status === 'error') {
                        progress = 0;
                        clearInterval(checkInterval);
                        document.getElementById('progressText').textContent = 'Error occurred';
                        document.getElementById('processBtn').disabled = false;
                    } else if (data.status === 'pending') {
                        progress = 10;
                    }
                    
                    document.getElementById('progressFill').style.width = progress + '%';
                    document.getElementById('progressText').textContent = progress + '%';
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                });
        }
        
        function viewResult() {
            if (currentJobId) {
                window.open(`/view/${currentJobId}`, '_blank');
            }
        }
        
        function downloadResult() {
            if (currentJobId) {
                window.location.href = `/download/${currentJobId}`;
            }
        }
        
        function newFile() {
            // Reset form
            document.getElementById('fileInput').value = '';
            document.getElementById('fileInfo').style.display = 'none';
            document.getElementById('fileInfo').innerHTML = '';
            document.getElementById('processBtn').disabled = true;
            document.getElementById('statusArea').style.display = 'none';
            document.getElementById('completedActions').style.display = 'none';
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressText').textContent = '0%';
            
            // Clean up old job
            if (currentJobId) {
                fetch(`/cleanup/${currentJobId}`);
                currentJobId = null;
            }
            
            if (checkInterval) {
                clearInterval(checkInterval);
                checkInterval = null;
            }
        }
        
        // Set default values
        window.onload = function() {
            // All inputs are already populated from template
        };
    </script>
</body>
</html>'''
    
    # Write template file
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # Run Flask app
    print("\n" + "="*60)
    print("GPR VR Viewer Web Interface")
    print("="*60)
    print("\nStarting server...")
    print("Open http://localhost:5006 in your browser")
    print("\nFeatures:")
    print("  - Upload CSV files via web interface")
    print("  - Auto-detects file encoding")
    print("  - Adjust processing parameters")
    print("  - Real-time processing status")
    print("  - View 3D VR visualization in browser")
    print("  - Download processed files")
    print("  - PROPER COORDINATE SYSTEM: Data positioned at actual coordinates")
    print("  - X-axis: East/West coordinates")
    print("  - Y-axis: Depth (positive down)")
    print("  - Z-axis: North/South coordinates")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5006)