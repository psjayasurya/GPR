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
import colorsys


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
    'font_size_multiplier': 1.0,
    'font_family': 'Arial',
    
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
    'max_points_per_layer': 500000,
    
    # Visualization
    'color_palette': 'Viridis'
}

# --- EXTENSIVE COLOR PALETTES ---
COLOR_PALETTES = {
    # Scientific palettes (colorblind-friendly)
    'Viridis': [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], 
                [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43]],
    
    'Plasma': [[13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150], 
               [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33]],
    
    'Inferno': [[0, 0, 4], [25, 11, 68], [66, 10, 104], [106, 23, 110], 
                [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 119, 44], [252, 166, 50]],
    
    'Magma': [[0, 0, 4], [26, 16, 70], [66, 10, 104], [106, 23, 110], 
              [147, 38, 103], [188, 55, 84], [226, 83, 78], [251, 135, 97], [252, 197, 131]],
    
    'Cividis': [[0, 32, 76], [0, 54, 93], [0, 75, 100], [14, 94, 95], 
                [57, 112, 87], [98, 129, 81], [140, 145, 80], [182, 160, 85], [234, 176, 100]],
    
    # Traditional seismic/geophysics palettes
    'Seismic': [[0, 0, 255], [127, 127, 255], [255, 255, 255], [255, 127, 127], [255, 0, 0]],
    
    'Rainbow': [[148, 0, 211], [75, 0, 130], [0, 0, 255], [0, 255, 0], 
                [255, 255, 0], [255, 127, 0], [255, 0, 0]],
    
    'Standard': [[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255]],
    
    # Diverging palettes
    'RdBu': [[103, 0, 31], [178, 24, 43], [214, 96, 77], [244, 165, 130], 
             [253, 219, 199], [247, 247, 247], [209, 229, 240], [146, 197, 222], 
             [67, 147, 195], [33, 102, 172], [5, 48, 97]],
    
    'Spectral': [[158, 1, 66], [213, 62, 79], [244, 109, 67], [253, 174, 97], 
                 [254, 224, 139], [255, 255, 191], [230, 245, 152], [171, 221, 164], 
                 [102, 194, 165], [50, 136, 189], [94, 79, 162]],
    
    # Sequential palettes
    'Blues': [[247, 251, 255], [222, 235, 247], [198, 219, 239], [158, 202, 225], 
              [107, 174, 214], [66, 146, 198], [33, 113, 181], [8, 81, 156], [8, 48, 107]],
    
    'Greens': [[247, 252, 245], [229, 245, 224], [199, 233, 192], [161, 217, 155], 
               [116, 196, 118], [65, 171, 93], [35, 139, 69], [0, 109, 44], [0, 68, 27]],
    
    'Oranges': [[255, 245, 235], [254, 230, 206], [253, 208, 162], [253, 174, 107], 
                [253, 141, 60], [241, 105, 19], [217, 72, 1], [166, 54, 3], [127, 39, 4]],
    
    # Alternative palettes
    'Turbo': [[35, 23, 27], [69, 61, 120], [97, 113, 178], [125, 170, 211], 
              [158, 222, 217], [199, 251, 194], [239, 251, 143], [255, 217, 95], 
              [255, 165, 62], [255, 109, 58], [230, 57, 43]],
    
    'Thermal': [[0, 0, 0], [64, 0, 0], [128, 0, 0], [192, 64, 0], 
                [255, 128, 0], [255, 192, 64], [255, 255, 128], [255, 255, 255]],
    
    'Ocean': [[0, 0, 128], [0, 0, 255], [0, 128, 255], [0, 255, 255], 
              [128, 255, 255], [255, 255, 255]],
    
    'Grayscale': [[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]],
    
    # Custom geological palettes
    'Geology': [[139, 69, 19], [160, 82, 45], [205, 133, 63], [222, 184, 135], 
                [245, 222, 179], [210, 180, 140], [165, 42, 42], [178, 34, 34]],
    
    'Earth': [[0, 56, 101], [0, 92, 135], [0, 128, 149], [0, 164, 143], 
              [85, 188, 124], [170, 212, 105], [255, 235, 86], [255, 189, 46]],
    
    # High contrast
    'HighContrast': [[230, 57, 70], [241, 135, 1], [255, 200, 87], 
                     [168, 218, 220], [69, 123, 157], [29, 53, 87]],
}

def interpolate_color(palette, value):
    """Interpolate color from palette for a value between 0 and 1"""
    if len(palette) == 1:
        return palette[0]
    
    val_scaled = value * (len(palette) - 1)
    idx = int(val_scaled)
    
    if idx >= len(palette) - 1:
        return palette[-1]
    
    t = val_scaled - idx
    c1 = np.array(palette[idx])
    c2 = np.array(palette[idx + 1])
    
    # Smooth interpolation
    return (c1 * (1 - t) + c2 * t).astype(int).tolist()

def get_color_from_palette(value, palette_name='Viridis'):
    """
    Get RGB color for a normalized value (0-1) or index from a palette.
    Returns [r, g, b] as 0-255 integers.
    """
    palette = COLOR_PALETTES.get(palette_name, COLOR_PALETTES['Viridis'])
    
    if isinstance(value, (int, np.integer)):
        # For discrete iso levels
        idx = value % len(palette)
        return palette[idx]
    else:
        # For continuous values 0-1
        return interpolate_color(palette, float(value))

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

def generate_surface_mesh(df, x_col, y_col, z_col, amp_col, resolution=100, palette_name='Viridis'):
    """Generate a surface mesh from GPR data"""
    print(f"  Generating surface mesh with {palette_name} palette...")
    
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
            
            # Get color from selected palette
            amp_val = amp_norm[i, j]
            c_int = get_color_from_palette(amp_val, palette_name)
            vertex_colors.append([c_int[0]/255.0, c_int[1]/255.0, c_int[2]/255.0])
    
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

def generate_depth_slices(df, x_col, y_col, z_col, amp_col, num_slices=5, resolution=50, palette_name='Viridis'):
    """Generate horizontal slice surfaces at different depths"""
    print(f"  Generating {num_slices} depth slices with {palette_name} palette...")
    
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
                c_int = get_color_from_palette(amp_val, palette_name)
                vertex_colors.append([c_int[0]/255.0, c_int[1]/255.0, c_int[2]/255.0])
        
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

def create_iso_colormap(iso_level, total_levels, palette_name='Viridis'):
    """Wrapper for backward compatibility or direct use"""
    return get_color_from_palette(iso_level, palette_name)

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
                    
                    // FIXED: Assign directly to index to match the checkbox ID
                    layers[{i}] = points; 
                    
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

def create_vr_viewer(ply_files, layer_info, legend_info, output_dir, settings, data_info, job_id,
                     has_surface=False, surface_info=None, num_slices=0, total_files=0, pipe_file=None):
    
    # ... (Keep existing loader generation code at the top) ...
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
    
    # Slice loading code (Keep existing)
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
    
    pipe_loader_js = ""
    if pipe_file:
        pipe_loader_js = f'''
        const pipeGroup = new THREE.Group();
        pipeGroup.visible = false; // Hidden by default
        mainGroup.add(pipeGroup);
        
        // Pipe Loader
        const pipeLoader = new PLYLoader();
        pipeLoader.load('/files/{job_id}/{pipe_file}', (geometry) => {{
            geometry.computeVertexNormals();
            const material = new THREE.MeshStandardMaterial({{ 
                color: 0xaaaaaa, metalness: 0.5, roughness: 0.5,
                side: THREE.DoubleSide
            }});
            const mesh = new THREE.Mesh(geometry, material);
            
            // Apply offset to align with centered GPR data
            const offsetX = {data_info.get('offset_x', 0)};
            const offsetY = {data_info.get('offset_y', 0)};
            mesh.position.set(-offsetX, -offsetY, 0);
            
            // Apply Data Scale Factor (if GPR was scaled down)
            const sf = {data_info.get('scale_factor', 1.0)};
            mesh.scale.setScalar(sf);
            
            // Rotate 90 degrees to make it horizontal
            mesh.rotation.x = Math.PI / 2;
            
            pipeGroup.add(mesh);
            console.log('Loaded Pipe with offset', -offsetX, -offsetY, 'scale', sf, 'rotation 90 deg');
            
        }}, undefined, (err) => {{ console.error('Pipe load error', err); }});
        '''
    
    x_len = data_info['x_max'] - data_info['x_min']
    y_len = data_info['y_max'] - data_info['y_min']
    z_len = abs(data_info['z_max'] - data_info['z_min'])
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
        
        #info {{
            position: absolute; top: 20px; left: 20px;
            background: rgba(0,0,0,0.9); padding: 20px;
            border-radius: 10px; z-index: 100;
            max-width: 400px; max-height: 90vh; overflow-y: auto;
        }}
        
        #compass-ui {{
            position: absolute; top: 80px; right: 20px;
            width: 120px; height: 120px; z-index: 100;
            pointer-events: none;
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
        
        /* NEW STYLES FOR CHECKLIST */
        #layer-list {{ 
            max-height: 150px; 
            overflow-y: auto; 
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
            padding: 5px;
            margin-top: 5px; 
        }}
        .layer-item {{
            display: flex;
            align-items: center;
            padding: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 11px;
        }}
        .layer-item:hover {{ background: rgba(255,255,255,0.1); }}
        .layer-item input {{ margin-right: 8px; }}
        .layer-label {{ cursor: pointer; flex: 1; display: flex; align-items: center; }}
        .color-swatch {{
            display: inline-block; width: 12px; height: 12px;
            border-radius: 2px; margin-right: 8px; border: 1px solid #555;
        }}
        .btn-small {{
            background: #444; color: white; border: 1px solid #666;
            padding: 2px 8px; font-size: 10px; cursor: pointer; border-radius: 3px;
            margin-right: 5px;
        }}
        .btn-small:hover {{ background: #555; }}
        
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
        .control-group-title {{ font-weight: bold; margin-bottom: 8px; color: #4CAF50; font-size: 12px; display: flex; justify-content: space-between; align-items: center; }}
        .file-info {{ font-size: 10px; color: #aaa; margin-top: 5px; border-top: 1px solid #333; padding-top: 5px; }}
        
        /* Palette info */
        .palette-info {{
            background: rgba(255,255,255,0.05);
            padding: 5px;
            border-radius: 3px;
            margin-top: 5px;
            font-size: 10px;
            color: #aaa;
        }}
        
        /* Legend Styles */
        #legend {{
            position: absolute; bottom: 20px; right: 20px;
            background: rgba(0,0,0,0.8); padding: 15px;
            border-radius: 8px; z-index: 100;
            font-size: 12px;
            min-width: 150px;
        }}
        .legend-title {{ 
            font-weight: bold; margin-bottom: 8px; 
            border-bottom: 1px solid #555; padding-bottom: 5px;
            color: #ddd; text-align: center;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .legend-color {{ width: 14px; height: 14px; margin-right: 8px; border: 1px solid #555; border-radius: 2px; }}
        .legend-label {{ color: #ccc; }}
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
            <strong>Color Palette:</strong> {settings['color_palette']}<br>
        </div>
        
        <div class="control-group">
            <div class="control-group-title">
                POINT CLOUD
                <div>
                    <button class="btn-small" onclick="toggleAllLayers(true)">All</button>
                    <button class="btn-small" onclick="toggleAllLayers(false)">None</button>
                </div>
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showPoints" checked>
                <label for="showPoints">Show GPR Model</label>
            </div>
            
            <div class="slider-container">
                <div class="slider-label"><span>Point Size:</span><span id="sizeValue">{settings['vr_point_size']}</span></div>
                <input type="range" id="sizeSlider" min="0.002" max="0.05" step="0.002" value="{settings['vr_point_size']}">
            </div>
            
            <!-- REPLACED SLIDER WITH CHECKLIST -->
            <div id="layer-list">
                {layer_info}
            </div>
        </div>
        
        <div class="control-group">
            <div class="control-group-title">SURFACES</div>
            <div class="toggle-container">
                <input type="checkbox" id="showSurface">
                <label for="showSurface">Show Amplitude Surface</label>
            </div>
            <div class="slider-container">
                <div class="slider-label"><span>Surface Opacity:</span><span id="surfaceOpacityValue">{settings['surface_opacity']}</span></div>
                <input type="range" id="surfaceOpacitySlider" min="0.1" max="1" step="0.1" value="{settings['surface_opacity']}">
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showSlices">
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
                <div class="slider-label"><span>Scale:</span><span id="scaleValue">0.1</span></div>
                <input type="range" id="scaleSlider" min="0.1" max="5" step="0.1" value="0.1">
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showGround">
                <label for="showGround">Show Ground Image</label>
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showAxes" checked>
                <label for="showAxes">Show Data Axes</label>
            </div>
            <div class="toggle-container">
                <input type="checkbox" id="showAxes" checked>
                <label for="showAxes">Show Data Axes</label>
            </div>
            <div class="toggle-container">
                <button onclick="resetPosition()">Reset Position to Floor</button>
            </div>
            <div class="toggle-container" id="pipeToggleContainer" style="display: {'block' if pipe_file else 'none'}; margin-top:10px; border-top:1px solid #555; padding-top:10px;">
                <input type="checkbox" id="show3DPipe">
                <label for="show3DPipe">Show 3D Pipe Model</label>
            </div>
            <div class="slider-container" id="pipeScaleContainer" style="display: {'block' if pipe_file else 'none'};">
                <div class="slider-label"><span>Pipe Scale:</span><span id="pipeScaleValue">1.0</span></div>
                <input type="range" id="pipeScaleSlider" min="0.1" max="10" step="0.1" value="1.0">
                
                <div class="slider-label"><span>Pipe Pos X:</span><span id="pipePosXValue">0</span></div>
                <input type="range" id="pipePosXSlider" min="-50" max="50" step="0.5" value="0">
                
                <div class="slider-label"><span>Pipe Pos Y:</span><span id="pipePosYSliderValue">0</span></div>
                <input type="range" id="pipePosYSlider" min="-50" max="50" step="0.5" value="0">
                
                <div class="slider-label"><span>Pipe Pos Z:</span><span id="pipePosZSliderValue">0</span></div>
                <input type="range" id="pipePosZSlider" min="-20" max="20" step="0.5" value="0">
                
                <div class="slider-label"><span>Pipe Rot Z:</span><span id="pipeRotZSliderValue">0</span></div>
                <input type="range" id="pipeRotZSlider" min="0" max="360" step="1" value="0">
            </div>
        </div>
        
        <div class="control-group" style="font-size: 10px;">
            <div class="control-group-title">VR INSTRUCTIONS</div>
            <strong>GRIP BUTTON:</strong> Grab and rotate the model (6-DoF).<br>
            <strong>TWO HANDS:</strong> Pull apart to zoom/scale.<br>
            <strong>LEFT TRIGGER:</strong> Cycle Layers (All -> Single -> None) | <strong>RIGHT TRIGGER:</strong> Toggle Pipe Model.
        </div>
        
        <div class="file-info">
            Processed on: {data_info['processing_date']}<br>
            Color Palette: {settings['color_palette']}<br>
            <a href="/" style="color: #4CAF50;">← Process another file</a>
        </div>
    </div>
    
    <div id="debug">Ready</div>
    
    <div id="legend">
        <div class="legend-title">Amplitude (Signal Strength)</div>
        {legend_info}
    </div>
    
    <button id="vr-button" disabled>Loading...</button>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ XRControllerModelFactory }} from 'three/addons/webxr/XRControllerModelFactory.js';
        import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';
        import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';

        // ... (Keep existing Compass drawing function) ...
        function drawCompassOnContext(ctx, angle, size) {{
            const cx = size/2, cy = size/2, r = size * 0.42;
            ctx.clearRect(0, 0, size, size);
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(angle);
            ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fillStyle = 'rgba(60, 60, 60, 0.9)'; ctx.fill();
            ctx.lineWidth = size * 0.04; ctx.strokeStyle = '#888'; ctx.stroke();
            ctx.beginPath(); ctx.arc(0, 0, r * 0.9, 0, Math.PI * 2); ctx.fillStyle = '#222'; ctx.fill();
            const needleLen = r * 0.75; const needleWide = r * 0.15;
            ctx.beginPath(); ctx.moveTo(0, -needleLen); ctx.lineTo(needleWide, 0); ctx.lineTo(-needleWide, 0); ctx.closePath(); ctx.fillStyle = '#FF3333'; ctx.fill();
            ctx.beginPath(); ctx.moveTo(0, needleLen); ctx.lineTo(needleWide, 0); ctx.lineTo(-needleWide, 0); ctx.closePath(); ctx.fillStyle = '#EEEEEE'; ctx.fill();
            ctx.save(); ctx.translate(0, -r + (size * 0.1)); ctx.rotate(0);
            ctx.fillStyle = 'white'; ctx.font = 'bold ' + (size * 0.2) + 'px Arial';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.shadowColor="black"; ctx.shadowBlur=4; ctx.fillText('N', 0, -5);
            ctx.restore();
            ctx.beginPath(); ctx.arc(0, 0, size * 0.05, 0, Math.PI * 2); ctx.fillStyle = '#GOLD'; ctx.fill();
            ctx.restore();
            ctx.save(); ctx.translate(cx, cy); ctx.strokeStyle = "rgba(255,255,255,0.3)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, -r); ctx.lineTo(0, -r-5); ctx.stroke(); ctx.restore();
        }}

        const compassCanvas = document.getElementById('compassCanvas');
        const compassCtx = compassCanvas.getContext('2d');
        drawCompassOnContext(compassCtx, 0, 120);

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
        
        // --- GROUND ---
        const texLoader = new THREE.TextureLoader();
        const groundGroup = new THREE.Group();
        scene.add(groundGroup);

        texLoader.load(
            '/static/ground.jpg',
            function (texture) {{
                // compute a ground size from data extents (use larger of x/y and add margin)
                const groundSize = {round(max(x_len, y_len) * 1.5, 3)};
                const geometry = new THREE.PlaneGeometry(groundSize, groundSize);
                const material = new THREE.MeshBasicMaterial({{ map: texture, side: THREE.DoubleSide, color: 0xffffff }});
                const plane = new THREE.Mesh(geometry, material);
                plane.rotation.x = -Math.PI / 2;
                plane.position.y = -0.05;
                groundGroup.add(plane);
                groundGroup.visible = false; // Hidden by default
            }},
            undefined,
            function (err) {{
                const grid = new THREE.GridHelper(50, 50, 0x666666, 0x444444);
                groundGroup.add(grid);
            }}
        );
        
        // --- DATA CONTAINER ---
        const mainGroup = new THREE.Group();
        scene.add(mainGroup);
        mainGroup.position.set(0, 0.05, 0); 
        mainGroup.rotation.x = -Math.PI / 2;
        mainGroup.scale.setScalar(0.1); // Initial Zoom 0.1
        
        const initialTransform = {{
            position: mainGroup.position.clone(),
            rotation: mainGroup.rotation.clone(),
            scale: mainGroup.scale.clone()
        }};
        
        window.resetPosition = function() {{
            mainGroup.position.copy(initialTransform.position);
            mainGroup.rotation.copy(initialTransform.rotation);
            mainGroup.scale.copy(initialTransform.scale);
            document.getElementById('scaleSlider').value = 0.1;
            document.getElementById('scaleValue').textContent = "0.1";
            controls.reset();
        }};
        
        const pointCloudGroup = new THREE.Group();
        mainGroup.add(pointCloudGroup);
        const surfaceGroup = new THREE.Group();
        surfaceGroup.visible = false;  // Hidden by default
        mainGroup.add(surfaceGroup);
        const sliceGroup = new THREE.Group();
        sliceGroup.visible = false;    // Hidden by default
        mainGroup.add(sliceGroup);
        const axesGroup = new THREE.Group();
        mainGroup.add(axesGroup);
        
        // Settings
        // Initialize layers array - This needs to match the indices in the Python loop
        // Settings
        const layers = []; // Initialize as empty, loaders will fill specific indices 
        const surfaces = [];
        const slices = [];
        let pointSize = {settings['vr_point_size']};
        let surfaceOpacity = {settings['surface_opacity']};
        let sliceOpacity = 0.5;

        // ... (Keep existing Axis Generator) ...
        function createTextSprite(text, color, size=40) {{
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            // Increased canvas size to prevent Clipping
            canvas.width = 512; canvas.height = 256;
            ctx.fillStyle = color; 
            
            // User settings
            const fontSize = size * {settings.get('font_size_multiplier', 1.0)};
            const fontFamily = "{settings.get('font_family', 'Arial')}";
            
            // Scale font size generally for the larger canvas
            ctx.font = "Bold " + (fontSize * 2) + "px " + fontFamily; 
            ctx.textAlign = "center"; 
            ctx.textBaseline = "middle"; 
            ctx.fillText(text, 256, 128);
            
            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({{ map: texture, depthTest: false }});
            const sprite = new THREE.Sprite(material);
            // Scale remains similar to world space size from before
            sprite.scale.set(1, 0.5, 1);
            return sprite;
        }}

        function buildDataAxes() {{
            const xLen = {x_len}, yLen = {y_len}, zLen = {z_len};
            // Data Coordinates (inside mainGroup): X=East/West, Y=North/South, Z=Depth
            const xMin = -xLen / 2, yMin = -yLen / 2, zMin = -zLen / 2;
            const xMax = xLen / 2, yMax = yLen / 2, zMax = zLen / 2;

            // --- 1. RESTORE ORIGINAL AXES (Red/Green/Blue Lines) ---
            // X Axis (Red)
            const xMat = new THREE.LineBasicMaterial({{ color: 0xff0000, linewidth: 2 }});
            const xPoints = [new THREE.Vector3(xMin, yMin, zMin), new THREE.Vector3(xMax, yMin, zMin)];
            axesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(xPoints), xMat));
            
            // Y Axis (Green)
            const yMat = new THREE.LineBasicMaterial({{ color: 0x00ff00, linewidth: 2 }});
            const yPoints = [new THREE.Vector3(xMin, yMin, zMin), new THREE.Vector3(xMin, yMax, zMin)];
            axesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(yPoints), yMat));
            
            // Z Axis (Blue)
            const zMat = new THREE.LineBasicMaterial({{ color: 0x0088ff, linewidth: 2 }});
            const zPoints = [new THREE.Vector3(xMin, yMin, zMin), new THREE.Vector3(xMin, yMin, zMax)];
            axesGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(zPoints), zMat));

            // --- 2. RESTORE DIMENSION LABELS ---
            const xLabel = createTextSprite("X: " + xLen.toFixed(1) + "m", "#ff5555", 35); 
            xLabel.position.set(0, yMin - 0.2, zMin); 
            axesGroup.add(xLabel);
            
            const yLabel = createTextSprite("Y: " + yLen.toFixed(1) + "m", "#55ff55", 35); 
            yLabel.position.set(xMin - 0.3, 0, zMin); 
            axesGroup.add(yLabel);

            const zLabel = createTextSprite("Z: " + zLen.toFixed(1) + "m", "#5555ff", 35); 
            zLabel.position.set(xMin, yMin - 0.2, 0); 
            axesGroup.add(zLabel);

            // --- 3. SURVEY BOX (Wireframe) ---
            const boxGeo = new THREE.BoxGeometry(xLen, yLen, zLen);
            const boxEdges = new THREE.EdgesGeometry(boxGeo);
            const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({{ color: 0xffff00, opacity: 0.3, transparent: true }}));
            axesGroup.add(boxLine);
            
            // --- 4. NEW DIRECTION LABELS ---
            // X Axis Directions
            const lblWest = createTextSprite("← West", "#cccccc", 30);
            lblWest.position.set(xMin - 0.5, 0, 0); 
            axesGroup.add(lblWest);
            
            const lblEast = createTextSprite("East →", "#cccccc", 30);
            lblEast.position.set(xMax + 0.5, 0, 0); 
            axesGroup.add(lblEast);
            
            // Y Axis Directions (North/South)
            const lblNorth = createTextSprite("↑ North", "#cccccc", 30);
            lblNorth.position.set(0, yMax + 0.5, 0); 
            axesGroup.add(lblNorth);
            
            const lblSouth = createTextSprite("↓ South", "#cccccc", 30);
            lblSouth.position.set(0, yMin - 0.5, 0); 
            axesGroup.add(lblSouth);
            
            // Z Axis Directions (Depth)
            // zMax is Surface (approx 0), zMin is Deep (approx -5)
            const lblSurf = createTextSprite("Surface", "#ccffcc", 30);
            lblSurf.position.set(0, 0, zMax + 0.2);
            axesGroup.add(lblSurf);
            
            const lblDepth = createTextSprite("Depth", "#ccffcc", 30);
            lblDepth.position.set(0, 0, zMin - 0.2);
            axesGroup.add(lblDepth);
        }}
        buildDataAxes();

        // VR HUD
        const vrHudCanvas = document.createElement('canvas');
        vrHudCanvas.width = 256; vrHudCanvas.height = 256;
        const vrHudCtx = vrHudCanvas.getContext('2d');
        const vrHudTexture = new THREE.CanvasTexture(vrHudCanvas);
        const vrHudMaterial = new THREE.SpriteMaterial({{ map: vrHudTexture, depthTest: false, depthWrite: false }});
        const vrHudSprite = new THREE.Sprite(vrHudMaterial);
        vrHudSprite.position.set(0.15, 0.15, -0.5); vrHudSprite.scale.set(0.15, 0.15, 1);

        // Loaders
        const plyLoader = new PLYLoader();
        const objLoader = new OBJLoader();
        let loadedCount = 0;
        const totalFiles = {total_files};
        const layerPromises = [];
        
        {layer_loaders_js}
        
        // MODIFIED LAYER LOADING LOGIC
        // We need to make sure 'layers' array is populated correctly at specific indices
        // The original loader pushed to array, here we ensure index assignment in the generated JS
        // (Note: The Python generate_layer_loaders function should be fine, it adds to pointCloudGroup)
        // But for the checklist to work, we need to map the loaded object to the layers array
        
        // Overwriting the array push in generated code logic:
        // The generated code does: layers.push(points); 
        // We will just let it push, but we know the order matches the file order.
        
        {surface_loader_js}
        {slice_loader_js}
        {pipe_loader_js}
        
        Promise.all(layerPromises).then(() => {{
            document.getElementById('loading').style.display = 'none';
            debug('Ready');
        }});

        // --- NEW CHECKLIST LOGIC ---
        window.toggleLayer = function(index, isChecked) {{
            if (layers[index]) {{
                layers[index].visible = isChecked;
            }} else {{
                console.warn("Layer " + index + " not ready yet");
            }}
        }};

        window.toggleAllLayers = function(visible) {{
            for(let i=0; i < {len(ply_files)}; i++) {{
                if (layers[i]) {{
                    layers[i].visible = visible;
                }}
                
                const cb = document.getElementById('layer_cb_' + i);
                if(cb) cb.checked = visible;
            }}
            
            // Update Toggle Master Checkbox State
            const master = document.getElementById('showPoints');
            if(master) master.checked = visible;
        }};

        // Slider Listeners
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
        
        // Pipe Scale Slider
        const pipeScaleSlider = document.getElementById('pipeScaleSlider');
        if (pipeScaleSlider) {{
            pipeScaleSlider.addEventListener('input', (e) => {{
                const val = parseFloat(e.target.value);
                document.getElementById('pipeScaleValue').textContent = val.toFixed(1);
                if (typeof pipeGroup !== 'undefined') {{
                    pipeGroup.scale.setScalar(val);
                }}
            }});
        }}
        
        // Pipe Position X
        const pipePosXSlider = document.getElementById('pipePosXSlider');
        if (pipePosXSlider) {{
            pipePosXSlider.addEventListener('input', (e) => {{
                const val = parseFloat(e.target.value);
                document.getElementById('pipePosXValue').textContent = val.toFixed(1);
                if (typeof pipeGroup !== 'undefined') pipeGroup.position.x = val;
            }});
        }}
        
        // Pipe Position Y
        const pipePosYSlider = document.getElementById('pipePosYSlider');
        if (pipePosYSlider) {{
            pipePosYSlider.addEventListener('input', (e) => {{
                const val = parseFloat(e.target.value);
                document.getElementById('pipePosYSliderValue').textContent = val.toFixed(1);
                if (typeof pipeGroup !== 'undefined') pipeGroup.position.y = val;
            }});
        }}
        
        // Pipe Position Z
        const pipePosZSlider = document.getElementById('pipePosZSlider');
        if (pipePosZSlider) {{
            pipePosZSlider.addEventListener('input', (e) => {{
                const val = parseFloat(e.target.value);
                document.getElementById('pipePosZSliderValue').textContent = val.toFixed(1);
                if (typeof pipeGroup !== 'undefined') pipeGroup.position.z = val;
            }});
        }}
        
        // Pipe Rotation Z (Heading in this coord system)
        const pipeRotZSlider = document.getElementById('pipeRotZSlider');
        if (pipeRotZSlider) {{
            pipeRotZSlider.addEventListener('input', (e) => {{
                const val = parseFloat(e.target.value);
                document.getElementById('pipeRotZSliderValue').textContent = val.toFixed(0);
                if (typeof pipeGroup !== 'undefined') pipeGroup.rotation.z = val * Math.PI / 180;
            }});
        }}
        
        // Pipe Visibility Checkbox
        const pipeCheckbox = document.getElementById('show3DPipe');
        if (pipeCheckbox) {{
            pipeCheckbox.addEventListener('change', (e) => {{
                if (typeof pipeGroup !== 'undefined') {{
                    pipeGroup.visible = e.target.checked;
                }}
            }});
        }}

        // VR Setup
        const controllerModelFactory = new XRControllerModelFactory();
        const cameraRig = new THREE.Group();
        scene.add(cameraRig);
        const controller0 = renderer.xr.getController(0);
        const controller1 = renderer.xr.getController(1);
        cameraRig.add(controller0); cameraRig.add(controller1);
        [controller0, controller1].forEach(c => {{
            const grp = renderer.xr.getControllerGrip(c === controller0 ? 0 : 1);
            grp.add(controllerModelFactory.createControllerModel(grp));
            cameraRig.add(grp);
            c.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,-1)]), new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.5 }})));
        }});

        // 6-DoF Interaction
        const grabbingControllers = new Set();
        const previousTransforms = new Map();
        
        function onSqueezeStart(event) {{
            const controller = event.target;
            grabbingControllers.add(controller);
            const controllerInv = controller.matrixWorld.clone().invert();
            const relativeMatrix = new THREE.Matrix4().multiplyMatrices(controllerInv, mainGroup.matrixWorld);
            previousTransforms.set(controller, {{
                relativeMatrix: relativeMatrix,
                startDist: grabbingControllers.size === 2 ? controller0.position.distanceTo(controller1.position) : 0,
                startScale: mainGroup.scale.x
            }});
        }}
        
        function onSqueezeEnd(event) {{
            const releasedController = event.target;
            grabbingControllers.delete(releasedController);
            previousTransforms.delete(releasedController);
            if (grabbingControllers.size === 1) {{
                const remainingController = grabbingControllers.values().next().value;
                const controllerInv = remainingController.matrixWorld.clone().invert();
                const relativeMatrix = new THREE.Matrix4().multiplyMatrices(controllerInv, mainGroup.matrixWorld);
                previousTransforms.set(remainingController, {{
                    relativeMatrix: relativeMatrix, startDist: 0, startScale: mainGroup.scale.x 
                }});
            }}
        }}
        
        controller0.addEventListener('squeezestart', onSqueezeStart);
        controller0.addEventListener('squeezeend', onSqueezeEnd);
        controller1.addEventListener('squeezestart', onSqueezeStart);
        controller1.addEventListener('squeezeend', onSqueezeEnd);
        
        // CHANGED VR INTERACTION: Cycle Layers (Left Trigger)
        let currentLayerIndex = -1; // -1: All, 0..N: Single, N+1: None
        
        controller0.addEventListener('selectstart', () => {{
            const layerCount = layers.length; // Max index + 1
            if (layerCount === 0) return;
            
            currentLayerIndex++;
            if (currentLayerIndex > layerCount) currentLayerIndex = -1;
            
            const masterCb = document.getElementById('showPoints');
            
            if (currentLayerIndex === -1) {{
                // SHOW ALL
                pointCloudGroup.visible = true;
                if(masterCb) masterCb.checked = true;
                toggleAllLayers(true);
                debug("Show All Layers");
            }} else if (currentLayerIndex === layerCount) {{
                // SHOW NONE
                pointCloudGroup.visible = false;
                if(masterCb) masterCb.checked = false;
                debug("Hide All Layers");
            }} else {{
                // SHOW SINGLE
                pointCloudGroup.visible = true;
                if(masterCb) masterCb.checked = true;
                
                // Set only currentLayerIndex visible
                layers.forEach((l, idx) => {{
                    const isTarget = (idx === currentLayerIndex);
                    if(l) l.visible = isTarget;
                    const lcb = document.getElementById('layer_cb_' + idx);
                    if(lcb) lcb.checked = isTarget;
                }});
                debug("Show Layer " + (currentLayerIndex + 1));
            }}
        }});
        
        // CHANGED VR INTERACTION: Toggle Pipe Model (Right Trigger)
        controller1.addEventListener('selectstart', () => {{
            const cb = document.getElementById('show3DPipe');
            if (cb) {{
                cb.checked = !cb.checked;
                cb.dispatchEvent(new Event('change'));
            }}
        }});
        
    

        // VR Button
        const vrButton = document.getElementById('vr-button');
        if ('xr' in navigator) {{
            navigator.xr.isSessionSupported('immersive-vr').then(ok => {{
                if (ok) {{
                    vrButton.disabled = false; vrButton.textContent = 'Enter VR';
                    vrButton.onclick = async () => {{
                        const session = await navigator.xr.requestSession('immersive-vr', {{ optionalFeatures: ['local-floor', 'bounded-floor'] }});
                        renderer.xr.setSession(session);
                        camera.add(vrHudSprite);
                        vrButton.textContent = 'VR Active'; vrButton.disabled = true;
                        session.addEventListener('end', () => {{ vrButton.textContent = 'Enter VR'; vrButton.disabled = false; camera.remove(vrHudSprite); }});
                    }};
                }} else vrButton.textContent = 'VR Not Supported';
            }});
        }} else vrButton.textContent = 'WebXR Not Available';

        // Render Loop
        renderer.setAnimationLoop(() => {{
            controls.update();
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
            
            let azimuth;
            if (renderer.xr.isPresenting) {{
                const camVec = new THREE.Vector3(); camera.getWorldDirection(camVec);
                azimuth = Math.atan2(camVec.x, camVec.z);
                drawCompassOnContext(vrHudCtx, azimuth, 256); vrHudTexture.needsUpdate = true;
            }} else {{
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
        print(f"Using color palette: {settings['color_palette']}")
        
        # Create output directory
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], job_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Copy pipe file if present
        if settings.get('pipe_filename'):
            src = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{settings['pipe_filename']}")
            dst = os.path.join(output_dir, settings['pipe_filename'])
            if os.path.exists(src):
                shutil.copy(src, dst)

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
        
        x_c, y_c = 0.0, 0.0
        if settings['center_coordinates']:
            x_c, y_c = data['x'].mean(), data['y'].mean()
            data['x'] -= x_c
            data['y'] -= y_c
        
        # Scale large coordinates
        max_range = max(data['x'].max()-data['x'].min(), data['y'].max()-data['y'].min())
        sf = 1.0
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
                palette_name = settings.get('color_palette', 'Viridis')
                if settings['generate_amplitude_surface']:
                    vertices, faces, colors, surface_info = generate_surface_mesh(
                        df_filtered, 'x', 'y', 'z', 'abs_amp', settings['surface_resolution'],
                        palette_name=palette_name
                    )
                    write_obj_mesh(
                        os.path.join(output_dir, 'surface_amplitude.obj'),
                        vertices, faces, colors
                    )
                
                # Generate depth slices
                if settings['surface_depth_slices'] > 0:
                    slices = generate_depth_slices(
                        df_filtered, 'x', 'y', 'z', 'abs_amp', 
                        settings['surface_depth_slices'], resolution=50,
                        palette_name=palette_name
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
        legend_html = ""
        amplitude_ranges = []
        total_output_points = 0
        
        palette_name = settings.get('color_palette', 'Viridis')
        palette_colors = COLOR_PALETTES.get(palette_name, COLOR_PALETTES['Viridis'])
        
        # Build CSS gradient string
        gradient_colors = []
        for col in palette_colors:
            gradient_colors.append(f"rgb({col[0]},{col[1]},{col[2]})")
        gradient_str = f"linear-gradient(to right, {', '.join(gradient_colors)})"
        
        legend_html = f'''
        <div style="margin-bottom:10px;">
            <div style="height:15px; width:100%; background:{gradient_str}; border-radius:3px; border:1px solid #555;"></div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#ccc; margin-top:2px;">
                <span>{amp_min:.0f}</span>
                <span>{amp_max:.0f}</span>
            </div>
        </div>
        '''
        
        for iso_level in range(actual_bins):
            iso_data = df_filtered[df_filtered['iso_range'] == iso_level]
            if len(iso_data) == 0:
                continue
            
            if len(iso_data) > settings['max_points_per_layer']:
                iso_data = iso_data.sample(n=settings['max_points_per_layer'], random_state=42)
            
            # Use the actual coordinates
            x = iso_data['x'].values
            y = iso_data['y'].values
            z = iso_data['z'].values
            
            color = get_color_from_palette(iso_level, palette_name)
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
            
            # Create Checkbox HTML
            layer_info_html += f'''
            <div class="layer-item">
                <input type="checkbox" id="layer_cb_{iso_level}" checked onchange="toggleLayer({iso_level}, this.checked)">
                <label for="layer_cb_{iso_level}" class="layer-label">
                    <span class="color-swatch" style="background:{color_hex}"></span>
                    L{iso_level+1}: {iso_min:.0f}-{iso_max:.0f}
                </label>
            </div>'''
            
            # Create Legend HTML
            legend_html += f'''
            <div class="legend-item">
                <span class="legend-color" style="background:{color_hex}"></span>
                <span class="legend-label">{iso_min:.0f} - {iso_max:.0f}</span>
            </div>'''
        
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
            'offset_x': float(x_c),
            'offset_y': float(y_c),
            'scale_factor': float(sf),
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate total files for loading
        total_files = len(ply_files)
        if settings['generate_surface']:
            total_files += 1  # surface_amplitude.obj
        if num_slices > 0:
            total_files += num_slices  # depth slices
        if settings.get('pipe_filename'):
            total_files += 1 # pipe file
        
        create_vr_viewer(
            ply_files, layer_info_html, legend_html, output_dir, settings, data_info, job_id,
            has_surface=settings['generate_surface'], 
            surface_info=surface_info, 
            num_slices=num_slices,
            total_files=total_files,
            pipe_file=settings.get('pipe_filename')
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
    return render_template('index.html', default_settings=DEFAULT_SETTINGS, color_palettes=list(COLOR_PALETTES.keys()))

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
    
    # Save Pipe File if present
    pipe_filename = None
    if 'pipe_file' in request.files:
        pfile = request.files['pipe_file']
        if pfile.filename != '':
            pipe_filename = secure_filename(pfile.filename)
            pfile.save(os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{pipe_filename}"))
    
    
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
    
    if pipe_filename:
        settings['pipe_filename'] = pipe_filename
    
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
    
    # Create HTML template with enhanced color palette selection
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
            border-radius: 5px;
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
        
        /* Color palette preview */
        .palette-preview {
            display: flex;
            height: 30px;
            margin-top: 5px;
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid #ddd;
        }
        .palette-color {
            flex: 1;
            height: 100%;
        }
        .palette-info {
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }
        
        /* Palette grid */
        .palette-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 10px;
        }
        .palette-option {
            border: 2px solid #ddd;
            border-radius: 5px;
            padding: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .palette-option:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .palette-option.selected {
            border-color: #4CAF50;
            background: #f0fff0;
        }
        .palette-name {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .palette-mini-preview {
            display: flex;
            height: 15px;
            border-radius: 2px;
            overflow: hidden;
        }
        .palette-mini-color {
            flex: 1;
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
                
                <div class="form-group" style="border-top:1px dashed #ccc; padding-top:10px;">
                    <label for="pipeInput">Optional: 3D Pipe Model (.ply)</label>
                    <input type="file" id="pipeInput" accept=".ply">
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
                    <h3>Color Palette Selection</h3>
                    <div class="setting-row">
                        <div class="setting-label">Selected Palette:</div>
                        <div class="setting-control">
                            <select id="colorPaletteSelect">
                                {% for palette in color_palettes %}
                                <option value="{{ palette }}" {% if palette == default_settings.color_palette %}selected{% endif %}>{{ palette }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    
                    <div id="palettePreviewContainer">
                        <div class="palette-preview" id="palettePreview"></div>
                        <div class="palette-info" id="paletteInfo">Click on palette below for preview</div>
                    </div>
                    
                    <div class="palette-grid" id="paletteGrid">
                        {% for palette in color_palettes %}
                        <div class="palette-option {% if palette == default_settings.color_palette %}selected{% endif %}" 
                             data-palette="{{ palette }}"
                             onclick="selectPalette('{{ palette }}')">
                            <div class="palette-name">{{ palette }}</div>
                            <div class="palette-mini-preview" id="miniPreview{{ palette }}"></div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="help-text">
                        <strong>Recommended:</strong><br>
                        • <strong>Viridis/Plasma/Inferno/Magma:</strong> Scientific, colorblind-friendly<br>
                        • <strong>Seismic:</strong> Traditional geophysics (blue-white-red)<br>
                        • <strong>RdBu/Spectral:</strong> Diverging data (high contrast)<br>
                        • <strong>Rainbow:</strong> Classic but not colorblind-friendly<br>
                        • <strong>Grayscale:</strong> For printing or monochrome displays
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
                        <div class="setting-control">
                            <input type="number" id="depthOffsetPerLevel" value="{{ default_settings.depth_offset_per_level }}" min="0" max="0.5" step="0.01" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Font Size Multiplier:</div>
                        <div class="setting-control">
                            <input type="number" id="fontSizeMultiplier" value="{{ default_settings.font_size_multiplier }}" min="0.5" max="3" step="0.1" class="small-input">
                        </div>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">Font Family:</div>
                        <div class="setting-control">
                            <select id="fontFamily">
                                <option value="Arial" {% if default_settings.font_family == 'Arial' %}selected{% endif %}>Arial</option>
                                <option value="Verdana" {% if default_settings.font_family == 'Verdana' %}selected{% endif %}>Verdana</option>
                                <option value="Helvetica" {% if default_settings.font_family == 'Helvetica' %}selected{% endif %}>Helvetica</option>
                                <option value="Times New Roman" {% if default_settings.font_family == 'Times New Roman' %}selected{% endif %}>Times New Roman</option>
                                <option value="Courier New" {% if default_settings.font_family == 'Courier New' %}selected{% endif %}>Courier New</option>
                                <option value="Georgia" {% if default_settings.font_family == 'Georgia' %}selected{% endif %}>Georgia</option>
                            </select>
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
        // Define color palettes (simplified versions)
        const COLOR_PALETTES = {
            'Viridis': ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
            'Plasma': ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
            'Inferno': ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d03c', '#fcffa4'],
            'Magma': ['#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f', '#cd4071', '#f1605d', '#fd9668', '#feca8d', '#fcfdbf'],
            'Cividis': ['#00204d', '#00336f', '#2f4880', '#575d8e', '#77729c', '#9589a9', '#b1a2b6', '#cbbcc2', '#e3d7cf', '#fcf5d6'],
            'Seismic': ['#0000ff', '#4040ff', '#8080ff', '#c0c0ff', '#ffffff', '#ffc0c0', '#ff8080', '#ff4040', '#ff0000'],
            'Rainbow': ['#9400d3', '#4b0082', '#0000ff', '#00ff00', '#ffff00', '#ff7f00', '#ff0000'],
            'Standard': ['#ff0000', '#ffa500', '#ffff00', '#00ff00', '#0000ff'],
            'RdBu': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
            'Spectral': ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'],
            'Blues': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            'Greens': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
            'Oranges': ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704'],
            'Turbo': ['#23171b', '#453a79', '#6575b2', '#84b2d6', '#a6edd8', '#ccf7b0', '#f5f18c', '#ffc35e', '#ff913e', '#ff5b3a', '#e62e46'],
            'Thermal': ['#000000', '#400000', '#800000', '#c04000', '#ff8000', '#ffc040', '#ffff80', '#ffffff'],
            'Ocean': ['#000080', '#0000ff', '#0080ff', '#00ffff', '#80ffff', '#ffffff'],
            'Grayscale': ['#000000', '#404040', '#808080', '#c0c0c0', '#ffffff'],
            'Geology': ['#8b4513', '#a0522d', '#cd853f', '#deb887', '#f5deb3', '#d2b48c', '#a52a2a', '#b22222'],
            'Earth': ['#003865', '#005c87', '#008095', '#00a48f', '#55bc7c', '#aad469', '#ffeb56', '#ffbd2e'],
            'HighContrast': ['#e63946', '#f18701', '#ffc857', '#a8dadc', '#457b9d', '#1d3557']
        };
        
        let currentJobId = null;
        let checkInterval = null;
        let selectedPalette = "{{ default_settings.color_palette }}";
        
        // Initialize palette previews
        function initPalettePreviews() {
            // Update main preview
            updatePalettePreview(selectedPalette);
            
            // Create mini previews for all palettes
            for (const paletteName in COLOR_PALETTES) {
                const miniPreview = document.getElementById('miniPreview' + paletteName);
                if (miniPreview) {
                    const colors = COLOR_PALETTES[paletteName];
                    miniPreview.innerHTML = '';
                    colors.forEach(color => {
                        const div = document.createElement('div');
                        div.className = 'palette-mini-color';
                        div.style.backgroundColor = color;
                        miniPreview.appendChild(div);
                    });
                }
            }
            
            // Set selected palette in dropdown
            document.getElementById('colorPaletteSelect').value = selectedPalette;
        }
        
        function updatePalettePreview(paletteName) {
            const palettePreview = document.getElementById('palettePreview');
            const paletteInfo = document.getElementById('paletteInfo');
            const colors = COLOR_PALETTES[paletteName] || COLOR_PALETTES['Viridis'];
            
            palettePreview.innerHTML = '';
            colors.forEach(color => {
                const div = document.createElement('div');
                div.className = 'palette-color';
                div.style.backgroundColor = color;
                palettePreview.appendChild(div);
            });
            
            paletteInfo.textContent = `${paletteName} palette (${colors.length} colors)`;
            
            // Update selection state
            document.querySelectorAll('.palette-option').forEach(option => {
                option.classList.remove('selected');
                if (option.dataset.palette === paletteName) {
                    option.classList.add('selected');
                }
            });
            
            // Update dropdown
            document.getElementById('colorPaletteSelect').value = paletteName;
            
            selectedPalette = paletteName;
        }
        
        function selectPalette(paletteName) {
            updatePalettePreview(paletteName);
        }
        
        document.getElementById('colorPaletteSelect').addEventListener('change', function(e) {
            selectPalette(e.target.value);
        });
        
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
            
            const pipeInput = document.getElementById('pipeInput');
            if(pipeInput.files.length > 0) {
                formData.append('pipe_file', pipeInput.files[0]);
            }
            
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
                'generate_amplitude_surface': true,
                'color_palette': selectedPalette,
                'font_size_multiplier': document.getElementById('fontSizeMultiplier').value,
                'font_family': document.getElementById('fontFamily').value
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
        
        // Initialize on load
        window.onload = function() {
            initPalettePreviews();
        };
    </script>
</body>
</html>'''
    
    #Write template file
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    

    app.run(debug=True, host='0.0.0.0', port=5006)
