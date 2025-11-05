# app.py
import io, base64, torch, numpy as np, colorsys
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

def tensor_to_2d(t):
    t = t.detach().cpu().float()
    while t.dim() > 2:
        t = t[0] if t.shape[0] == 1 else t.mean(0)
    return t.numpy()

def normalize_to_uint8(arr):
    a = np.nan_to_num(arr)
    mn, mx = a.min(), a.max()
    if np.isclose(mx, mn):
        return np.full_like(a, 128, dtype=np.uint8)
    return ((a - mn) / (mx - mn) * 255).round().astype(np.uint8)

def array_to_base64_png(arr):
    img = Image.fromarray(normalize_to_uint8(arr), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def hsv_to_rgb(h, s=1.0, v=1.0):
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v))

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['model']
    state = torch.load(io.BytesIO(file.read()), map_location='cpu')

    data, overall, ind = {}, {'num_layers':0,'num_params':0}, {
        'possible_overfit':False,'high_zeros':False,'over_divergence':False,'notes':[]}

    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v = v.float()
            overall['num_layers'] += 1
            overall['num_params'] += v.numel()
            flat = v.flatten().cpu().numpy()
            flat_list = [float(x) for x in flat[:10000]]

            # Scalar stats
            mean_val = v.mean().item()
            std_val  = v.std().item()
            min_val  = v.min().item()
            max_val  = v.max().item()
            zero_pct = ((v == 0).sum() / v.numel() * 100).item()

            # Histogram
            hist, bins = np.histogram(flat, bins=50)
            hist_list = [int(x) for x in hist]
            bins_list = [float(x) for x in bins]

            # 2D Heatmap: use original 2D if possible
            arr2d = v.squeeze()
            if arr2d.dim() >= 2:
                if arr2d.dim() > 2:
                    arr2d = arr2d.mean(dim=0)
                heatmap_img = array_to_base64_png(arr2d.cpu().numpy())
            else:
                heatmap_img = array_to_base64_png(tensor_to_2d(v))

            stats = {
                'shape': list(v.shape),
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'zero_percentage': zero_pct,
                'flat_1d': flat_list,
                '2d_image': heatmap_img,
                'hist': hist_list,
                'bins': bins_list,
            }

            # --- 3D Point Cloud: Smart Layout ---
            N = min(10000, flat.size)
            idx = np.linspace(0, flat.size-1, N, dtype=int)
            vals = flat[idx]
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                norm = (vals - vmin) / (vmax - vmin)
            else:
                norm = np.full_like(vals, 0.5)

            # Size: 1.0 → 2.0
            sizes = 1.0 + norm

            # Layout: 3D → 2D → 1D
            orig_shape = v.shape
            positions = []
            colors = []

            if v.dim() >= 3 and all(s >= 2 for s in orig_shape[-3:]):
                d, h, w = orig_shape[-3:]
                step_d = max(1, d // 20)
                step_h = max(1, h // 50)
                step_w = max(1, w // 50)
                grid = v[::step_d, ::step_h, ::step_w].cpu().numpy().flatten()

                # Normalize values safely (NumPy 2.0 compatible)
                if np.ptp(grid) > 0:
                    grid_norm = (grid - grid.min()) / np.ptp(grid)
                else:
                    grid_norm = np.full_like(grid, 0.5)

                # Generate positions, colors, and sizes
                for i in range(len(grid)):
                    z = (i // ((h // step_h) * (w // step_w))) / (d // step_d)
                    y = ((i // (w // step_w)) % (h // step_h)) / (h // step_h)
                    x = (i % (w // step_w)) / (w // step_w)
                    positions += [x, y, z]

                    # COLOR: Red (low) → Blue (high)
                    hue = grid_norm[i] * 0.6667  # 0.0 = red, 0.666 = blue
                    r, g, b = hsv_to_rgb(hue)
                    colors += [r / 255, g / 255, b / 255]

                # SIZE: 1.0 (low) → 2.0 (high)
                sizes = 1.0 + grid_norm

            elif v.dim() >= 2:
                h, w = orig_shape[-2:]
                step_h = max(1, h // 100)
                step_w = max(1, w // 100)
                grid = v[::step_h, ::step_w].cpu().numpy().flatten()

                if np.ptp(grid) > 0:
                    grid_norm = (grid - grid.min()) / np.ptp(grid)
                else:
                    grid_norm = np.full_like(grid, 0.5)

                for i in range(len(grid)):
                    y = (i // (w // step_w)) / (h // step_h)
                    x = (i % (w // step_w)) / (w // step_w)
                    positions += [x, y, 0.5]

                    # COLOR: Red → Blue
                    hue = grid_norm[i] * 0.6667
                    r, g, b = hsv_to_rgb(hue)
                    colors += [r / 255, g / 255, b / 255]

                # SIZE: 1.0 → 2.0
                sizes = 1.0 + grid_norm

            else:
                side = int(N ** (1/3)) + 1
                for i in range(N):
                    x = (i % side) / side
                    y = ((i // side) % side) / side
                    z = (i // (side * side)) / side
                    positions += [x, y, z]

                    # COLOR: Red → Blue
                    hue = norm[i] * 0.6667
                    r, g, b = hsv_to_rgb(hue)
                    colors += [r / 255, g / 255, b / 255]

                # SIZE: 1.0 → 2.0 (already computed earlier)
                # sizes = 1.0 + norm

            stats['3d_positions'] = positions
            stats['3d_colors'] = colors
            stats['3d_sizes'] = [float(s) for s in sizes]

            data[k] = stats

            # Indicators
            if std_val < 1e-4: ind['possible_overfit']=True; ind['notes'].append(f'Low variance {k}')
            if zero_pct > 50: ind['high_zeros']=True; ind['notes'].append(f'High zeros {k}')
            if abs(max_val) > 1e6: ind['over_divergence']=True; ind['notes'].append(f'Extreme value {k}')
        else:
            pass

    overall['hyperparams'] = state.get('hyperparams','Not found')
    return jsonify({'data':data,'overall':overall,'indicators':ind})

if __name__=='__main__':
    app.run(debug=True)
