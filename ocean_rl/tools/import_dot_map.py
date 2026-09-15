"""Convert a regular black-dot obstacle diagram into an exact-palette PNG.

This calibrated importer is for diagrams with a complete dot border. It is not
an arbitrary photograph segmenter. Pixel endpoint annotations are supplied by
the user/operator; map labels and obstacle classifications come from the image.
"""
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
from ocean_rl.environments.racetrack import RacetrackEnv


def lattice(profile, offset):
    active = np.flatnonzero(profile >= .75 * profile.max())
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    centers = np.array([part.mean() + offset for part in groups if len(part)])
    if len(centers) < 3:
        raise ValueError('Cannot identify a repeated dot border in the selected band.')
    indices = np.arange(len(centers))
    spacing, origin = np.polyfit(indices, centers, 1)
    fitted = origin + spacing * indices
    if spacing < 3 or np.max(np.abs(fitted - centers)) > .2 * spacing:
        raise ValueError('Detected border dots are not regularly spaced. Adjust grid box/band.')
    return fitted


def shortest_cardinal_path(grid, start, finish):
    parents = {start: None}
    pending = deque([start])
    while pending:
        cell = pending.popleft()
        if cell == finish:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parents[cell]
            return path[::-1]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = cell[0] + dr, cell[1] + dc
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != 0 and nxt not in parents):
                parents[nxt] = cell
                pending.append(nxt)
    raise ValueError('No cardinal path between the annotated endpoints.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-stem', type=Path, required=True)
    parser.add_argument('--grid-box', nargs=4, type=int, required=True, metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    parser.add_argument('--start-pixel', nargs=2, type=float, required=True)
    parser.add_argument('--finish-pixel', nargs=2, type=float, required=True)
    parser.add_argument('--border-band', type=int, default=9)
    parser.add_argument('--dark-threshold', type=int, default=60)
    args = parser.parse_args()
    with Image.open(args.source) as source:
        pixels = np.asarray(source.convert('RGB'))
    left, top, right, bottom = args.grid_box
    if not (0 <= left < right <= pixels.shape[1] and 0 <= top < bottom <= pixels.shape[0]):
        parser.error('grid-box must be inside the image')
    if not 1 <= args.border_band < min(right-left, bottom-top) or not 0 <= args.dark_threshold <= 255:
        parser.error('Invalid border-band or dark-threshold')
    dark = pixels.max(axis=2) < args.dark_threshold
    xs = lattice(dark[top:top+args.border_band, left:right].sum(axis=0), left)
    ys = lattice(dark[top:bottom, left:left+args.border_band].sum(axis=1), top)
    radius = max(1, int(round(min(np.diff(xs).mean(), np.diff(ys).mean()) * .2)))
    coverage = np.zeros((len(ys), len(xs)))
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            ix, iy = int(round(x)), int(round(y))
            coverage[r, c] = dark[max(0,iy-radius):iy+radius+1, max(0,ix-radius):ix+radius+1].mean()
    grid = np.where(coverage >= .55, 0, 1).astype(np.int32)
    endpoints = []
    for label, point in [(2,args.start_pixel),(3,args.finish_pixel)]:
        x, y = point
        if not left <= x < right or not top <= y < bottom:
            parser.error('Endpoint pixel must lie inside the grid box')
        cell = int(np.argmin(abs(ys-y))), int(np.argmin(abs(xs-x)))
        if cell in endpoints:
            parser.error('Start and finish resolve to the same cell')
        # Endpoint labels cover image pixels. Explicit annotation takes precedence.
        grid[cell] = label
        endpoints.append(cell)
    env = RacetrackEnv.from_numpy(grid, movement_mode='bidirectional', max_velocity=2)
    path = shortest_cardinal_path(grid, *endpoints)
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    palette = np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0]], dtype=np.uint8)
    image_path = stem.with_suffix('.png')
    Image.fromarray(palette[grid]).save(image_path)
    config = {'map': image_path.name, 'name': stem.name,
              'environment': {'movement_mode':'bidirectional','max_velocity':2,'fail_prob':.1,'max_episode_steps':5000}}
    stem.with_suffix('.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    # Prove the actual image importer reconstructs exactly the derived grid.
    imported = RacetrackEnv.from_file(stem.with_suffix('.json'))
    np.testing.assert_array_equal(imported.grid, grid)
    report = {'source':str(args.source),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'grid_box':args.grid_box,'border_band':args.border_band,'dark_threshold':args.dark_threshold,
              'start_pixel':args.start_pixel,'finish_pixel':args.finish_pixel,
              'shape':list(grid.shape),'start_cell':endpoints[0],'finish_cell':endpoints[1],
              'obstacle_count':int((grid==0).sum()),'x_centers':xs.tolist(),'y_centers':ys.tolist(),
              'occupancy_threshold':.55,'patch_radius':radius,
              'geometric_cardinal_path':path,'geometric_path_steps':len(path)-1,
              'note':'Geometric BFS connectivity check, not a learned policy. Coordinates zero-based row/column.'}
    stem.with_name(stem.name+'_extraction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].imshow(pixels)
        for label, point, color in [('Start',args.start_pixel,'red'),('Finish',args.finish_pixel,'lime')]:
            ax[0].scatter(*point,s=120,facecolors='none',edgecolors=color,label=label)
        ax[0].set_title('Source image: selected S and E labels')
        ax[0].legend()
        colors = ['#131722','#e2e8f0','#ef4444','#22c55e']
        ax[1].imshow(grid,cmap=ListedColormap(colors),vmin=0,vmax=3,interpolation='nearest')
        ax[1].set_title(f'Extracted {grid.shape[0]} x {grid.shape[1]} grid')
        ax[1].set(xlabel='Column (zero-based)',ylabel='Row (zero-based)')
        ax[1].legend(handles=[Patch(color=color,label=label) for color,label in zip(colors,['Obstacle','Free','Start','Finish'])],loc='upper right')
        fig.tight_layout()
        fig.savefig(stem.with_name(stem.name+'_preview.png'),dpi=160)
        plt.close(fig)
    print(json.dumps({k:report[k] for k in ('shape','start_cell','finish_cell','obstacle_count','geometric_path_steps')},indent=2))
    print('Image import round-trip matches extracted grid exactly.')


if __name__ == '__main__':
    main()
