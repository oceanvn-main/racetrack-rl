"""Validated map interchange. Rows increase downward; columns increase rightward."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import json
import numpy as np

ASCII_PALETTE = {'#': 0, '.': 0, ' ': 1, 'F': 1, '1': 1,
                 'S': 2, 's': 2, '2': 2, 'G': 3, 'g': 3, 'E': 3, '3': 3}
IMAGE_PALETTE = {(0, 0, 0): 0, (255, 255, 255): 1,
                 (255, 0, 0): 2, (0, 255, 0): 3}

@dataclass(frozen=True, eq=False)
class MapSpec:
    grid: np.ndarray
    name: str = 'custom'
    metadata: object = None

    def __post_init__(self):
        grid = np.asarray(self.grid)
        if grid.ndim != 2 or not all(grid.shape):
            raise ValueError('Map must be a nonempty two-dimensional grid.')
        if grid.dtype.kind not in 'iu' or not np.isin(grid, [0, 1, 2, 3]).all():
            raise ValueError('Map labels must be integers: 0 wall, 1 floor, 2 start, 3 finish.')
        if not np.any(grid == 2) or not np.any(grid == 3):
            raise ValueError('Map requires explicit start (2/S) and finish (3/G) cells.')
        # A bytes-backed view cannot have writeability re-enabled by a caller.
        immutable = np.frombuffer(grid.astype(np.int32).tobytes(), dtype=np.int32).reshape(grid.shape)
        object.__setattr__(self, 'grid', immutable)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata or {})))

    @property
    def start_cells(self):
        return np.argwhere(self.grid == 2)

    @property
    def finish_cells(self):
        return np.argwhere(self.grid == 3)

    def validate_reachability(self, movement_mode):
        """Reverse cardinal flood fill: speed-one paths are realizable with acceleration.

        At a cardinal turn one acceleration simultaneously brakes the old component
        and accelerates the next, so zero velocity off the start is unnecessary.
        """
        if movement_mode not in ('forward-only', 'bidirectional'):
            raise ValueError("movement_mode must be 'forward-only' or 'bidirectional'.")
        predecessors = [(1, 0), (0, -1)]
        if movement_mode == 'bidirectional':
            predecessors += [(-1, 0), (0, 1)]
        seen = {tuple(cell) for cell in self.finish_cells}
        pending = list(seen)
        h, w = self.grid.shape
        while pending:
            r, c = pending.pop()
            for dr, dc in predecessors:
                cell = (r + dr, c + dc)
                rr, cc = cell
                if 0 <= rr < h and 0 <= cc < w and self.grid[cell] and cell not in seen:
                    seen.add(cell)
                    pending.append(cell)
        missing = [tuple(map(int, cell)) for cell in self.start_cells if tuple(cell) not in seen]
        if missing:
            raise ValueError(f'Start cells cannot reach a finish under {movement_mode} movement: {missing}')


def from_ascii(source, char_map=None, *, name=None, metadata=None):
    if isinstance(source, (str, Path)):
        path = Path(source)
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        name = name or path.stem
    else:
        lines = list(source)
    if not lines or not all(isinstance(line, str) for line in lines):
        raise ValueError('ASCII map requires a nonempty sequence of rows.')
    if not lines[0] or any(len(line) != len(lines[0]) for line in lines):
        raise ValueError('ASCII map rows must have the same nonzero width; whitespace is significant.')
    palette = ASCII_PALETTE if char_map is None else char_map
    unknown = [(r, c, char) for r, line in enumerate(lines) for c, char in enumerate(line) if char not in palette]
    if unknown:
        raise ValueError(f'Unknown ASCII map symbols (row, col, symbol): {unknown[:10]}')
    return MapSpec(np.array([[palette[char] for char in line] for line in lines]), name or 'custom', metadata)


def from_image(source, *, palette=None, start_cells=None, finish_cells=None,
               wall_thresh=50, name=None, metadata=None):
    """One image pixel is one grid cell. No rescaling or inferred start/finish.

    Default exact palette: black wall, white floor, red start, green finish.
    Supplying both annotation lists explicitly enables grayscale threshold mode.
    """
    from PIL import Image
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            pixels = np.asarray(image.convert('RGB'))
        name = name or Path(source).stem
    else:
        pixels = np.asarray(source.convert('RGB'))
    annotations = start_cells is not None or finish_cells is not None
    if annotations:
        if start_cells is None or finish_cells is None or palette is not None:
            raise ValueError('Threshold mode requires both start_cells and finish_cells and no palette.')
        if not 0 <= wall_thresh <= 255:
            raise ValueError('wall_thresh must be in [0, 255].')
        grid = (pixels.mean(axis=2) >= wall_thresh).astype(np.int32)
        occupied = set()
        for label, cells in [(2, start_cells), (3, finish_cells)]:
            for cell in cells:
                coords = np.asarray(cell)
                if coords.shape != (2,) or coords.dtype.kind not in 'iu':
                    raise ValueError('Annotations must be integer (row, col) coordinates.')
                r, c = map(int, coords)
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or not grid[r, c] or (r, c) in occupied:
                    raise ValueError(f'Invalid, wall, or overlapping annotation: {(r, c)}')
                grid[r, c] = label
                occupied.add((r, c))
    else:
        palette = IMAGE_PALETTE if palette is None else palette
        grid = np.full(pixels.shape[:2], -1, dtype=np.int32)
        for color, label in palette.items():
            if not isinstance(label, (int, np.integer)) or label not in (0, 1, 2, 3):
                raise ValueError('Image palette labels must be integers 0 through 3.')
            grid[np.all(pixels == color, axis=2)] = label
        if np.any(grid < 0):
            raise ValueError('Image contains colors outside the exact palette; use a labeled image or explicit threshold annotations.')
    return MapSpec(grid, name or 'custom', metadata)


def load_map(path):
    """Load ASCII, .npy, labeled image, Tiled JSON, or JSON map configuration."""
    path = Path(path)
    if path.suffix.lower() == '.tmj':
        from ocean_rl.maps.tiled import from_tiled
        return from_tiled(path)
    if path.suffix.lower() == '.json':
        config = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(config, dict) and config.get('type') == 'map':
            from ocean_rl.maps.tiled import from_tiled
            return from_tiled(path, config)
        if not isinstance(config, dict) or 'map' not in config:
            raise ValueError('Map JSON requires a map file path.')
        if not isinstance(config['map'], str) or not config['map'].strip():
            raise ValueError('Map JSON map must be a nonempty file path string.')
        if 'name' in config and not isinstance(config['name'], str):
            raise ValueError('Map JSON name must be a string.')
        if 'environment' in config and not isinstance(config['environment'], dict):
            raise ValueError('Map JSON environment must be an object.')
        unknown = set(config) - {'map', 'name', 'environment'}
        if unknown:
            raise ValueError(f'Unknown map configuration keys: {sorted(unknown)}')
        source = path.parent / config['map']
        if source.suffix.lower() == '.json':
            raise ValueError('Nested JSON map references are not supported.')
        spec = load_map(source)
        return MapSpec(spec.grid, config.get('name', spec.name), config.get('environment', {}))
    if path.suffix.lower() == '.npy':
        return MapSpec(np.load(path, allow_pickle=False), path.stem)
    if path.suffix.lower() in ('.png', '.bmp', '.jpg', '.jpeg'):
        return from_image(path)
    return from_ascii(path)
