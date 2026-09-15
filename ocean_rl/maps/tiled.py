"""Tiled JSON adapter for a finite orthogonal RL tile layer.

The layer named Racetrack is authoritative; other layers are decorative. Each
used tile must declare integer custom property cell_type (0 wall, 1 floor,
2 start, 3 finish). Erased cells (GID 0) are walls. Geometry is never inferred
from tile artwork. Supports inline tilesets and external JSON .tsj tilesets.
"""
import json
from pathlib import Path
import numpy as np


def properties(items):
    if not isinstance(items, list):
        raise ValueError('Tiled properties must be an array')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('name'), str) or 'value' not in item:
            raise ValueError('Malformed Tiled property')
        if item['name'] in result:
            raise ValueError('Duplicate Tiled property: ' + item['name'])
        result[item['name']] = item['value']
    return result


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def from_tiled(path, document=None):
    from ocean_rl.maps.io import MapSpec
    path = Path(path)
    doc = document if document is not None else json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(doc, dict) or doc.get('type') != 'map':
        raise ValueError('Expected a Tiled JSON map (.tmj or .json)')
    if doc.get('orientation') != 'orthogonal' or doc.get('infinite', False):
        raise ValueError('Use a finite orthogonal Tiled map')
    width = integer(doc.get('width'), 'Map width', 1)
    height = integer(doc.get('height'), 'Map height', 1)
    layers = doc.get('layers', [])
    if not isinstance(layers, list):
        raise ValueError('Tiled layers must be an array')
    selected = [layer for layer in layers if isinstance(layer, dict) and layer.get('name') == 'Racetrack']
    if len(selected) != 1 or selected[0].get('type') != 'tilelayer':
        raise ValueError('Create exactly one top-level tile layer named Racetrack')
    layer = selected[0]
    if layer.get('width') != width or layer.get('height') != height:
        raise ValueError('Racetrack layer dimensions must match the map')
    if any(layer.get(key, 0) != 0 for key in ('x','y','offsetx','offsety')) or 'chunks' in layer:
        raise ValueError('Racetrack layer must have zero offsets and no chunks')
    if layer.get('encoding', 'csv') != 'csv' or layer.get('compression'):
        raise ValueError('Save tile layer data as CSV / JSON array, without compression')
    data = layer.get('data')
    if not isinstance(data, list) or len(data) != width * height:
        raise ValueError('Racetrack data must be an integer array with width * height entries')
    gids = []
    for value in data:
        integer(value, 'Global tile ID')
        if value > 0xffffffff:
            raise ValueError('Global tile ID exceeds 32 bits')
        # Flipping/rotation changes artwork only, not semantic cell type.
        gids.append(value & 0x0fffffff)
    labels = {0: 0}
    tilesets = doc.get('tilesets', [])
    if not isinstance(tilesets, list):
        raise ValueError('Tiled tilesets must be an array')
    ranges = []
    for reference in tilesets:
        if not isinstance(reference, dict):
            raise ValueError('Malformed tileset reference')
        first = integer(reference.get('firstgid'), 'firstgid', 1)
        tileset = reference
        if 'source' in reference:
            if not isinstance(reference['source'], str):
                raise ValueError('Tileset source must be a path string')
            source = path.parent / reference['source']
            if source.suffix.lower() not in ('.tsj', '.json'):
                raise ValueError('Use an external JSON tileset (.tsj), not XML .tsx')
            tileset = json.loads(source.read_text(encoding='utf-8-sig'))
        if not isinstance(tileset, dict):
            raise ValueError('Tileset must be a JSON object')
        count = integer(tileset.get('tilecount'), 'tilecount', 1)
        if any(first < end and first + count > begin for begin, end in ranges):
            raise ValueError('Overlapping tileset GID ranges')
        ranges.append((first, first + count))
        entries = tileset.get('tiles', [])
        if not isinstance(entries, list):
            raise ValueError('Tiles must be an array')
        seen = set()
        for tile in entries:
            if not isinstance(tile, dict):
                raise ValueError('Malformed tile definition')
            local = integer(tile.get('id'), 'Tile id')
            if local >= count or local in seen:
                raise ValueError('Tile ID outside tileset or duplicated')
            seen.add(local)
            value = properties(tile.get('properties', [])).get('cell_type')
            if value is not None:
                integer(value, 'cell_type')
                if value > 3:
                    raise ValueError('cell_type must be 0 wall, 1 floor, 2 start or 3 finish')
                labels[first + local] = value
    unknown = set(gids) - set(labels)
    if unknown:
        raise ValueError(f'Used tiles missing integer cell_type property: {sorted(unknown)}')
    settings = properties(doc.get('properties', []))
    allowed = {'movement_mode', 'max_velocity', 'fail_prob', 'max_episode_steps', 'seed', 'objective'}
    metadata = {key: settings[key] for key in allowed if key in settings}
    # A general level editor defaults to unrestricted direction; explicit properties override.
    metadata.setdefault('movement_mode', 'bidirectional')
    grid = np.asarray([labels[gid] for gid in gids], dtype=np.int32).reshape(height, width)
    return MapSpec(grid, path.stem, metadata)
