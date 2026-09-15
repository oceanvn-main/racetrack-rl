"""Tiled adapter regressions using actual tile-map files."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ocean_rl.maps.tiled import from_tiled
from ocean_rl.maps.io import load_map
from ocean_rl.environments.racetrack import RacetrackEnv
from test_environment import map_directory
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class TiledTests(unittest.TestCase):
    def document(self):
        return {'type':'map','orientation':'orthogonal','infinite':False,'width':3,'height':1,
                'layers':[{'type':'tilelayer','name':'Racetrack','width':3,'height':1,'data':[12,11,13]}],
                'tilesets':[{'firstgid':10,'tilecount':4,'tiles':[{'id':i,'properties':[{'name':'cell_type','type':'int','value':i}]} for i in range(4)]}]}

    def test_inline_nondefault_firstgid_and_default_direction(self):
        spec=from_tiled('inline.tmj',self.document())
        self.assertEqual(spec.grid.tolist(),[[2,1,3]])
        self.assertEqual(spec.metadata['movement_mode'],'bidirectional')

    def test_external_starters_match_originals(self):
        for name, original in [('harbor_bend','harbor_bend')]:
            actual=load_map(ROOT/'maps'/'tiled'/(name+'.tmj'))
            expected=load_map(ROOT/'maps'/(original+'.json'))
            np.testing.assert_array_equal(actual.grid,expected.grid)
            RacetrackEnv.from_file(ROOT/'maps'/'tiled'/(name+'.tmj'))

    def test_flipped_semantics_and_erased_wall(self):
        doc=self.document()
        doc['layers'][0]['data']=[12|0x80000000,0,13|0x40000000]
        self.assertEqual(from_tiled('x.tmj',doc).grid.tolist(),[[2,0,3]])

    def test_plain_json_export_and_config_wrapper(self):
        with map_directory() as directory:
            (directory/'map.json').write_text(json.dumps(self.document()))
            self.assertEqual(load_map(directory/'map.json').grid.tolist(),[[2,1,3]])
            (directory/'map.tmj').write_text(json.dumps(self.document()))
            (directory/'config.json').write_text(json.dumps({'map':'map.tmj','environment':{'movement_mode':'forward-only'}}))
            self.assertEqual(RacetrackEnv.from_file(directory/'config.json').movement_mode,'forward-only')

    def test_reject_unsupported_or_ambiguous_geometry(self):
        mutations=[lambda d:d.update(infinite=True),lambda d:d.update(orientation='isometric'),
                   lambda d:d['layers'][0].update(offsetx=1),lambda d:d['layers'][0].update(data='encoded'),
                   lambda d:d['layers'][0].update(name='Other'),lambda d:d['layers'].append(copy.deepcopy(d['layers'][0])),
                   lambda d:d['layers'][0].update(data=[12,99,13]),lambda d:d['layers'][0].update(data=[12,True,13]),
                   lambda d:d['tilesets'][0]['tiles'][0]['properties'][0].update(value=7)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                doc=self.document(); mutate(doc)
                with self.assertRaises(ValueError): from_tiled('x.tmj',doc)

if __name__=='__main__': unittest.main()
