"""Map and dynamics regressions: small explicit geometry fixtures, no training mocks."""
import json
from pathlib import Path
import sys
from contextlib import contextmanager
import uuid
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocean_rl.maps.io import MapSpec, from_ascii, from_image, load_map
from ocean_rl.environments.racetrack import RacetrackEnv, traversed_cells, spaces


@contextmanager
def map_directory():
    # Inherit workspace permissions; Windows sandbox cannot enter mode-0700
    # directories created by Python's TemporaryDirectory.
    root = Path(__file__).resolve().parent
    directory = root / ('map_test_' + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == root
        for file in directory.iterdir():
            file.unlink()
        directory.rmdir()


class MapTests(unittest.TestCase):
    def test_strict_grid_labels_and_shape(self):
        for grid in ([], [0, 2, 3], [[2., 3.]], [[2, 4, 3]], [[1, 3]], [[2, 1]], [[True, False]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                MapSpec(grid)

    def test_grid_copied_and_immutable(self):
        original = np.array([[2, 3]])
        spec = MapSpec(original)
        original[0, 0] = 0
        self.assertEqual(spec.grid[0, 0], 2)
        with self.assertRaises(ValueError):
            spec.grid[0, 0] = 1
        with self.assertRaises(ValueError):
            spec.grid.flags.writeable = True
        with self.assertRaises(TypeError):
            spec.metadata['x'] = 1

    def test_ascii_rejects_ragged_unknown_and_empty(self):
        for rows in ([], [''], ['SG', '#'], ['S?G'], ['SG', '']):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                from_ascii(rows)

    def test_ascii_preserves_whitespace_and_custom_palette(self):
        spec = from_ascii(['S G', '   '])
        np.testing.assert_array_equal(spec.grid, [[2, 1, 3], [1, 1, 1]])
        self.assertEqual(from_ascii(['ab'], {'a': 2, 'b': 3}).grid.tolist(), [[2, 3]])

    def test_files_and_relative_json_metadata(self):
        with map_directory() as directory:
            root = Path(directory)
            (root / 'floor.map').write_text('S G\n', encoding='utf-8-sig')
            np.save(root / 'floor.npy', np.array([[2, 1, 3]]))
            (root / 'floor.json').write_text(json.dumps({'map': 'floor.map', 'name': 'floor',
                'environment': {'max_velocity': 3, 'movement_mode': 'bidirectional'}}), encoding='utf-8-sig')
            np.testing.assert_array_equal(load_map(root / 'floor.npy').grid, load_map(root / 'floor.map').grid)
            env = RacetrackEnv.from_file(root / 'floor.json', max_velocity=4)
            self.assertEqual(env.map_spec.name, 'floor')
            self.assertEqual(env.max_velocity, 4)
            self.assertEqual(env.movement_mode, 'bidirectional')
            with self.assertRaises(FileNotFoundError):
                load_map(root / 'missing.map')

    def test_invalid_json_and_unknown_configuration(self):
        with map_directory() as directory:
            path = Path(directory) / 'bad.json'
            for config in ({}, {'map': 'bad.json'}, {'map': 'a.map', 'guess': True},
                           {'map': 2}, {'map': 'a.map', 'environment': []}, {'map': 'a.map', 'name': 1}):
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    load_map(path)
            (path.parent / 'a.map').write_text('SG\n')
            path.write_text(json.dumps({'map': 'a.map', 'environment': {'typo': 1}}))
            with self.assertRaises(ValueError):
                RacetrackEnv.from_file(path)

    def test_palette_image_and_explicit_annotations(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        image = Image.fromarray(np.array([[[255, 0, 0], [255, 255, 255], [0, 255, 0]]], dtype=np.uint8))
        self.assertEqual(from_image(image).grid.tolist(), [[2, 1, 3]])
        gray = Image.fromarray(np.full((2, 3), 255, dtype=np.uint8))
        spec = from_image(gray, start_cells=[(0, 0)], finish_cells=[(1, 2)])
        self.assertEqual(spec.grid[1, 2], 3)
        with self.assertRaises(ValueError):
            from_image(gray)
        with self.assertRaises(ValueError):
            from_image(gray, start_cells=[(0, 0)])
        for start, finish in (([(0, 0)], [(0, 0)]), ([(9, 0)], [(1, 2)]), ([(0.5, 0)], [(1, 2)])):
            with self.subTest(start=start), self.assertRaises(ValueError):
                from_image(gray, start_cells=start, finish_cells=finish)
        with self.assertRaises(ValueError):
            from_image(Image.fromarray(np.array([[[10, 10, 10]]], dtype=np.uint8)))

    def test_directional_reachability_and_each_start(self):
        with self.assertRaises(ValueError):
            RacetrackEnv.from_ascii(['S ', ' G'])
        RacetrackEnv.from_ascii(['S ', ' G'], movement_mode='bidirectional')
        for mode in ('forward-only', 'bidirectional'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                RacetrackEnv.from_ascii(['SG#S'], movement_mode=mode)

    def test_all_shipped_maps_validate(self):
        root = Path(__file__).resolve().parents[1] / 'maps'
        paths = [path for path in root.glob('*.json')
                 if 'map' in json.loads(path.read_text(encoding='utf-8-sig'))]
        self.assertTrue({'track_1.json', 'house.json', 'maze.json'} <= {path.name for path in paths})
        for path in paths:
            with self.subTest(path=path):
                env = RacetrackEnv.from_file(path)
                self.assertGreater(len(env.start_cells), 0)


class DynamicsTests(unittest.TestCase):
    def test_finish_before_wall_clamps_observation(self):
        env = RacetrackEnv.from_ascii(['S G#'], fail_prob=0, max_velocity=6)
        env.reset()
        env.current_state = (0, 0, 4, 0)
        obs, reward, done, truncated, info = env.step(4)
        self.assertEqual(tuple(obs), (0, 2, 4, 0))
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        self.assertFalse(truncated or info['crashed'])
        if spaces is not None:
            self.assertTrue(env.observation_space.contains(obs))

    def test_wall_before_finish_crashes_even_at_speed(self):
        env = RacetrackEnv.from_ascii(['S # G', '     '], fail_prob=0, max_velocity=6,
                                      movement_mode='bidirectional')
        env.reset()
        env.current_state = (0, 0, 4, 0)
        obs, reward, done, _, info = env.step(4)
        self.assertTrue(info['crashed'])
        self.assertFalse(done)
        self.assertEqual(tuple(obs), (0, 0, 0, 0))
        self.assertEqual(reward, -1)

    def test_corner_wall_beats_simultaneous_finish(self):
        env = RacetrackEnv.from_ascii(['SG', '# '], fail_prob=0, movement_mode='bidirectional')
        env.reset()
        obs, _, done, _, info = env.step(6)  # +x, -y (down/right)
        self.assertTrue(info['crashed'])
        self.assertFalse(done)
        self.assertEqual(tuple(obs), (0, 0, 0, 0))

    def test_supercover_exact_corner_groups(self):
        self.assertEqual(list(traversed_cells(0, 0, 2, 2)),
                         [[(1, 0), (0, 1), (1, 1)], [(2, 1), (1, 2), (2, 2)]])
        self.assertEqual(list(traversed_cells(1, 1, 0, -1)), [[(1, 0)]])

    def test_signed_movement_and_action_mask(self):
        env = RacetrackEnv.from_ascii(['GS'], fail_prob=0, movement_mode='bidirectional')
        obs, _ = env.reset()
        self.assertIn(1, env.get_valid_actions(obs))
        obs, _, done, _, _ = env.step(1)
        self.assertEqual(tuple(obs), (0, 0, -1, 0))
        self.assertTrue(done)
        if spaces is not None:
            self.assertTrue(env.observation_space.contains(obs))

    def test_cannot_stop_off_start_and_velocity_clips(self):
        env = RacetrackEnv.from_ascii(['S   G'], fail_prob=0, max_velocity=2)
        env.reset()
        env.step(7)
        self.assertNotIn(1, env.get_valid_actions(env.current_state))
        obs, _, _, _, _ = env.step(1)
        self.assertEqual(tuple(obs), (0, 2, 1, 0))
        obs, _, _, _, _ = env.step(7)
        self.assertEqual(tuple(obs), (0, 3, 1, 0))

    def test_outside_map_crashes_to_valid_start(self):
        env = RacetrackEnv.from_ascii(['SG'], fail_prob=0)
        env.reset()
        obs, _, done, _, info = env.step(5)
        self.assertTrue(info['crashed'])
        self.assertFalse(done)
        self.assertEqual(tuple(obs), (0, 0, 0, 0))

    def test_noise_preserves_rest_and_truncation(self):
        env = RacetrackEnv.from_ascii(['SG'], fail_prob=1, max_episode_steps=2)
        initial, _ = env.reset()
        obs, _, done, truncated, info = env.step(7)
        np.testing.assert_array_equal(obs, initial)
        self.assertTrue(info['acceleration_failed'])
        self.assertFalse(done or truncated)
        self.assertTrue(env.step(7)[3])
        with self.assertRaises(RuntimeError):
            env.step(7)

    def test_reset_seed_clone_and_global_rng_independence(self):
        env = RacetrackEnv.from_ascii(['SSG'], fail_prob=.5)
        clone = env.clone(seed=13)
        np.random.seed(7)
        expected = np.random.random(3)
        np.random.seed(7)
        env.reset(seed=13)
        clone.reset(seed=13)
        for _ in range(8):
            left, right = env.step(4), clone.step(4)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])
        np.testing.assert_array_equal(np.random.random(3), expected)
        self.assertEqual(tuple(env.reset(options={'start_cell': (0, 1)})[0]), (0, 1, 0, 0))
        with self.assertRaises(ValueError):
            env.reset(options={'start_cell': (0, 2)})
        with self.assertRaises(ValueError):
            env.reset(options={'start_cell': (0., 1.)})

    def test_argument_errors_and_step_lifecycle(self):
        for args in ({'max_velocity': 1}, {'max_velocity': 2.5}, {'fail_prob': float('nan')},
                     {'max_episode_steps': 0}, {'movement_mode': 'auto'}, {'track_id': 2}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                RacetrackEnv(**args)
        env = RacetrackEnv.from_ascii(['SG'])
        with self.assertRaises(RuntimeError):
            env.step(7)
        env.reset()
        for action in (-1, 9, 1.5, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)

    def test_custom_actions(self):
        from ocean_rl.environments.racetrack import make_action_grid
        grid_25 = make_action_grid(2, 2)
        self.assertEqual(len(grid_25), 25)
        env_25 = RacetrackEnv.from_ascii(['S   G'], custom_actions=grid_25, fail_prob=0)
        self.assertEqual(env_25.action_count, 25)
        self.assertEqual(env_25.actions, grid_25)
        if spaces is not None:
            self.assertEqual(env_25.action_space.n, 25)

        cloned = env_25.clone()
        self.assertEqual(cloned.action_count, 25)
        self.assertEqual(cloned.actions, grid_25)

        # Invalid custom action specifications
        for bad in ([], [1, 2], [(1,)], [("a", 1)], [(1.5, 0)], [(True, 1)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RacetrackEnv.from_ascii(['SG'], custom_actions=bad)

    @unittest.skipIf(spaces is None, 'Gymnasium is not installed')
    def test_gymnasium_checker(self):
        from gymnasium.utils.env_checker import check_env
        for mode in ('forward-only', 'bidirectional'):
            check_env(RacetrackEnv.from_ascii(['S  G'], movement_mode=mode), skip_render_check=True)


if __name__ == '__main__':
    unittest.main()
