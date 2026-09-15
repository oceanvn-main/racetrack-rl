"""Reward strategies use actual motion, including terminal boundary contact."""
import math
import unittest
from ocean_rl.environments.racetrack import RacetrackEnv
from ocean_rl.algorithms.monte_carlo.off_policy import OffPolicyMCAgent
from ocean_rl.objectives import register_objective


class ObjectiveTests(unittest.TestCase):
    def test_json_cli_override_and_saved_distance(self):
        import contextlib
        import io
        import json
        from test_agent import test_directory
        from ocean_rl.experiments.racetrack import main
        with test_directory() as folder:
            (folder / 'track.map').write_text('S G')
            source = folder / 'track.json'
            source.write_text(json.dumps({'map': 'track.map', 'environment': {
                'objective': 'minimum-time', 'fail_prob': 0, 'max_velocity': 2}}))
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(['--maps', str(source), '--objective', 'geometric-distance',
                               '--episodes', '20', '--output-dir', str(folder / 'out')])
            self.assertEqual(result, 0)
            report = json.loads(next((folder / 'out').glob('*.json')).read_text())
            self.assertEqual(report['environment']['objective'], 'geometric-distance')
            for trial in report['evaluation']:
                self.assertAlmostEqual(trial['return'], -trial['distance'])

    def env(self, rows, **kwargs):
        return RacetrackEnv.from_ascii(rows, objective='geometric-distance',
                                      fail_prob=0, **kwargs)

    def test_distance_and_terminal_segment(self):
        env = self.env(['S  G'])
        env.reset()
        _, reward, done, _, info = env.step(env.actions.index((1, 0)))
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        _, reward, done, _, info = env.step(env.actions.index((1, 0)))
        self.assertTrue(done)
        self.assertEqual(info['distance'], 1.5)
        self.assertEqual(reward, -1.5)

    def test_diagonal_cost(self):
        env = self.env(['  G', '   ', 'S  '])
        env.reset()
        _, reward, _, _, _ = env.step(env.actions.index((1, 1)))
        self.assertAlmostEqual(reward, -math.sqrt(2))

    def test_collision_excludes_reset_teleport(self):
        env = self.env(['S #', '  G'], movement_mode='bidirectional')
        env.reset()
        env.step(env.actions.index((1, 0)))
        state, reward, _, _, info = env.step(env.actions.index((1, 0)))
        self.assertTrue(info['crashed'])
        self.assertEqual(tuple(state[:2]), (0, 0))
        self.assertEqual(reward, -.5)

    def test_wait_mask_clone_and_discount(self):
        env = self.env(['S G'])
        state, _ = env.reset()
        wait = env.actions.index((0, 0))
        self.assertNotIn(wait, env.get_valid_actions(state))
        with self.assertRaises(ValueError):
            env.step(wait)
        self.assertEqual(env.clone().objective, 'geometric-distance')
        with self.assertRaises(ValueError):
            OffPolicyMCAgent(env, gamma=.9)

    def test_extension_without_environment_edits(self):
        class Cost:
            allow_stationary = True
            def reward(self, transition):
                return -2 * transition.distance
        import uuid
        name = 'test-' + uuid.uuid4().hex
        register_objective(name, Cost)
        env = RacetrackEnv.from_ascii(['S G'], objective=name, fail_prob=0)
        env.reset()
        self.assertEqual(env.step(env.actions.index((1, 0)))[1], -2)
