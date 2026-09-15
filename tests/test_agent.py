"""Behavioral MC regressions with real small environments and exact returns."""
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocean_rl.environments.racetrack import RacetrackEnv
from ocean_rl.algorithms.monte_carlo.off_policy import OffPolicyMCAgent
from ocean_rl.errors import EpisodeTruncatedError


@contextlib.contextmanager
def test_directory():
    # Inherit workspace ACLs; Windows sandbox cannot reopen mode-0700 tempdirs.
    folder = ROOT / 'tests' / ('agent_test_' + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        for item in sorted(folder.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if item.is_file():
                # Windows scanners can briefly hold a newly written checkpoint.
                for attempt in range(5):
                    try:
                        item.unlink()
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(.05 * (attempt + 1))
            elif item.is_dir():
                item.rmdir()
        folder.rmdir()


class AgentTests(unittest.TestCase):
    def env(self, **kwargs):
        return RacetrackEnv.from_ascii(['S G'], max_velocity=2, fail_prob=0, **kwargs)

    def test_distribution_includes_all_valid_actions_and_ties(self):
        env = self.env()
        state, _ = env.reset()
        agent = OffPolicyMCAgent(env, epsilon=.2)
        valid = env.get_valid_actions(state)
        p = agent.behavior_distribution(state)
        np.testing.assert_allclose(p[valid], 1 / len(valid))
        agent._values(state)[valid[-1]] = 0
        p = agent.behavior_distribution(state)
        self.assertAlmostEqual(p.sum(), 1)
        self.assertAlmostEqual(p[valid[-1]], .8 + .2 / len(valid))
        self.assertTrue(np.all(p[valid] > 0))
        self.assertTrue(np.all(p[[i for i in range(env.action_count) if i not in valid]] == 0))

    def test_recorded_probability_drives_exact_weighted_update(self):
        env = self.env()
        agent = OffPolicyMCAgent(env, init_q=-100)
        state0, state1 = (0, 0, 0, 0), (0, 1, 1, 0)
        right, coast = env.actions.index((1, 0)), env.actions.index((0, 0))
        # Last transition has probability .25: predecessor gets importance weight 4.
        transitions = [(state0, right, -1., .5), (state1, coast, 0., .25)]
        agent._update_episode(transitions)
        self.assertAlmostEqual(agent.Q[state0][right], -1)
        self.assertAlmostEqual(agent.log_C[state0][right], math.log(4))
        self.assertAlmostEqual(agent.Q[state1][coast], 0)
        # Its stored generating probability remains .25 even though Q changed.
        self.assertNotAlmostEqual(agent.get_behavior_probability(state1, coast), .25)

    def test_log_weights_stay_finite(self):
        env = self.env()
        agent = OffPolicyMCAgent(env)
        state = (0, 1, 1, 0)
        coast = env.actions.index((0, 0))
        agent._update_episode([(state, coast, -1., .01)] * 1000)
        self.assertTrue(np.isfinite(agent.Q[state]).all())
        self.assertGreater(agent.log_C[state][coast], 4000)

    def test_signed_states_do_not_alias(self):
        env = RacetrackEnv.from_ascii(['G S'], movement_mode='bidirectional')
        agent = OffPolicyMCAgent(env)
        agent._values((0, 1, -1, 0))[0] = 17
        self.assertEqual(agent._values((0, 1, 1, 0))[0], -1000)
        self.assertEqual(len(agent.Q), 2)

    def test_timeout_does_not_update_q_or_weights(self):
        env = RacetrackEnv.from_ascii(['S G'], fail_prob=1, max_episode_steps=2)
        agent = OffPolicyMCAgent(env, seed=0)
        with self.assertRaises(EpisodeTruncatedError):
            agent.train(1, 1, verbose=False)
        self.assertEqual(agent.status['truncated_episodes'], 1)
        self.assertFalse(agent.status['completed'])
        self.assertTrue(all(np.all(value == agent.init_q) for value in agent.Q.values()))
        self.assertTrue(all(np.isneginf(value).all() for value in agent.log_C.values()))

    def test_real_training_learns_easy_map_and_is_reproducible(self):
        agents = []
        for _ in range(2):
            agent = OffPolicyMCAgent(self.env(seed=4, max_episode_steps=10000), seed=9)
            history = agent.train(50, 7, verbose=False)
            self.assertEqual(history[0][-1], 50)
            agents.append(agent)
        self.assertEqual(agents[0].training_history, agents[1].training_history)
        self.assertTrue(all(item['finished'] for item in agents[0].evaluate_learned_policy()))

    def test_evaluation_does_not_mutate_training_state_rng_or_tables(self):
        env = self.env(seed=5)
        agent = OffPolicyMCAgent(env, seed=7)
        env.reset()
        state = env.current_state
        env_rng = json.dumps(env.rng.bit_generator.state)
        agent_rng = json.dumps(agent.rng.bit_generator.state)
        tables = len(agent.Q)
        first = agent.evaluate_learned_policy(max_steps=10)
        self.assertEqual(first, agent.evaluate_learned_policy(max_steps=10))
        self.assertEqual(env.current_state, state)
        self.assertEqual(json.dumps(env.rng.bit_generator.state), env_rng)
        self.assertEqual(json.dumps(agent.rng.bit_generator.state), agent_rng)
        self.assertEqual(len(agent.Q), tables)

    def test_invalid_training_inputs(self):
        for epsilon in (0, -1, float('nan'), 1.1):
            with self.assertRaises(ValueError):
                OffPolicyMCAgent(self.env(), epsilon=epsilon)
        for episodes, interval in [(0, 1), (1, 0), (1.5, 1), (True, 1)]:
            with self.assertRaises(ValueError):
                OffPolicyMCAgent(self.env()).train(episodes, interval, verbose=False)

    def test_cli_import_from_other_working_directory(self):
        with test_directory() as folder:
            folder = Path(folder)
            (folder / 'custom.map').write_text('S G\n')
            (folder / 'custom.json').write_text(json.dumps({'map': 'custom.map', 'environment': {'max_velocity': 2, 'fail_prob': 0}}))
            sub_env = dict(sys.executable_env if hasattr(sys, 'executable_env') else {}, PYTHONPATH=str(ROOT))
            import os
            sub_env = dict(os.environ, PYTHONPATH=str(ROOT))
            result = subprocess.run([sys.executable, '-m', 'ocean_rl',
                                     '--maps', str(folder / 'custom.json'), '--episodes', '20',
                                     '--seeds', '3', '--output-dir', str(folder / 'results')],
                                    cwd=folder, env=sub_env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(next((folder / 'results').glob('*.json')).read_text())
            self.assertTrue(report['status']['completed'])
            self.assertEqual(len(report['training']), 20)
            self.assertTrue(next((folder / 'results').glob('*_policy.npz')).exists())

    def test_cli_timeout_reports_failure(self):
        with test_directory() as folder:
            folder = Path(folder)
            source = folder / 'timeout.map'
            source.write_text('S G\n')
            result = subprocess.run([sys.executable, '-m', 'ocean_rl',
                                     '--maps', str(source), '--episodes', '1', '--fail-prob', '1',
                                     '--max-steps', '1', '--output-dir', str(folder / 'results')],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads(next((folder / 'results').glob('*.json')).read_text())
            self.assertIn('Zero Q-value updates', report['error'])
            self.assertFalse(report['status']['completed'])


if __name__ == '__main__':
    unittest.main()
