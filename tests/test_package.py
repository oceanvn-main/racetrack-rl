"""Extension and compatibility tests with real MC training."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_agent import test_directory
from ocean_rl.algorithms.monte_carlo.off_policy import OffPolicyMCAgent
from ocean_rl.environments.racetrack import RacetrackEnv
from ocean_rl.evaluation.racetrack import evaluate_racetrack
from ocean_rl.experiments.racetrack import main
from ocean_rl.registry import register_algorithm, create_agent, resolve_options


class EncapsulatedMC:
    """An actual learner exposing only the experiment contract, without Q/log_C."""
    def __init__(self, env, *, seed=None, epsilon=.2):
        self.learner = OffPolicyMCAgent(env, seed=seed, epsilon=epsilon)

    @property
    def training_history(self):
        return self.learner.training_history

    @property
    def status(self):
        return self.learner.status

    def train(self, *args, **kwargs):
        return self.learner.train(*args, **kwargs)

    def act(self, state):
        return self.learner.act(state)

    def save_checkpoint(self, prefix):
        return self.learner.save_checkpoint(prefix)


class PackageTests(unittest.TestCase):
    def test_default_assets_and_metadata_resolve(self):
        self.assertGreater(len(RacetrackEnv().start_cells), 0)
        self.assertEqual(OffPolicyMCAgent.getMetadata()['id'], 'OffPolicyMCAgent')

    def test_checkpoint_does_not_reset_unused_environment(self):
        env = RacetrackEnv.from_ascii(['S G'], seed=6)
        agent = OffPolicyMCAgent(env)
        random_state = json.dumps(env.rng.bit_generator.state)
        with test_directory() as folder:
            path = agent.save_checkpoint(folder / 'untrained')
            with np.load(path) as data:
                self.assertEqual(data['states'].shape, (0, 0))
                self.assertEqual(data['Q'].shape, (0, env.action_count))
        self.assertIsNone(env.current_state)
        self.assertEqual(json.dumps(env.rng.bit_generator.state), random_state)

    def test_registry_validation_and_option_isolation(self):
        defaults = resolve_options('off-policy-mc')
        defaults['epsilon'] = .9
        self.assertEqual(resolve_options('off-policy-mc')['epsilon'], .1)
        env = RacetrackEnv.from_ascii(['S G'])
        with self.assertRaises(ValueError):
            create_agent('unknown-algorithm', env)
        with self.assertRaises(ValueError):
            create_agent('off-policy-mc', env, nonexistent=4)
        with self.assertRaises(ValueError):
            register_algorithm('off-policy-mc', OffPolicyMCAgent)
        with self.assertRaises(ValueError):
            register_algorithm('../bad-name', OffPolicyMCAgent)

    def test_shared_evaluation_with_real_bidirectional_training(self):
        env = RacetrackEnv.from_ascii(['G S'], movement_mode='bidirectional',
                                     max_velocity=2, fail_prob=0, seed=7)
        agent = create_agent('off-policy-mc', env, seed=11)
        agent.train(50, 10, verbose=False)
        results = evaluate_racetrack(env, agent.act, max_steps=20)
        self.assertTrue(results[0]['finished'])
        self.assertEqual(results[0]['steps'], 2)

    def test_runner_accepts_learner_without_exposed_tables(self):
        name = 'encapsulated-' + uuid.uuid4().hex
        register_algorithm(name, EncapsulatedMC, defaults={'epsilon': .2})
        with test_directory() as folder:
            source = folder / 'map.map'
            source.write_text('S G\n')
            options = folder / 'options.json'
            options.write_text(json.dumps({'epsilon': .3}))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['--algorithm', name, '--maps', str(source),
                             '--episodes', '20', '--fail-prob', '0', '--max-velocity', '2',
                             '--agent-options', str(options), '--epsilon', '.4',
                             '--output-dir', str(folder / 'results')])
            self.assertEqual(code, 0)
            report = json.loads(next((folder / 'results').glob('*.json')).read_text())
            self.assertEqual(report['algorithm'], name)
            self.assertEqual(report['agent'], {'epsilon': .4})
            self.assertEqual(report['status']['episodes'], 20)
            self.assertTrue(Path(report['checkpoint']).exists())

    def test_module_cli_help(self):
        result = subprocess.run([sys.executable, '-m', 'ocean_rl', '--help'],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--algorithm', result.stdout)
        self.assertIn('--agent-options', result.stdout)


if __name__ == '__main__':
    unittest.main()
