"""Bundled assets and public dispatch regressions."""
import contextlib
import io
from pathlib import Path
import unittest
from ocean_rl.cli import main
from ocean_rl.paths import MAPS_DIR, RESULTS_DIR


class DistributionTests(unittest.TestCase):
    def test_default_assets_match_examples(self):
        examples = Path(__file__).resolve().parents[1] / 'maps'
        for asset in MAPS_DIR.iterdir():
            self.assertEqual(asset.read_bytes(), (examples / asset.name).read_bytes())

    def test_output_is_relative(self):
        self.assertEqual(RESULTS_DIR, Path('results'))

    def test_system_dispatch(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                main(['--system', 'racetrack', '--help'])
        self.assertEqual(result.exception.code, 0)
        self.assertIn('--algorithm', output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(['--system', 'unknown'])
        self.assertEqual(result.exception.code, 2)
