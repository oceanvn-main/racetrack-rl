"""Verify a built wheel outside the source tree (no dependency downloads)."""
from pathlib import Path
import subprocess
import site
import sys
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / 'dist').glob('*.whl'))
    if len(wheels) != 1:
        raise SystemExit('Expected exactly one wheel in dist; build a clean release.')
    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(wheels[0]) as archive:
            archive.extractall(folder)
        code = '''
import sys
sys.path.insert(0, sys.argv[1])
sys.path.append(sys.argv[2])  # Dependency location; do not execute editable .pth files.
from ocean_rl.environments.racetrack import RacetrackEnv
from ocean_rl.algorithms.monte_carlo.off_policy import OffPolicyMCAgent
from ocean_rl.experiments.racetrack import build_parser
from ocean_rl.cli import main
assert len(RacetrackEnv().start_cells) > 0
assert OffPolicyMCAgent.getMetadata()['id'] == 'OffPolicyMCAgent'
for source in build_parser().parse_args([]).maps:
    RacetrackEnv.from_file(source)
main(['--system', 'racetrack', '--help'])
'''
        subprocess.run([sys.executable, '-I', '-c', code, folder, site.getusersitepackages()],
                       cwd=folder, check=True)


if __name__ == '__main__':
    main()
