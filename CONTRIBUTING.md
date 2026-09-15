# Contributing

## Setup and checks

Use Python 3.11 or newer. From this directory:

```sh
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m ruff check ocean_rl tests
python -m build
python tools/check_wheel.py
```

## Extension rules

Keep learning equations readable. Algorithms depend on environment interfaces,
not maps or plotting. Add algorithms under their mathematical family and register
them in registry.py. New systems get an environment, experiment and evaluator;
extend cli.py when their implementation is ready. Continuous control requires
interfaces beyond the current discrete integer-vector state contract.

Test real transitions and updates with fixed seeds, including termination,
truncation and invalid inputs. Document scientific assumptions. A finished
training budget does not establish convergence or optimality.

Bundled defaults live in ocean_rl/data/maps; editable copies live in maps.
Update both when changing defaults. Legacy launchers support existing source
checkouts; new integrations import ocean_rl.

## Review and publication

Describe the problem, behavior change and validation. Provide commands, seeds
and configuration for bugs. Exclude generated results, environments and secrets.
Reports contain local source paths; review them before sharing.

No open-source license has been selected. The owner must choose a license and
check image/map provenance before representing this as open source. Review the
user-supplied obstacle screenshot and derived maps in particular. Import support
does not imply permission to redistribute third-party images.

The GitHub workflow lives at the Ocean Library repository root. If publishing
only this Python folder, copy that workflow to .github/workflows/python-rl.yml
in the new repository; it supports either layout.
