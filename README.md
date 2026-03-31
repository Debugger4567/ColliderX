# ColliderX

ColliderX is a local-first particle decay simulator (MVP).

Run a short end-to-end flow: event simulation → Feynman diagram → summary plots.

## Install

From PyPI:

```bash
pip install colliderx
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Graphviz (optional, only needed for rendered `feynman.png`):

macOS:

```bash
brew install graphviz
```

Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y graphviz
```

Windows (PowerShell):

```powershell
# Option 1: winget
winget install Graphviz.Graphviz

# Option 2: Chocolatey
choco install graphviz
```

Verify Graphviz is available:

```bash
dot -V
```

Note: `pip install colliderx` cannot reliably auto-install system Graphviz (`dot`) across all OS/package managers. ColliderX will still run without it and fall back to DOT-only output (`feynman.dot`).

## Quick Run

If installed from PyPI, run the installed CLI command from any folder:

```bash
colliderx -p "Muon" -n 1000 --save --no-show -o artifacts/flow_muon
```

If running from source, first `cd` into the repository root, then use `main.py`:

```bash
cd /path/to/ColliderX
python main.py -p "Muon" -n 1000 --save --no-show -o artifacts/flow_muon
```

Other examples (from source):

Default (interactive, show-only):

```bash
python main.py -p "Muon" -n 1000
```

Save outputs to disk and avoid interactive windows:

```bash
python main.py -p "Muon" -n 1000 --save --no-show -o artifacts/flow_muon
```

Run with a seed:

```bash
python main.py -p "Muon" -n 1000 -s 42
```

## CLI Flags (short)

- `-p, --particle` : parent particle name (positional also supported)
- `-n, --events`   : number of events to simulate
- `-s, --seed`     : random seed
- `-E, --energy`   : override parent energy (MeV)
- `--save`         : persist generated files to `-o/--out`
- `--no-show`      : do not open interactive plot windows
- `-o, --out`      : output directory (default `artifacts/run_<timestamp>`)
- `--decay`        : force a particular root decay mode
- `--afb`          : enable forward-backward asymmetry (where applicable)

## Saved Outputs (`--save`)

- `feynman.dot` (DOT source)
- `feynman.png` (if Graphviz is available and rendering enabled)
- `graphs/` (PNG plots: decay modes, spectra, process-specific plots)
- `summary.txt` (run summary and counts)

## Tests

```bash
pytest -q
```

If Graphviz is not installed, the tool falls back to writing DOT files only and continues the flow.
