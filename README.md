# ColliderX

ColliderX is a local-first particle decay simulator (MVP).

Run a short end-to-end flow: event simulation → Feynman diagram → summary plots.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (optional) install Graphviz if you want rendered diagrams
# macOS: brew install graphviz
```

## Quick Run

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
