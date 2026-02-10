# Path-Aware-Multipath-Instability
Fluid-model simulation for studying multipath scheduling instability in path-aware networks (PAN).

## Directory Structure

- `src/`              Core model components (network, algorithms, scheduler)
- `scripts/`          Experiment and figure-generation scripts
- `outputs/`          All generated figures and result CSVs
- `archive/`          Older/unused scripts (for reference only)

## Requirements

Python 3.8+ with:
- numpy
- matplotlib
- pandas
- seaborn

## Usage

Run any script in `scripts/` directly, e.g.:

```bash
python scripts/generate_fig_ring_N8_instability.py
python scripts/sweep_stability_margin_vs_N.py
