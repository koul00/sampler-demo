# Data files (not stored in git)

The pickled feature files are hundreds of MB each and exceed GitHub's limits,
so they are **not** committed. Download them and either drop them here or point
the script at wherever they live:

    python run_attention_maps.py --data-dir /path/to/SAMPLERtest

Files used by this demo (from the SAMPLER test bundle, Zenodo record 10346576):

| File | Size | What it holds |
|------|------|---------------|
| `TCGA_BRCA_DX20X.p` | ~250 MB | Train/test SAMPLER representations + labels |
| `Dtab.p` | ~15 MB | Trained likelihood-ratio lookup tables (the "model") |
| `TCGA-AR-A0TP-...svs.p` | ~250 MB | Tile-level features + locations for slide 1 |
| `TCGA-B6-A0RQ-...svs.p` | ~185 MB | Tile-level features + locations for slide 2 |

SAMPLER representations for many TCGA/CPTAC cohorts are linked from the
original repo: https://github.com/TheJacksonLaboratory/SAMPLER
