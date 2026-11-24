# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction of the CAAF-ResUNet paper (`docs/PDF/medicina-61-01126.pdf`) for lung-nodule segmentation on 64×64 CT patches from LIDC-IDRI and LUNA16. Goal: reproduce Tables 5, 6, 7 and Figures 2, 3. Code comments, docstrings, docs and error messages are in **Vietnamese** — keep new ones in Vietnamese for consistency.

Planning documents are the source of truth and code references them by section (e.g. `SPECS §6.1`, `Phụ lục A`):
- `docs/PLANNING/SPECS.md` — detailed spec organized Phase 0 → Phase 9 + Appendices A (CLI), B (notebooks), C (tests).
- `docs/PLANNING/PLAN.md` — phase plan, success criteria, run matrix, risks.
- `docs/PLANNING/OPEN_ISSUES.md` — ambiguity log (IDs like B3–B17); `docs/SPEC_REVIEW.md` — rationale for decisions.

Config comments tag every value: `[P]` from the paper, `[A]` team assumption, `[F]` fix of a paper error. Preserve these tags when editing configs.

LUNA16 is a subset of LIDC-IDRI → never cross-train or merge the two datasets.

## Where things run

- **Local Windows** (Python 3.12.10, `requirements-local.txt`): no torch installed. Only write code and run torch-free tests.
- **Colab Pro CPU High-RAM**: data survey/indexing and preprocessing (notebooks `00_`, `01_`), working directly on Google Drive.
- **Colab Pro GPU** (`requirements-colab.txt`): model/loss tests, training, evaluation, stats/figures (notebooks `10_`, `11_`, `12_`).

`/data/` and `/experiments/` are gitignored (note the leading `/` so `src/data/` is still tracked).

## Commands

All scripts are run as modules from the repo root: `python -m src.<module>`.

```powershell
# Tests (torch-dependent tests auto-skip via pytest.importorskip when torch is absent)
pytest -q
pytest tests/test_config.py tests/test_ct_utils.py      # the ones meaningful locally
pytest tests/test_losses.py -k hausdorff                # single test subset

# Data pipeline (normally driven by notebooks 00/01 on Colab)
python -m src.data.build_index --lidc-dir data/raw/LIDC-IDRI --luna16-dir data/raw/LUNA16 --out-dir data/index
python -m src.data.preprocess_luna16 --out-dir data/interim/luna16 --limit 3     # --limit for debugging
python -m src.data.preprocess_lidc   --out-dir data/interim/lidc --limit 3
python -m src.data.make_splits --index data/interim/luna16/patches.csv --out data/splits/luna16.json
python -m src.data.pack_h5 --interim-dir data/interim/luna16 --splits data/splits/luna16.json --out-dir <dir>

# Train / evaluate / report (GPU)
python -m src.train --config configs/caaf_hausdorff.yaml --dataset luna16 --data-dir <processed> --out-dir <experiments>
python -m src.train ... --max-batches 2 --epochs 2        # smoke test
python -m src.train ... --overfit-steps 150               # 1-batch overfit sanity check
python -m src.train ... --resume
python -m src.evaluate --run-dir <experiments>/luna16/caaf_hausdorff_s42 [--ckpt last|best]
python -m src.stats_test --a <A>/per_patch_metrics.csv --b <B>/per_patch_metrics.csv --out <json>
python -m src.report --exp-root <experiments> --out-dir reports --ckpt last --seed 42
python -m src.visualize --run-dirs <run_dir> --kind {loss,dotplot,diffmap,gradcam,weights}
```

## Architecture

### Config system (`src/utils.py`)
- Experiment configs declare `base: base.yaml` and only list differences; `load_config` deep-merges, rejects duplicate YAML keys, and raises `KeyError` on unknown keys from `--set a.b.c=value`.
- `configs/data.yaml` is separate (preprocessing/splitting params, Colab Drive paths) and is consumed by `src/data/*` scripts, not by training.
- `src/utils.py` must stay torch-free (lazy torch import in checkpoint helpers) so `tests/test_config.py` runs locally. It therefore duplicates `DATASETS`, `MODEL_NAMES`, `AAF_MODES`, `LOSS_NAMES` — keep them in sync with `src/models` and `src/losses`.
- Run naming is deterministic (`make_run_name`): `{config_stem}_s{seed}`, plus ordered tokens for the 11 train shortcut flags (`model, aaf, loss, lam, ep, bs, lr, wd, amp|noamp, aug|noaug`) when they differ from the config, plus `set-<sha1[:6]>` for other `--set` overrides. A shortcut flag and the equivalent `--set` produce the same name.
- Each run dir `<out-dir>/<dataset>/<run_name>/` holds `config_resolved.yaml` (includes `dataset`, which evaluate/visualize read back), `overrides.json`, `commit.txt`, `env.json`, `train_log.csv`, `last.pt` (full state, atomic each epoch), `best.pt` (weights only). On `--resume` only `train.epochs` may differ from the resolved config.

### Model (`src/models/`)
`ResUNetFamily` in `caaf_resunet.py` is one network parameterized by `enc_attn` ∈ {`se`, `pa`, `ca`, `static`, `adaptive`}; all ablations (Tables 5–7) and the ResUNet++ baseline (`se`) are variants of it. `build_model(cfg)` picks via `model.name` and `aaf.mode`.
- Encoder: stem → RCM ×3 (stride 2), channels 32/64/128/256; an encoder-attention block after stem, RCM1, RCM2.
- `AAC` (`aac.py`) computes per-sample fusion weights `w = (w1 PA, w2 CA)` once from the stem output; the same `w` is reused at all three `AAF` blocks. `pa`/`ca`/`static` use constant `w`.
- Decoder uses `AttentionModule` (not AAF); skip connections take features **before** AAF. ASPP bridge and ASPP before the 1×1 head.
- `forward` returns a dict: `prob`, `logits`, `w`, `feats` (`aaf1..3`, used by Grad-CAM).

### Losses (`src/losses/`)
`BoundaryAwareLoss(cfg)` = Dice + λ·boundary, `loss.name` ∈ {`dice`, `dice+sobel`, `dice+laplacian`, `dice+hausdorff`}. Always computed in fp32 with autocast disabled. Hausdorff uses the precomputed EDT `dist` stored in the `.h5` (3 modes via `loss.hausdorff_mode`) and reports `d_pg`, `d_gp`, `empty_pred_frac`.

### Data (`src/data/`)
Pipeline: `build_index` (DICOM headers only) → `preprocess_luna16` / `preprocess_lidc` (per-scan `_DONE` markers for resume; LUNA16 reads `.mhd` CT + LIDC DICOM-SEG masks mapped via world coordinates, matched 1-1 to `annotations.csv`) → `make_splits` (patient-level, 0% leakage, exactly 1,200 test patches) → `pack_h5` (`{train,val,test}.h5` with `image`, `mask`, `dist`, `patch_id`, `patient_id` + SHA-256 manifest). Arrays are `(Z, Y, X)`; `ct_utils` functions must not mutate inputs and write atomically. `H5SegDataset` loads everything into RAM; optional D4 augmentation is applied identically to image/mask/dist and is deterministic in `(seed, epoch, index)` — call `set_epoch` each epoch.

### Metrics convention
Per-patch CSVs store metrics in `[0, 1]`; JSON summaries store percentages.

## Notebook ↔ CLI contract
`tests/test_notebooks_contract.py` parses each script's `build_parser()` via AST and checks that the 5 notebooks expose 100% of the CLI parameters in their first (parameter) cell and pass only flags that exist, and that no tokens/secrets are hardcoded. When adding/removing/renaming a CLI flag, update `build_parser()`, the corresponding notebook, and SPECS Appendix A together.
