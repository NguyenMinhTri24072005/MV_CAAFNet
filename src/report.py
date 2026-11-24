"""Sinh bảng so sánh Markdown với bài báo — SPECS §9.2.

    python -m src.report --exp-root "$DRIVE_ROOT/experiments" --out-dir reports --ckpt last --seed 42 \
        [--include-sensitivity]

Đọc `<exp-root>/<ds>/<run>/eval_test_<ckpt>/results.json` (+ `per_patch_metrics.csv` cho t-test).
Bảng chính chỉ lấy run chuẩn `{config}_s{seed}` (bỏ mọi thư mục có hậu tố `__*`). Mỗi bảng có cột
**Bài báo** và **Δ = ta − bài báo**. Sinh: `compare_table5.md`, `compare_table6.md` (gọi `stats_test`),
`compare_table7.md`, `seed_variance.md`, và `sensitivity.md` khi có `--include-sensitivity`.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.stats_test import paired_test
from src.utils import save_json

DATASETS = ("luna16", "lidc")
METRICS = ("dice", "iou", "sensitivity", "miss_rate", "specificity")
METRIC_LABEL = {"dice": "Dice (%)", "iou": "IoU (%)", "sensitivity": "Sensitivity (%)", "miss_rate": "Miss rate (%)",
                "specificity": "Specificity (%)", "w1": "w1 (PA)", "w2": "w2 (CA)"}
DS_LABEL = {"luna16": "LUNA16", "lidc": "LIDC-IDRI"}

# Hằng số tham chiếu từ bài báo (mean, std) theo %, w1/w2 theo tỉ lệ — SPECS §9.2
PAPER_T5: dict[tuple[str, str], dict[str, Any]] = {
    ("luna16", "caaf_sobel"): {"dice": (88.97, 9.33), "iou": (81.22, 13.07), "sensitivity": (89.63, 11.96),
                               "miss_rate": (10.37, 11.96), "specificity": (99.46, 0.98), "w1": 0.27, "w2": 0.73},
    ("luna16", "caaf_laplacian"): {"dice": (85.37, 22.20), "iou": (78.99, 23.56), "sensitivity": (87.30, 22.13),
                                   "miss_rate": (12.70, 22.13), "specificity": (99.44, 0.90), "w1": 0.36, "w2": 0.64},
    ("luna16", "caaf_hausdorff"): {"dice": (90.88, 6.16), "iou": (83.83, 9.88), "sensitivity": (91.08, 10.20),
                                   "miss_rate": (8.92, 10.20), "specificity": (99.49, 1.06), "w1": 0.26, "w2": 0.74},
    ("lidc", "caaf_sobel"): {"dice": (85.67, 16.96), "iou": (77.93, 20.58), "sensitivity": (91.33, 17.63),
                             "miss_rate": (8.67, 17.63), "specificity": (99.38, 0.99), "w1": 0.24, "w2": 0.76},
    ("lidc", "caaf_laplacian"): {"dice": (85.33, 17.86), "iou": (77.62, 21.07), "sensitivity": (90.31, 18.41),
                                 "miss_rate": (9.69, 18.41), "specificity": (99.31, 1.23), "w1": 0.31, "w2": 0.69},
    ("lidc", "caaf_hausdorff"): {"dice": (85.92, 16.70), "iou": (78.24, 20.37), "sensitivity": (92.38, 16.88),
                                 "miss_rate": (7.62, 16.88), "specificity": (99.28, 1.18), "w1": 0.22, "w2": 0.78},
}
PAPER_T6: dict[tuple[str, str], dict[str, Any]] = {
    ("luna16", "pcam_sobel"): {"dice": (87.46, 7.32), "t": 5.98, "p": 2.74e-9},
    ("luna16", "pcam_laplacian"): {"dice": (84.14, 16.90), "t": 2.38, "p": 0.017},
    ("luna16", "pcam_hausdorff"): {"dice": (88.40, 7.04), "t": 15.91, "p": 6.18e-52},
}
PAPER_T7: dict[tuple[str, str], float] = {
    ("luna16", "baseline_resunetpp"): 83.89,
    ("luna16", "pa_resunet"): 82.79,
    ("luna16", "ca_resunet"): 84.44,
    ("luna16", "pcam_dice"): 85.96,
    ("lidc", "pcam_dice"): 85.37,
}
T7_CONFIGS = ("baseline_resunetpp", "pa_resunet", "ca_resunet", "pcam_dice", "caaf_hausdorff")
BOUNDARY = ("sobel", "laplacian", "hausdorff")
_RUN_RE = re.compile(r"^(?P<config>.+)_s(?P<seed>\d+)$")
DASH = "—"


# ---------------------------------------------------------------------------
# Thu thập kết quả
# ---------------------------------------------------------------------------
def collect(exp_root: str | Path, ckpt: str, split: str = "test") -> dict[tuple[str, str], dict[str, Any]]:
    """`{(dataset, run_name): {"results", "eval_dir"}}` cho mọi run có `eval_<split>_<ckpt>/results.json`."""
    out = {}
    for res in sorted(Path(exp_root).glob(f"*/*/eval_{split}_{ckpt}/results.json")):
        run_dir = res.parent.parent
        out[(run_dir.parent.name, run_dir.name)] = {
            "results": json.loads(res.read_text(encoding="utf-8")),
            "eval_dir": res.parent,
        }
    return out


def parse_run(name: str) -> tuple[str, int, str] | None:
    """`caaf_hausdorff_s42__loss-x` → (`caaf_hausdorff`, 42, `loss-x`); hậu tố rỗng nếu là run chuẩn."""
    base, _, suffix = name.partition("__")
    m = _RUN_RE.match(base)
    return (m["config"], int(m["seed"]), suffix) if m else None


def paper_dice(ds: str, config: str) -> float | None:
    if (ds, config) in PAPER_T5:
        return PAPER_T5[(ds, config)]["dice"][0]
    if (ds, config) in PAPER_T6:
        return PAPER_T6[(ds, config)]["dice"][0]
    return PAPER_T7.get((ds, config))


def _metric(res: dict | None, name: str) -> tuple[float | None, float | None]:
    if res is None:
        return None, None
    if name in ("w1", "w2"):
        return res.get(f"mean_{name}"), None
    m = res.get("metrics", {}).get(name) or {}
    return m.get("mean"), m.get("std")


def _f(v: float | None, nd: int = 2) -> str:
    return DASH if v is None else f"{v:.{nd}f}"


def _ms(mean: float | None, std: float | None, nd: int = 2) -> str:
    if mean is None:
        return DASH
    return f"{mean:.{nd}f}" if std is None else f"{mean:.{nd}f} ± {std:.{nd}f}"


def _delta(ours: float | None, ref: float | None, nd: int = 2) -> str:
    return DASH if ours is None or ref is None else f"{ours - ref:+.{nd}f}"


def _p(v: float | None) -> str:
    return DASH if v is None else f"{v:.3g}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _doc(title: str, intro: str, body: str) -> str:
    return f"# {title}\n\n{intro}\n\n{body}\n"


# ---------------------------------------------------------------------------
# Các bảng
# ---------------------------------------------------------------------------
def table5(found: dict, seed: int) -> str:
    rows = []
    for ds in DATASETS:
        for loss in BOUNDARY:
            cfg = f"caaf_{loss}"
            res = found.get((ds, f"{cfg}_s{seed}"), {}).get("results")
            ref = PAPER_T5[(ds, cfg)]
            for name in (*METRICS, "w1", "w2"):
                mean, std = _metric(res, name)
                nd = 3 if name in ("w1", "w2") else 2
                ref_v = ref[name] if name in ("w1", "w2") else ref[name][0]
                ref_s = _f(ref[name], 2) if name in ("w1", "w2") else _ms(*ref[name])
                rows.append([DS_LABEL[ds], f"`{cfg}`", METRIC_LABEL[name], _ms(mean, std, nd), ref_s, _delta(mean, ref_v, nd)])
    return _table(["Dataset", "Config", "Chỉ số", "Ta (mean ± std)", "Bài báo", "Δ (ta − bài báo)"], rows)


def table6(found: dict, seed: int, stats_dir: Path) -> str:
    rows = []
    for ds in DATASETS:
        for loss in BOUNDARY:
            pcam, caaf = f"pcam_{loss}", f"caaf_{loss}"
            rp, rc = found.get((ds, f"{pcam}_s{seed}")), found.get((ds, f"{caaf}_s{seed}"))
            ref = PAPER_T6.get((ds, pcam), {})
            if rp is None and rc is None and not ref:
                continue
            st = None
            if rp is not None and rc is not None:
                st = paired_test(rc["eval_dir"] / "per_patch_metrics.csv", rp["eval_dir"] / "per_patch_metrics.csv", "dice")
                save_json(st, stats_dir / f"{ds}_{caaf}_s{seed}__vs__{pcam}_s{seed}.json")
            d_mean, d_std = _metric(rp["results"] if rp else None, "dice")
            c_mean, _ = _metric(rc["results"] if rc else None, "dice")
            ref_d = ref.get("dice")
            rows.append([
                DS_LABEL[ds], f"`{caaf}` vs `{pcam}`", _ms(d_mean, d_std), _ms(*ref_d) if ref_d else DASH,
                _delta(d_mean, ref_d[0] if ref_d else None), _f(c_mean),
                _delta(c_mean, d_mean), _f(st["t"] if st else None), _f(ref.get("t")),
                _delta(st["t"] if st else None, ref.get("t")), _p(st["p"] if st else None), _p(ref.get("p")),
            ])
    return _table(
        ["Dataset", "Cặp (A vs B)", "Dice PCAM ta", "Dice PCAM bài báo", "Δ Dice PCAM", "Dice CAAF ta",
         "CAAF − PCAM ta", "t ta", "t bài báo", "Δ t", "p ta", "p bài báo"],
        rows,
    )


def table7(found: dict, seed: int) -> str:
    rows = []
    for ds in DATASETS:
        for cfg in T7_CONFIGS:
            res = found.get((ds, f"{cfg}_s{seed}"), {}).get("results")
            ref = paper_dice(ds, cfg) if (ds, cfg) in PAPER_T7 or cfg == "caaf_hausdorff" else None
            if res is None and ref is None:
                continue
            mean, std = _metric(res, "dice")
            rows.append([DS_LABEL[ds], f"`{cfg}`", _ms(mean, std), _f(ref), _delta(mean, ref)])
    return _table(["Dataset", "Config", "Dice ta (%)", "Dice bài báo (%)", "Δ (ta − bài báo)"], rows)


def seed_variance(found: dict) -> str:
    parts = []
    for ds in DATASETS:
        runs = sorted(
            ((p[1], run) for (d, run) in found if d == ds and (p := parse_run(run)) and p[0] == "caaf_hausdorff" and not p[2]),
        )
        if not runs:
            continue
        ref = paper_dice(ds, "caaf_hausdorff")
        rows, means = [], {k: [] for k in (*METRICS, "w1")}
        for seed, run in runs:
            res = found[(ds, run)]["results"]
            row = [str(seed)]
            for name in (*METRICS, "w1"):
                mean, std = _metric(res, name)
                if mean is not None:
                    means[name].append(mean)
                row.append(_ms(mean, std, 3 if name == "w1" else 2))
            row += [_f(ref), _delta(_metric(res, "dice")[0], ref)]
            rows.append(row)
        summary = ["**mean ± std giữa các seed**"]
        for name in (*METRICS, "w1"):
            vals = np.asarray(means[name], dtype=float)
            summary.append(_ms(float(vals.mean()) if vals.size else None,
                               float(vals.std(ddof=1)) if vals.size > 1 else None, 3 if name == "w1" else 2))
        dice_mean = float(np.mean(means["dice"])) if means["dice"] else None
        summary += [_f(ref), _delta(dice_mean, ref)]
        rows.append(summary)
        parts.append(f"## {DS_LABEL[ds]} — `caaf_hausdorff` ({len(runs)} seed)\n\n" + _table(
            ["Seed", "Dice (%)", "IoU (%)", "Sens. (%)", "Miss (%)", "Spec. (%)", "w1", "Dice bài báo", "Δ Dice"], rows))
    return "\n\n".join(parts) if parts else "_Chưa có run `caaf_hausdorff_s*` nào._"


def sensitivity(found: dict, found_last: dict, found_best: dict) -> str:
    rows = []
    for (ds, run), item in sorted(found.items()):
        p = parse_run(run)
        if not p or not p[2]:
            continue
        config, seed, suffix = p
        base = found.get((ds, f"{config}_s{seed}"))
        mean, std = _metric(item["results"], "dice")
        base_mean = _metric(base["results"] if base else None, "dice")[0]
        ref = paper_dice(ds, config)
        rows.append([DS_LABEL.get(ds, ds), f"`{run}`", f"`{config}_s{seed}`", f"`{suffix}`", _ms(mean, std),
                     _f(base_mean), _delta(mean, base_mean), _f(ref), _delta(mean, ref)])
    t1 = _table(["Dataset", "Run", "Run chuẩn", "Hậu tố", "Dice ta (%)", "Dice run chuẩn", "Δ so với run chuẩn",
                 "Dice bài báo", "Δ (ta − bài báo)"], rows) if rows else "_Chưa có run nào có hậu tố `__*`._"
    rows2 = []
    for key in sorted(set(found_last) & set(found_best)):
        ds, run = key
        last = _metric(found_last[key]["results"], "dice")[0]
        best = _metric(found_best[key]["results"], "dice")[0]
        p = parse_run(run)
        ref = paper_dice(ds, p[0]) if p else None
        rows2.append([DS_LABEL.get(ds, ds), f"`{run}`", _f(last), _f(best), _delta(best, last), _f(ref),
                      _delta(last, ref), _delta(best, ref)])
    t2 = _table(["Dataset", "Run", "Dice last", "Dice best", "Δ (best − last)", "Dice bài báo", "Δ last − bài báo",
                 "Δ best − bài báo"], rows2) if rows2 else "_Chưa có run nào được đánh giá cả `last` và `best`._"
    return f"## Run thử nghiệm độ nhạy (hậu tố `__*`, S1/S3/S4/S5)\n\n{t1}\n\n## S2 — checkpoint `best` vs `last`\n\n{t2}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-root", required=True, help="Thư mục experiments/ (chứa <ds>/<run>/)")
    parser.add_argument("--out-dir", default="reports", help="Nơi ghi các bảng .md")
    parser.add_argument("--ckpt", choices=("last", "best"), default="last", help="Checkpoint dùng cho bảng chính")
    parser.add_argument("--seed", type=int, default=42, help="Seed của run chuẩn {config}_s{seed}")
    parser.add_argument("--include-sensitivity", action="store_true", help="Sinh thêm sensitivity.md (S1–S5)")
    return parser


def generate(exp_root: str | Path, out_dir: str | Path, ckpt: str = "last", seed: int = 42,
             include_sensitivity: bool = False) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = collect(exp_root, ckpt)
    main_runs = {k: v for k, v in found.items() if "__" not in k[1]}
    stamp = time.strftime("%Y-%m-%d %H:%M")
    intro = (f"Sinh tự động bởi `src.report` lúc {stamp} từ `{Path(exp_root).as_posix()}` — checkpoint `{ckpt}`, "
             f"run chuẩn `{{config}}_s{seed}` (bỏ qua hậu tố `__*`). Đơn vị %, std `ddof=1`; "
             f"Δ = ta − bài báo; `{DASH}` = chưa có run hoặc bài báo không công bố.")
    files = {
        "compare_table5.md": _doc("Table 5 — CAAF-ResUNet với 3 boundary loss", intro, table5(main_runs, seed)),
        "compare_table6.md": _doc(
            "Table 6 — PCAM tĩnh vs CAAF (paired t-test hai phía)",
            intro + " t-test: `d_i = Dice_CAAF − Dice_PCAM` trên cùng tập test (t > 0 ⇔ CAAF tốt hơn).",
            table6(main_runs, seed, out_dir / "stats"),
        ),
        "compare_table7.md": _doc("Table 7 — Ablation khối chú ý (Dice loss)", intro + " Dòng `caaf_hausdorff` lấy số "
                                  "bài báo từ Table 5.", table7(main_runs, seed)),
        "seed_variance.md": _doc("Dao động giữa các seed — `caaf_hausdorff`", intro, seed_variance(main_runs)),
    }
    if include_sensitivity:
        found_last = found if ckpt == "last" else collect(exp_root, "last")
        found_best = found if ckpt == "best" else collect(exp_root, "best")
        files["sensitivity.md"] = _doc("Thử nghiệm độ nhạy (Phase 8b)", intro.replace("(bỏ qua hậu tố `__*`)", ""),
                                       sensitivity(found, found_last, found_best))
    paths = {}
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        paths[name] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate(args.exp_root, args.out_dir, args.ckpt, args.seed, args.include_sensitivity)
    for name, path in paths.items():
        print(f"đã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
