"""Hợp đồng Notebook ↔ CLI ↔ SPECS Phụ lục A/B (không cần torch).

- 5 notebook là JSON / nbformat hợp lệ, cell đầu tiên là code cell tham số, không còn chuỗi cũ.
- Mọi cờ `--...` notebook lắp vào lệnh `python -m src.<script>` đều có thật trong `build_parser()`.
- Mỗi script ở Phụ lục A có đúng số lượng & tên tham số; 100% tham số được lộ ra ở cell 1 và truyền vào lệnh.
- Không có token / secret hardcode trong notebook hay mã nguồn.
Cờ CLI được đọc tĩnh từ mã nguồn `build_parser()` (AST) nên không phải import `src.train` (cần torch).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import utils  # noqa: E402

NB_DIR = ROOT / "notebooks"
NOTEBOOKS = ["00_colab_data_survey.ipynb", "01_colab_preprocess.ipynb", "10_colab_train.ipynb",
             "11_colab_evaluate.ipynb", "12_colab_stats_figures.ipynb"]
SCRIPTS = {
    "src.data.build_index": "src/data/build_index.py",
    "src.data.preprocess_lidc": "src/data/preprocess_lidc.py",
    "src.data.preprocess_luna16": "src/data/preprocess_luna16.py",
    "src.data.make_splits": "src/data/make_splits.py",
    "src.data.pack_h5": "src/data/pack_h5.py",
    "src.train": "src/train.py",
    "src.evaluate": "src/evaluate.py",
    "src.stats_test": "src/stats_test.py",
    "src.report": "src/report.py",
    "src.visualize": "src/visualize.py",
}
# Tham số CLI → biến ở cell 1 của notebook tương ứng (hợp đồng Phụ lục B)
EXPOSURE = {
    "00_colab_data_survey.ipynb": {"src.data.build_index": {
        "--lidc-dir": "RAW_LIDC_DIR", "--luna16-dir": "RAW_LUNA16_DIR", "--out-dir": "INDEX_OUT_DIR", "--workers": "WORKERS"}},
    "01_colab_preprocess.ipynb": {
        "src.data.preprocess_lidc": {"--raw-dir": "RAW_LIDC_DIR", "--index-dir": "INDEX_DIR", "--out-dir": "INTERIM_DIR",
                                     "--config": "DATA_CFG", "--workers": "WORKERS", "--limit": "LIMIT_SCANS"},
        "src.data.preprocess_luna16": {"--raw-dir": "RAW_LUNA16_DIR", "--lidc-dir": "RAW_LIDC_DIR",
                                       "--index-dir": "INDEX_DIR", "--out-dir": "INTERIM_DIR", "--config": "DATA_CFG",
                                       "--workers": "WORKERS", "--limit": "LIMIT_SCANS"},
        "src.data.make_splits": {"--index": "INTERIM_DIR", "--out": "SPLITS_OUT", "--config": "DATA_CFG",
                                 "--mode": "SPLIT_MODE"},
        "src.data.pack_h5": {"--interim-dir": "INTERIM_DIR", "--splits": "SPLITS_OUT", "--out-dir": "LOCAL_PACK_TMP",
                             "--config": "DATA_CFG", "--copy-to-drive-dir": "DRIVE_PROCESSED_OUT"},
    },
    "10_colab_train.ipynb": {"src.train": {
        "--config": "CONFIG", "--dataset": "DATASET", "--data-dir": "LOCAL_DATA_DIR", "--out-dir": "OUT_DIR",
        "--set": "EXTRA_SET", "--resume": "RESUME", "--run-name": "RUN_NAME", "--max-batches": "MAX_BATCHES",
        "--overfit-steps": "OVERFIT_STEPS", "--model": "MODEL", "--aaf-mode": "AAF_MODE", "--loss": "LOSS",
        "--lambda": "LAMBDA", "--epochs": "EPOCHS", "--batch-size": "BATCH_SIZE", "--lr": "LR",
        "--weight-decay": "WEIGHT_DECAY", "--amp": "AMP", "--augment": "AUGMENT", "--seed": "SEED"}},
    "11_colab_evaluate.ipynb": {"src.evaluate": {
        "--run-dir": "RUN_DIRS", "--data-dir": "LOCAL_DATA_DIR", "--dataset": "DATASET_OVERRIDE", "--ckpt": "CKPTS",
        "--split": "SPLIT", "--batch-size": "BATCH_SIZE", "--threshold": "THRESHOLD", "--save-preds": "SAVE_PREDS"}},
    "12_colab_stats_figures.ipynb": {
        "src.report": {"--exp-root": "EXP_ROOT", "--out-dir": "OUT_REPORTS_DIR", "--ckpt": "REPORT_CKPT",
                       "--seed": "REPORT_SEED", "--include-sensitivity": "INCLUDE_SENSITIVITY"},
        "src.stats_test": {"--a": "STATS_PAIRS", "--b": "STATS_PAIRS", "--metric": "STATS_METRIC", "--out": "STATS_OUT_DIR"},
        "src.visualize": {"--run-dirs": "VIZ_RUN_DIRS", "--kind": "VIZ_KINDS", "--out-dir": "OUT_REPORTS_DIR",
                          "--data-dir": "LOCAL_DATA_DIR", "--ckpt": "VIZ_CKPT", "--split": "VIZ_SPLIT",
                          "--patch-ids": "PATCH_IDS", "--layer": "GRADCAM_LAYERS", "--dataset": "VIZ_DATASET_OVERRIDE"},
    },
}
SPEC = (ROOT / "docs" / "PLANNING" / "SPECS.md").read_text(encoding="utf-8")
HELPERS = {"src.train": ("train_args(",)}  # hàm trong notebook dựng danh sách tham số cho `python -m src.train`


# ---------------------------------------------------------------------------
# Trích xuất
# ---------------------------------------------------------------------------
def parser_actions(module: str) -> list[list[str]]:
    """Danh sách option_strings của từng action trong `build_parser()` (đọc AST, không import script)."""
    tree = ast.parse((ROOT / SCRIPTS[module]).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser")
    calls = sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
    actions: list[list[str]] = []
    for node in calls:
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name == "add_argument":
            opts = [a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")]
            boolean = any(isinstance(k.value, ast.Attribute) and k.value.attr == "BooleanOptionalAction"
                          for k in node.keywords if k.arg == "action")
            actions.append(opts + ([f"--no-{opts[0][2:]}"] if boolean else []))
        elif name in ("add_common_args", "add_train_shortcuts"):
            p = getattr(utils, name)(argparse.ArgumentParser(add_help=False))
            actions += [list(a.option_strings) for a in p._actions]
    return actions


def appendix_a() -> dict[str, tuple[set[str], int]]:
    section = SPEC.split("## Phụ lục A", 1)[1].split("\n## ", 1)[0]
    out = {}
    for line in section.splitlines():
        m = re.match(r"\|\s*P\d+\s*\|\s*`(src\.[\w.]+)`\s*\|(.*)\|\s*(\d+)\s*\|\s*$", line)
        if not m:
            continue
        flags = set(re.findall(r"`(--[a-z0-9-]+)`", m[2]))
        if "cờ tắt" in m[2]:
            flags |= {s for a in utils.add_train_shortcuts(argparse.ArgumentParser(add_help=False))._actions
                      for s in a.option_strings}
        out[m[1]] = (flags, int(m[3]))
    return out


def load_nb(name: str) -> dict:
    return json.loads((NB_DIR / name).read_text(encoding="utf-8"))


def code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell1_vars(src: str) -> set[str]:
    return {t.id for n in ast.parse(src).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}


def flags_in(src: str) -> set[str]:
    return set(re.findall(r"""["'](--[a-z][a-z0-9-]*)["']""", src))


ALL_FLAGS = {m: {s for a in parser_actions(m) for s in a} for m in SCRIPTS}


# ---------------------------------------------------------------------------
# Kiểm thử
# ---------------------------------------------------------------------------
def test_notebook_set():
    assert sorted(p.name for p in NB_DIR.iterdir() if p.name != "desktop.ini") == NOTEBOOKS


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_valid_json_first_cell_is_parameters(name):
    nb = load_nb(name)
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 4
    first = nb["cells"][0]
    assert first["cell_type"] == "code", "cell đầu tiên phải là cell tham số"
    src = "".join(first["source"])
    assert {"REPO_URL", "BRANCH", "DRIVE_ROOT"} <= cell1_vars(src)
    assert "import " not in src and "run(" not in src, "cell tham số chỉ gán biến"
    for code in code_cells(nb):
        compile(code, name, "exec")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_legacy_strings(name):
    text = (NB_DIR / name).read_text(encoding="utf-8")
    for legacy in ("DRIVE_CT208E_ROOT", "LIDC_IDRI_raw", "LUNA16_Extracted", "pylidc", "SOURCES/"):
        assert legacy not in text, legacy


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_notebook_flag_exists_in_parser(name):
    for code in code_cells(load_nb(name))[1:]:
        used = flags_in("\n".join(ln for ln in code.splitlines() if "git" not in ln))  # bỏ cờ của lệnh git (setup)
        if not used:
            continue
        modules = [m for m in SCRIPTS if f'"{m}"' in code or any(h in code for h in HELPERS.get(m, ()))]
        assert modules, f"cell dùng cờ {sorted(used)} nhưng không gọi script nào"
        allowed = set().union(*(ALL_FLAGS[m] for m in modules))
        assert used <= allowed, f"{name}: cờ không tồn tại trong {modules}: {sorted(used - allowed)}"


def test_appendix_a_matches_parsers():
    spec = appendix_a()
    assert set(spec) == set(SCRIPTS)
    for module, (flags, count) in spec.items():
        actions = parser_actions(module)
        assert len(actions) == count, f"{module}: {len(actions)} tham số, Phụ lục A ghi {count}"
        assert {s for a in actions for s in a} == flags, module


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_all_cli_args_exposed_in_cell1(name):
    codes = code_cells(load_nb(name))
    params = cell1_vars(codes[0])
    for module, mapping in EXPOSURE[name].items():
        for action in parser_actions(module):
            flag = action[0]
            var = mapping.get(flag)
            assert var is not None, f"{module} {flag} chưa có biến ở cell 1"
            assert var in params, f"{var} ({flag}) không có ở cell 1"
            assert any(any(f'"{opt}"' in c for opt in action) and var in c for c in codes[1:]), \
                f"{flag} không được truyền từ {var}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_cell1_keeps_appendix_b_lines(name):
    title = f"`{name}`"
    block = SPEC.split("### Cell tham số đầy đủ", 1)[1]
    body = block[block.index(title):].split("```python\n", 1)[1].split("```", 1)[0]
    want = [ln for ln in body.splitlines() if ln.strip()]
    have = iter(ln for ln in code_cells(load_nb(name))[0].splitlines() if ln.strip())
    assert all(any(w == h for h in have) for w in want), "cell 1 phải chứa nguyên văn (theo thứ tự) các dòng Phụ lục B"


SECRET_PATTERNS = [
    r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"gho_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"\bsk-[A-Za-z0-9]{20,}", r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"""(?i)(token|secret|password|api_key)\s*=\s*["'][A-Za-z0-9_\-]{16,}["']""",
    r"https://[^/\s\"']+:[^@\s\"']+@github\.com",
]


def scanned_files() -> list[Path]:
    files = [*NB_DIR.glob("*.ipynb"), *ROOT.glob("src/**/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.yaml"),
             *ROOT.glob("requirements-*.txt"), ROOT / ".gitignore"]
    return [f for f in files if f.name != Path(__file__).name]


def test_no_hardcoded_secrets():
    for f in scanned_files():
        text = f.read_text(encoding="utf-8")
        for pat in SECRET_PATTERNS:
            assert not re.search(pat, text), f"{f.relative_to(ROOT)} khớp mẫu secret {pat}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_token_only_via_colab_secrets(name):
    src = "\n".join(code_cells(load_nb(name)))
    assert 'userdata.get("GH_TOKEN")' in src and "GIT_ASKPASS" in src
    assert not re.search(r"print\([^)]*token", src, re.I)
    assert "{token}" not in src and "echo $GH_TOKEN" not in src  # không nhét token vào URL / log


def test_no_pylidc_or_sources_imports():
    pattern = re.compile(r"^\s*(import pylidc|from pylidc|import SOURCES|from SOURCES|.*sys\.path.*SOURCES)", re.M)
    for f in [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/*.py"), *NB_DIR.glob("*.ipynb"), *ROOT.glob("requirements-*.txt")]:
        assert not pattern.search(f.read_text(encoding="utf-8")), f.relative_to(ROOT)
    for req in ROOT.glob("requirements-*.txt"):
        pkgs = [ln.split("#", 1)[0].strip().lower() for ln in req.read_text(encoding="utf-8").splitlines()]
        assert not any(p.startswith(("pylidc", "sqlalchemy")) for p in pkgs if p), req.name
