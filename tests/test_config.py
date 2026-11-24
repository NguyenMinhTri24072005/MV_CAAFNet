"""Kiểm thử Phase 0 và Phase 2 — SPECS §0.1–0.3, §2.1–2.2, §6.1.1–6.1.2, Phụ lục A và C.

Chạy ở máy local, không cần torch (checkpoint được kiểm bằng một module `torch` giả):
    pytest tests/test_config.py -v
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import random
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import utils  # noqa: E402
from src.utils import (  # noqa: E402
    add_common_args,
    add_train_shortcuts,
    apply_overrides,
    deep_merge,
    get_by_dotted,
    load_checkpoint,
    load_config,
    make_run_name,
    parse_set_value,
    save_checkpoint,
    save_json,
    save_yaml,
)

CONFIG_DIR = ROOT / "configs"
BASE_YAML = CONFIG_DIR / "base.yaml"
DATA_YAML = CONFIG_DIR / "data.yaml"
CAAF = CONFIG_DIR / "caaf_hausdorff.yaml"

# SPECS §0.3.3: tệp → (model.name, aaf.mode ghi trong tệp hoặc None, loss.name)
EXPERIMENTS = {
    "baseline_resunetpp": ("resunetpp", None, "dice"),
    "pa_resunet": ("caaf_resunet", "pa", "dice"),
    "ca_resunet": ("caaf_resunet", "ca", "dice"),
    "pcam_dice": ("caaf_resunet", "static", "dice"),
    "pcam_sobel": ("caaf_resunet", "static", "dice+sobel"),
    "pcam_laplacian": ("caaf_resunet", "static", "dice+laplacian"),
    "pcam_hausdorff": ("caaf_resunet", "static", "dice+hausdorff"),
    "caaf_sobel": ("caaf_resunet", "adaptive", "dice+sobel"),
    "caaf_laplacian": ("caaf_resunet", "adaptive", "dice+laplacian"),
    "caaf_hausdorff": ("caaf_resunet", "adaptive", "dice+hausdorff"),
}
IDENTITY_KEYS = {"model.name", "aaf.mode", "loss.name"}
REQUIRED_PACKAGES = {
    "numpy", "scipy", "pandas", "h5py", "pyyaml", "pydicom",
    "simpleitk", "scikit-image", "matplotlib", "tqdm", "pytest",
}
SHORTCUT_FLAGS = {
    "--model", "--aaf-mode", "--loss", "--lambda", "--epochs", "--batch-size", "--lr",
    "--weight-decay", "--amp", "--no-amp", "--augment", "--no-augment", "--seed",
}
HASH_RE = re.compile(r"__(?:.*_)?set-[0-9a-f]{6}$")


# ---------------------------------------------------------------------------
# Tiện ích cho test
# ---------------------------------------------------------------------------
def make_train_parser() -> argparse.ArgumentParser:
    """Parser giống `src.train` (Phụ lục A): 5 tham số chung + 4 điều khiển + 11 cờ tắt."""
    parser = argparse.ArgumentParser(prog="src.train")
    add_common_args(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=None)
    add_train_shortcuts(parser)
    return parser


def resolve(*extra: str, config: Path | str = CAAF, dataset: str = "luna16"):
    args = make_train_parser().parse_args(["--config", str(config), "--dataset", dataset, *extra])
    config_only = load_config(args.config)
    resolved = apply_overrides(config_only, args)
    name, overrides = make_run_name(args.config, config_only, resolved, args)
    return args, config_only, resolved, name, overrides


def run_name(*extra: str, **kw) -> str:
    return resolve(*extra, **kw)[3]


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, f"{full}."))
        else:
            out[full] = value
    return out


def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def requirement_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


# ---------------------------------------------------------------------------
# Phase 0 — môi trường và cấu trúc repo (§0.1, §0.2)
# ---------------------------------------------------------------------------
class TestPhase0Repo:
    def test_src_package_and_notebooks_dir(self):
        assert (ROOT / "src" / "__init__.py").is_file()
        assert (ROOT / "notebooks").is_dir()

    @pytest.mark.parametrize("name", ["requirements-local.txt", "requirements-colab.txt"])
    def test_requirements_follow_spec(self, name):
        lines = requirement_names(ROOT / name)
        lowered = {re.split(r"[<>=!~\[; ]", line, 1)[0].lower() for line in lines}
        assert lowered == REQUIRED_PACKAGES
        for banned in ("pylidc", "sqlalchemy", "torch", "torchvision"):
            assert banned not in lowered
        assert not any(re.search(r"numpy\s*<", line, re.I) for line in lines)

    def test_gitignore_entries(self):
        lines = {line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
        required = {
            ".venv/", "desktop.ini", "/data/", "/experiments/", "*.pt", "*.h5", "*.npz",
            ".ipynb_checkpoints/", "__pycache__/",
        }
        assert required <= lines

    def test_utils_import_does_not_pull_torch_or_numpy(self):
        code = "import sys, src.utils; print('torch' in sys.modules, 'numpy' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
        )
        assert out.stdout.split() == ["False", "False"]


# ---------------------------------------------------------------------------
# YAML và --set
# ---------------------------------------------------------------------------
class TestYamlParsing:
    @pytest.mark.parametrize(
        "raw, expected, typ",
        [
            ("8", 8, int),
            ("1e-4", 1e-4, float),
            ("1.0e-4", 1e-4, float),
            ("-2.5E3", -2500.0, float),
            ("5e+2", 500.0, float),
            ("0.5", 0.5, float),
            ("[1,2]", [1, 2], list),
            ("[6, 12, 18]", [6, 12, 18], list),
            ("true", True, bool),
            ("false", False, bool),
            ("null", None, type(None)),
            ("dice+sobel", "dice+sobel", str),
            ("'8'", "8", str),
            ("{a: 1}", {"a": 1}, dict),
        ],
    )
    def test_parse_set_value_types(self, raw, expected, typ):
        value = parse_set_value(raw)
        assert value == expected
        assert type(value) is typ

    def test_parse_set_value_invalid_yaml(self):
        with pytest.raises(ValueError):
            parse_set_value("[1, 2")

    def test_duplicate_key_rejected(self, tmp_path):
        path = tmp_path / "dup.yaml"
        path.write_text("train:\n  epochs: 10\ntrain:\n  epochs: 20\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_scientific_float_in_file(self, tmp_path):
        path = write_yaml(tmp_path / "c.yaml", {})
        path.write_text("lr: 1e-4\nwd: 5E-3\n", encoding="utf-8")
        cfg = load_config(path)
        assert cfg == {"lr": 1e-4, "wd": 5e-3}
        assert isinstance(cfg["lr"], float) and isinstance(cfg["wd"], float)


# ---------------------------------------------------------------------------
# load_config: kế thừa base, deep-merge, --set (§0.3.1, §2.1)
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_multilevel_merge_with_relative_bases(self, tmp_path):
        write_yaml(tmp_path / "grand.yaml", {
            "a": 1, "keep": "grand",
            "nested": {"x": 1, "y": [1, 2, 3], "deep": {"k": 1}},
        })
        write_yaml(tmp_path / "mid" / "mid.yaml", {
            "base": "../grand.yaml", "mid_only": True,
            "nested": {"x": 2, "deep": {"k2": 2}},
        })
        leaf = write_yaml(tmp_path / "mid" / "sub" / "leaf.yaml", {
            "base": "../mid.yaml", "a": 3, "nested": {"y": [9]},
        })
        cfg = load_config(leaf)
        assert cfg == {
            "a": 3, "keep": "grand", "mid_only": True,
            "nested": {"x": 2, "y": [9], "deep": {"k": 1, "k2": 2}},
        }
        assert "base" not in cfg

    def test_absolute_base_path(self, tmp_path):
        child = write_yaml(tmp_path / "child.yaml", {"base": str(BASE_YAML), "train": {"epochs": 3}})
        cfg = load_config(child)
        assert cfg["train"]["epochs"] == 3
        assert cfg["train"]["batch_size"] == load_config(BASE_YAML)["train"]["batch_size"]

    def test_deep_merge_does_not_mutate_inputs(self):
        parent = {"a": {"b": 1, "l": [1]}, "c": 1}
        child = {"a": {"b": 2}, "d": {"e": 1}}
        merged = deep_merge(parent, child)
        merged["a"]["l"].append(2)
        merged["d"]["e"] = 99
        assert parent == {"a": {"b": 1, "l": [1]}, "c": 1}
        assert child == {"a": {"b": 2}, "d": {"e": 1}}
        assert merged["a"]["b"] == 2

    def test_dict_replaced_by_scalar_and_vice_versa(self):
        assert deep_merge({"a": {"b": 1}}, {"a": None}) == {"a": None}
        assert deep_merge({"a": 1}, {"a": {"b": 1}}) == {"a": {"b": 1}}

    def test_base_cycle_raises(self, tmp_path):
        write_yaml(tmp_path / "a.yaml", {"base": "b.yaml", "x": 1})
        write_yaml(tmp_path / "b.yaml", {"base": "a.yaml", "y": 1})
        with pytest.raises(ValueError, match="Vòng lặp"):
            load_config(tmp_path / "a.yaml")

    def test_missing_file_and_missing_base(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")
        child = write_yaml(tmp_path / "child.yaml", {"base": "nope.yaml"})
        with pytest.raises(FileNotFoundError):
            load_config(child)

    def test_non_mapping_root_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(TypeError):
            load_config(path)

    def test_set_override_types(self):
        cfg = load_config(CAAF, [
            "train.batch_size=8",
            "train.optimizer.lr=1e-4",
            "model.aspp.rates=[1,2]",
            "model.aspp.gap=true",
            "train.grad_clip=null",
            "loss.name=dice+sobel",
        ])
        assert cfg["train"]["batch_size"] == 8 and type(cfg["train"]["batch_size"]) is int
        assert cfg["train"]["optimizer"]["lr"] == 1e-4 and type(cfg["train"]["optimizer"]["lr"]) is float
        assert cfg["model"]["aspp"]["rates"] == [1, 2]
        assert cfg["model"]["aspp"]["gap"] is True
        assert cfg["train"]["grad_clip"] is None
        assert cfg["loss"]["name"] == "dice+sobel"

    def test_set_later_item_wins(self):
        cfg = load_config(CAAF, ["train.batch_size=8", "train.batch_size=4"])
        assert cfg["train"]["batch_size"] == 4

    @pytest.mark.parametrize(
        "item",
        [
            "train.batchsize=8",       # gõ sai khóa lá
            "trian.batch_size=8",      # gõ sai khóa gốc
            "model.aspp.gap.x=1",      # đi xuống một giá trị không phải mapping
            "model.attention.sigmoid=true",
            "base=base.yaml",          # `base` đã bị xóa sau khi merge
        ],
    )
    def test_set_unknown_key_raises_keyerror(self, item):
        with pytest.raises(KeyError):
            load_config(CAAF, [item])

    @pytest.mark.parametrize("item", ["train.batch_size", "=3", "a..b=1", ".a=1", "a.=1"])
    def test_set_malformed_item_raises_valueerror(self, item):
        with pytest.raises(ValueError):
            load_config(CAAF, [item])


# ---------------------------------------------------------------------------
# Các tệp config thật (§0.3.2–0.3.4)
# ---------------------------------------------------------------------------
class TestConfigFiles:
    def test_exactly_ten_experiment_configs(self):
        stems = {p.stem for p in CONFIG_DIR.glob("*.yaml")} - {"base", "data"}
        assert stems == set(EXPERIMENTS)

    def test_experiments_are_distinct(self):
        combos = {
            (cfg["model"]["name"], cfg["aaf"]["mode"], cfg["loss"]["name"])
            for cfg in (load_config(CONFIG_DIR / f"{stem}.yaml") for stem in EXPERIMENTS)
        }
        assert len(combos) == len(EXPERIMENTS)

    def test_base_yaml_values(self):
        cfg = load_config(BASE_YAML)
        assert cfg["seed"] == 42 and cfg["dataset"] is None
        model = cfg["model"]
        assert model["name"] == "caaf_resunet" and model["in_channels"] == 1
        assert model["channels"] == [32, 64, 128, 256]
        assert model["upsample"] == "convtranspose"
        assert model["aac"] == {"pool_size": 8, "qkv_bias": False}
        assert model["attention"] == {"pam_reduction": 8, "att_sigmoid": False}
        assert model["aspp"] == {"rates": [6, 12, 18], "gap": False}
        assert model["se_reduction"] == 8
        assert cfg["aaf"] == {"mode": "adaptive", "static_weights": [0.5, 0.5]}
        loss = cfg["loss"]
        assert loss["name"] == "dice" and loss["lambda"] == 1.0
        assert loss["hausdorff_mode"] == "soft_pg_detach_gp"
        assert loss["hausdorff_empty_value"] == pytest.approx(64 * 2 ** 0.5, abs=0.01)
        for key in ("lambda", "dice_eps", "sobel_eps", "hausdorff_empty_value"):
            assert type(loss[key]) is float, key
        assert loss["dice_eps"] == 1e-6 and loss["sobel_eps"] == 1e-8
        assert cfg["data"] == {"patch_size": 64, "augment": False, "num_workers": 2}
        train = cfg["train"]
        assert train["epochs"] == 100 and train["batch_size"] == 16
        assert train["optimizer"] == {
            "name": "adamw", "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999],
        }
        assert type(train["optimizer"]["lr"]) is float
        assert train["scheduler"] == {"name": "steplr", "step_size": 30, "gamma": 0.1}
        assert train["amp"] is False and train["grad_clip"] is None and train["deterministic"] is False
        assert cfg["eval"] == {"threshold": 0.5, "ckpt": "last"}

    @pytest.mark.parametrize("stem", sorted(EXPERIMENTS))
    def test_raw_experiment_file(self, stem):
        raw = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
        model, aaf, loss = EXPERIMENTS[stem]
        assert raw["base"] == "base.yaml"
        assert isinstance(raw["loss"], dict), "loss phải là mapping {name: ...}"
        assert raw["loss"] == {"name": loss}
        assert raw["model"] == {"name": model}
        if aaf is None:
            assert "aaf" not in raw
        else:
            assert raw["aaf"] == {"mode": aaf}
        assert set(raw) <= {"base", "model", "aaf", "loss"}

    @pytest.mark.parametrize("stem", sorted(EXPERIMENTS))
    def test_experiment_config_loads(self, stem):
        base = load_config(BASE_YAML)
        cfg = load_config(CONFIG_DIR / f"{stem}.yaml")
        model, aaf, loss = EXPERIMENTS[stem]
        assert "base" not in cfg
        assert cfg["model"]["name"] == model
        assert cfg["aaf"]["mode"] == (aaf if aaf is not None else base["aaf"]["mode"])
        assert cfg["loss"]["name"] == loss
        assert cfg["dataset"] is None
        assert cfg["model"]["attention"]["att_sigmoid"] is False
        assert cfg["loss"]["hausdorff_mode"] == "soft_pg_detach_gp"
        flat, flat_base = flatten(cfg), flatten(base)
        assert set(flat) == set(flat_base), "config con không được thêm khóa lạ"
        differing = {k for k in flat if flat[k] != flat_base[k]}
        assert differing <= IDENTITY_KEYS

    @pytest.mark.parametrize("stem", sorted(EXPERIMENTS))
    def test_default_run_name_per_config(self, stem):
        _, _, resolved, name, overrides = resolve(config=CONFIG_DIR / f"{stem}.yaml")
        assert name == f"{stem}_s42"
        assert overrides["tokens"] == [] and overrides["set_hash"] is None
        assert resolved["dataset"] == "luna16"

    def test_data_yaml(self):
        cfg = load_config(DATA_YAML)
        assert set(cfg) == {"paths", "common", "lidc", "luna16", "split"}
        assert cfg["paths"]["lidc_raw"].endswith("MV_CAAFNet/data/raw/LIDC-IDRI")
        assert cfg["paths"]["luna16_raw"].endswith("MV_CAAFNet/data/raw/LUNA16")
        assert cfg["paths"]["index_dir"].endswith("MV_CAAFNet/data/index")
        common = cfg["common"]
        assert common["patch_size"] == load_config(BASE_YAML)["data"]["patch_size"] == 64
        assert common["hu_floor"] == -1000 and common["pad_value_hu"] == -1000
        lo, hi = common["hu_window"]
        assert lo == -1000 and lo < hi
        assert common["resample_mm"] is None and common["resample_order"] == 1
        assert common["min_mask_area_px"] >= 1
        assert common["include_other_nodules"] is True
        assert common["center_jitter_px"] == 0 and common["workers"] >= 1
        assert cfg["lidc"] == {"clevel": 0.5, "min_readers": 1, "min_diameter_mm": 3.0}
        assert cfg["luna16"] == {"annotations_csv": "annotations.csv", "match_tolerance": "radius", "clevel": 0.5}
        assert cfg["split"] == {"mode": "patient", "seed": 2024, "test_patches": 1200, "val_ratio": 0.1}
        assert not any("pylidc" in key.lower() for key in flatten(cfg))


# ---------------------------------------------------------------------------
# CLI: add_common_args, add_train_shortcuts (Phụ lục A, §6.1.1)
# ---------------------------------------------------------------------------
class TestCli:
    def test_common_args_defaults(self):
        parser = add_common_args(argparse.ArgumentParser())
        args = parser.parse_args(["--config", "x.yaml", "--dataset", "lidc"])
        assert (args.config, args.dataset) == ("x.yaml", "lidc")
        assert args.data_dir == "data/processed" and args.out_dir == "experiments"
        assert args.set == []

    @pytest.mark.parametrize(
        "argv",
        [
            ["--dataset", "luna16"],
            ["--config", "x.yaml"],
            ["--config", "x.yaml", "--dataset", "lidc-idri"],
        ],
    )
    def test_common_args_required_and_choices(self, argv):
        parser = add_common_args(argparse.ArgumentParser())
        with pytest.raises(SystemExit):
            parser.parse_args(argv)

    def test_set_accumulates_in_order(self):
        parser = add_common_args(argparse.ArgumentParser())
        argv = ["--config", "x", "--dataset", "luna16", "--set", "a=1", "--set", "b.c=2"]
        assert parser.parse_args(argv).set == ["a=1", "b.c=2"]
        assert parser.parse_args(argv[:4]).set == []

    def test_argument_counts_match_appendix_a(self):
        common = add_common_args(argparse.ArgumentParser(add_help=False))
        assert len(common._actions) == 5
        shortcuts = add_train_shortcuts(argparse.ArgumentParser(add_help=False))
        assert len(shortcuts._actions) == 11
        flags = {s for action in shortcuts._actions for s in action.option_strings}
        assert flags == SHORTCUT_FLAGS
        train = make_train_parser()
        assert len([a for a in train._actions if not isinstance(a, argparse._HelpAction)]) == 20

    def test_shortcut_dests_match_table(self):
        parser = add_train_shortcuts(argparse.ArgumentParser(add_help=False))
        assert [a.dest for a in parser._actions] == [dest for dest, _ in utils.TRAIN_SHORTCUTS]

    def test_shortcuts_default_none(self):
        args = make_train_parser().parse_args(["--config", "x", "--dataset", "luna16"])
        for dest, _ in utils.TRAIN_SHORTCUTS:
            assert getattr(args, dest) is None, dest

    @pytest.mark.parametrize(
        "argv, dest, expected",
        [
            (["--amp"], "amp", True),
            (["--no-amp"], "amp", False),
            (["--augment"], "augment", True),
            (["--no-augment"], "augment", False),
        ],
    )
    def test_boolean_shortcuts(self, argv, dest, expected):
        args = make_train_parser().parse_args(["--config", "x", "--dataset", "luna16", *argv])
        assert getattr(args, dest) is expected

    @pytest.mark.parametrize(
        "argv",
        [
            ["--loss", "dice+focal"],
            ["--model", "unet"],
            ["--aaf-mode", "none"],
            ["--epochs", "0"],
            ["--batch-size", "-1"],
            ["--lr", "0"],
            ["--lr", "nan"],
            ["--weight-decay", "-0.1"],
            ["--lambda", "inf"],
            ["--seed", "-3"],
        ],
    )
    def test_shortcut_validation(self, argv):
        with pytest.raises(SystemExit):
            make_train_parser().parse_args(["--config", "x", "--dataset", "luna16", *argv])


# ---------------------------------------------------------------------------
# apply_overrides: ánh xạ cờ tắt và thứ tự ưu tiên (§2.1, §6.1.1)
# ---------------------------------------------------------------------------
class TestApplyOverrides:
    @pytest.mark.parametrize(
        "argv, key, expected",
        [
            (["--model", "resunetpp"], "model.name", "resunetpp"),
            (["--aaf-mode", "pa"], "aaf.mode", "pa"),
            (["--loss", "dice+sobel"], "loss.name", "dice+sobel"),
            (["--lambda", "0.5"], "loss.lambda", 0.5),
            (["--epochs", "5"], "train.epochs", 5),
            (["--batch-size", "8"], "train.batch_size", 8),
            (["--lr", "0.001"], "train.optimizer.lr", 0.001),
            (["--weight-decay", "0"], "train.optimizer.weight_decay", 0.0),
            (["--amp"], "train.amp", True),
            (["--augment"], "data.augment", True),
            (["--seed", "7"], "seed", 7),
        ],
    )
    def test_each_shortcut_maps_to_its_yaml_key(self, argv, key, expected):
        _, config_only, resolved, _, _ = resolve(*argv)
        assert get_by_dotted(resolved, key) == expected
        diff = {k for k, v in flatten(resolved).items() if flatten(config_only)[k] != v}
        assert diff == {key, "dataset"}

    def test_does_not_mutate_input(self):
        config_only = load_config(CAAF)
        snapshot = json.dumps(config_only, sort_keys=True)
        args = make_train_parser().parse_args(
            ["--config", str(CAAF), "--dataset", "lidc", "--batch-size", "8", "--set", "model.aspp.gap=true"]
        )
        resolved = apply_overrides(config_only, args)
        assert json.dumps(config_only, sort_keys=True) == snapshot
        resolved["model"]["channels"].append(512)
        assert config_only["model"]["channels"] == [32, 64, 128, 256]

    def test_dataset_is_stored(self):
        _, config_only, resolved, _, overrides = resolve(dataset="lidc")
        assert config_only["dataset"] is None
        assert resolved["dataset"] == "lidc" and overrides["dataset"] == "lidc"

    def test_dataset_flag_beats_set(self):
        _, _, resolved, _, _ = resolve("--set", "dataset=lidc", dataset="luna16")
        assert resolved["dataset"] == "luna16"

    def test_precedence_base_config_shortcut_set(self, tmp_path):
        child = write_yaml(tmp_path / "exp.yaml", {
            "base": str(BASE_YAML), "train": {"batch_size": 32, "epochs": 50},
        })
        assert load_config(BASE_YAML)["train"]["batch_size"] == 16            # base.yaml
        assert load_config(child)["train"]["batch_size"] == 32                # --config > base
        _, _, cfg, _, _ = resolve("--batch-size", "8", config=child)
        assert cfg["train"]["batch_size"] == 8                                # cờ tắt > --config
        assert cfg["train"]["epochs"] == 50                                   # khóa khác giữ theo --config
        _, _, cfg, _, _ = resolve("--batch-size", "8", "--set", "train.batch_size=4", config=child)
        assert cfg["train"]["batch_size"] == 4                                # --set > cờ tắt
        _, _, cfg, _, _ = resolve("--set", "train.batch_size=4", "--batch-size", "8", config=child)
        assert cfg["train"]["batch_size"] == 4                                # không phụ thuộc thứ tự CLI

    def test_unknown_set_key_raises(self):
        with pytest.raises(KeyError):
            resolve("--set", "train.optimiser.lr=0.1")

    def test_prepare_run_config_matches_manual_pipeline(self):
        argv = ["--config", str(CAAF), "--dataset", "lidc", "--lr", "0.001", "--set", "model.aspp.gap=true"]
        args = make_train_parser().parse_args(argv)
        assert utils.prepare_run_config(args) == resolve(*argv[4:], dataset="lidc")[1:]


# ---------------------------------------------------------------------------
# make_run_name (§2.1, §6.1.2)
# ---------------------------------------------------------------------------
class TestRunName:
    def test_default_name(self):
        assert run_name() == "caaf_hausdorff_s42"

    def test_deterministic(self):
        argv = ("--lr", "0.001", "--no-amp", "--set", "model.aspp.gap=true", "--set", "train.grad_clip=1.0")
        first, second = resolve(*argv), resolve(*argv)
        assert first[3] == second[3]
        assert first[4] == second[4]

    def test_name_does_not_depend_on_dataset(self):
        assert run_name("--batch-size", "8", dataset="luna16") == run_name("--batch-size", "8", dataset="lidc")

    def test_noop_shortcuts_add_no_token(self):
        name = run_name(
            "--model", "caaf_resunet", "--aaf-mode", "adaptive", "--loss", "dice+hausdorff",
            "--lambda", "1", "--epochs", "100", "--batch-size", "16", "--lr", "1e-4",
            "--weight-decay", "0.01", "--no-amp", "--no-augment", "--seed", "42",
        )
        assert name == "caaf_hausdorff_s42"

    def test_noop_set_adds_no_token(self):
        name = run_name("--set", "model.aspp.gap=false", "--set", "train.optimizer.lr=1.0e-4", "--set", "seed=42")
        assert name == "caaf_hausdorff_s42"

    def test_seed_shortcut_equals_set_seed(self):
        a = resolve("--seed", "7")
        b = resolve("--set", "seed=7")
        assert a[3] == b[3] == "caaf_hausdorff_s7"
        assert a[2] == b[2]
        assert a[4]["tokens"] == b[4]["tokens"] == []

    def test_token_order_is_fixed(self):
        name = run_name(
            "--augment", "--amp", "--weight-decay", "0", "--lr", "0.001", "--batch-size", "8",
            "--epochs", "5", "--lambda", "0.5", "--loss", "dice+sobel", "--aaf-mode", "pa",
            "--model", "resunetpp",
        )
        assert name == (
            "caaf_hausdorff_s42__model-resunetpp_aaf-pa_loss-dice+sobel_lam-0.5_ep-5_bs-8"
            "_lr-0.001_wd-0_amp_aug"
        )
        with_set = run_name("--model", "resunetpp", "--augment", "--set", "model.aspp.gap=true")
        assert with_set.startswith("caaf_hausdorff_s42__model-resunetpp_aug_set-")
        assert HASH_RE.search(with_set)

    def test_noamp_and_noaug_tokens(self, tmp_path):
        child = write_yaml(tmp_path / "amp_on.yaml", {
            "base": str(BASE_YAML), "train": {"amp": True}, "data": {"augment": True},
        })
        assert run_name("--no-amp", "--no-augment", config=child) == "amp_on_s42__noamp_noaug"
        assert run_name("--amp", "--augment", config=child) == "amp_on_s42"

    def test_set_on_shortcut_key_gives_same_name_as_shortcut(self):
        assert run_name("--set", "train.batch_size=8") == run_name("--batch-size", "8") == "caaf_hausdorff_s42__bs-8"
        assert run_name("--set", "train.optimizer.lr=1e-3") == run_name("--lr", "0.001")
        assert run_name("--set", "train.amp=true") == "caaf_hausdorff_s42__amp"

    def test_set_hash_matches_spec_formula(self):
        name = run_name("--set", "train.grad_clip=1.0", "--set", "model.aspp.gap=true")
        payload = "model.aspp.gap=True\ntrain.grad_clip=1.0"
        assert name == f"caaf_hausdorff_s42__set-{hashlib.sha1(payload.encode()).hexdigest()[:6]}"

    def test_set_hash_independent_of_order(self):
        a = run_name("--set", "model.aspp.gap=true", "--set", "train.grad_clip=1.0")
        b = run_name("--set", "train.grad_clip=1.0", "--set", "model.aspp.gap=true")
        assert a == b and HASH_RE.search(a)

    def test_set_hash_changes_with_value(self):
        a = run_name("--set", "train.grad_clip=1.0")
        b = run_name("--set", "train.grad_clip=2.0")
        assert a != b
        assert HASH_RE.search(a) and HASH_RE.search(b)

    def test_seed_excluded_from_set_hash(self):
        plain = run_name("--set", "model.aspp.gap=true")
        via_set = run_name("--set", "seed=7", "--set", "model.aspp.gap=true")
        via_flag = run_name("--seed", "7", "--set", "model.aspp.gap=true")
        assert via_set == via_flag
        assert via_set.split("__", 1)[1] == plain.split("__", 1)[1]
        assert via_set.startswith("caaf_hausdorff_s7__set-")

    def test_repeated_set_key_uses_final_value(self):
        assert run_name("--set", "train.grad_clip=1.0", "--set", "train.grad_clip=null") == "caaf_hausdorff_s42"

    def test_explicit_run_name(self):
        _, _, _, name, overrides = resolve("--run-name", "my_run", "--batch-size", "8")
        assert name == "my_run" and overrides["run_name"] == "my_run"
        assert overrides["auto_run_name"] == "caaf_hausdorff_s42__bs-8"

    @pytest.mark.parametrize("bad", ["a/b", "a\\b", "..", "  ", "x:y"])
    def test_invalid_explicit_run_name(self, bad):
        with pytest.raises(ValueError):
            resolve("--run-name", bad)

    def test_overrides_dict_content(self):
        _, _, _, name, ov = resolve("--batch-size", "8", "--seed", "7", "--set", "model.aspp.gap=true")
        json.dumps(ov)  # phải ghi được ra overrides.json
        assert ov["run_name"] == name
        assert ov["config"].endswith("configs/caaf_hausdorff.yaml")
        assert ov["dataset"] == "luna16" and ov["seed"] == 7
        assert ov["shortcuts"] == {"train.batch_size": 8, "seed": 7}
        assert ov["set"] == ["model.aspp.gap=true"]
        assert ov["changed"]["train.batch_size"] == {"config": 16, "resolved": 8}
        assert ov["changed"]["seed"] == {"config": 42, "resolved": 7}
        assert ov["changed"]["model.aspp.gap"] == {"config": False, "resolved": True}
        assert ov["tokens"][0] == "bs-8" and ov["tokens"][-1] == f"set-{ov['set_hash']}"


# ---------------------------------------------------------------------------
# config_resolved.yaml, overrides.json (§0.3.1, Phụ lục C)
# ---------------------------------------------------------------------------
class TestSaveResolved:
    def test_config_resolved_roundtrip_keeps_dataset(self, tmp_path):
        _, _, resolved, name, overrides = resolve("--batch-size", "8", "--lr", "1e-3", dataset="lidc")
        run_dir = tmp_path / "experiments" / resolved["dataset"] / name
        path = save_yaml(resolved, run_dir / utils.RESOLVED_CONFIG_NAME)
        assert path.name == "config_resolved.yaml"
        loaded = load_config(path)
        assert loaded == resolved
        assert loaded["dataset"] == "lidc"
        assert type(loaded["train"]["optimizer"]["lr"]) is float
        assert type(loaded["loss"]["dice_eps"]) is float
        save_json(overrides, run_dir / utils.OVERRIDES_NAME)
        assert json.loads((run_dir / "overrides.json").read_text(encoding="utf-8")) == overrides
        assert sorted(p.name for p in run_dir.iterdir()) == ["config_resolved.yaml", "overrides.json"]

    def test_save_json_utf8(self, tmp_path):
        path = save_json({"ghi_chú": "nốt phổi"}, tmp_path / "a.json")
        assert "nốt phổi" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint atomic với module torch giả (§2.1)
# ---------------------------------------------------------------------------
class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.fail_save = False
        self.load_calls: list[dict] = []

    def save(self, obj, fh):
        if self.fail_save:
            fh.write(b"partial")
            raise RuntimeError("lỗi giả lập khi đang ghi")
        pickle.dump(obj, fh)

    def load(self, path, map_location=None, weights_only=None):
        self.load_calls.append({"map_location": map_location, "weights_only": weights_only})
        with open(path, "rb") as fh:
            return pickle.load(fh)


@pytest.fixture
def fake_torch(monkeypatch):
    module = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


class TestCheckpoint:
    def test_roundtrip_and_weights_only_false(self, tmp_path, fake_torch):
        path = tmp_path / "run" / "nested" / "last.pt"
        state = {"epoch": 3, "model": {"w": [1.0, 2.0]}, "rng": {"python": random.getstate()}}
        assert save_checkpoint(path, state) == path
        assert load_checkpoint(path, map_location="cuda:0") == state
        assert fake_torch.load_calls[-1] == {"map_location": "cuda:0", "weights_only": False}
        load_checkpoint(path)
        assert fake_torch.load_calls[-1] == {"map_location": "cpu", "weights_only": False}
        assert [p.name for p in path.parent.iterdir()] == ["last.pt"]

    def test_failure_keeps_previous_checkpoint(self, tmp_path, fake_torch):
        path = tmp_path / "last.pt"
        save_checkpoint(path, {"epoch": 1})
        fake_torch.fail_save = True
        with pytest.raises(RuntimeError):
            save_checkpoint(path, {"epoch": 2})
        fake_torch.fail_save = False
        assert load_checkpoint(path) == {"epoch": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]

    def test_tmp_dir_on_local_disk(self, tmp_path, fake_torch):
        local, drive = tmp_path / "local_ssd", tmp_path / "drive" / "run"
        path = drive / "last.pt"
        save_checkpoint(path, {"epoch": 1}, tmp_dir=local)
        save_checkpoint(path, {"epoch": 2}, tmp_dir=local)
        assert load_checkpoint(path) == {"epoch": 2}
        assert list(local.iterdir()) == []
        assert [p.name for p in drive.iterdir()] == ["last.pt"]

    def test_tmp_dir_failure_cleans_up(self, tmp_path, fake_torch):
        local, path = tmp_path / "local_ssd", tmp_path / "drive" / "last.pt"
        save_checkpoint(path, {"epoch": 1}, tmp_dir=local)
        fake_torch.fail_save = True
        with pytest.raises(RuntimeError):
            save_checkpoint(path, {"epoch": 2}, tmp_dir=local)
        fake_torch.fail_save = False
        assert load_checkpoint(path) == {"epoch": 1}
        assert list(local.iterdir()) == []
        assert [p.name for p in path.parent.iterdir()] == ["last.pt"]

    def test_load_missing_checkpoint(self, tmp_path, fake_torch):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "missing.pt")

    def test_torch_is_imported_lazily(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # mô phỏng máy chưa cài torch
        with pytest.raises(ImportError):
            save_checkpoint(tmp_path / "last.pt", {})
        with pytest.raises(ImportError):
            load_checkpoint(tmp_path / "last.pt")


# ---------------------------------------------------------------------------
# seed_everything, get_logger, git_commit (§2.1)
# ---------------------------------------------------------------------------
class TestSeedLoggerGit:
    def test_python_random_reproducible(self):
        utils.seed_everything(123)
        a = [random.random() for _ in range(5)]
        utils.seed_everything(123)
        b = [random.random() for _ in range(5)]
        utils.seed_everything(124)
        c = [random.random() for _ in range(5)]
        assert a == b and a != c

    def test_numpy_reproducible(self):
        np = pytest.importorskip("numpy")
        utils.seed_everything(2024)
        a = np.random.rand(4)
        utils.seed_everything(2024)
        assert (np.random.rand(4) == a).all()

    def test_works_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        utils.seed_everything(1, deterministic=True)

    @pytest.mark.parametrize("deterministic", [False, True])
    def test_torch_flags(self, monkeypatch, deterministic):
        calls = {}
        torch = types.ModuleType("torch")
        torch.manual_seed = lambda s: calls.setdefault("manual_seed", s)
        torch.cuda = types.SimpleNamespace(manual_seed_all=lambda s: calls.setdefault("cuda_seed", s))
        torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=None, benchmark=None))
        torch.use_deterministic_algorithms = lambda flag, warn_only=False: calls.setdefault(
            "det_algo", (flag, warn_only)
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        utils.seed_everything(5, deterministic=deterministic)
        assert calls["manual_seed"] == 5 and calls["cuda_seed"] == 5
        assert torch.backends.cudnn.deterministic is deterministic
        assert torch.backends.cudnn.benchmark is (not deterministic)
        assert calls["det_algo"] == (deterministic, True)

    def test_logger_no_duplicate_handlers_and_utf8_file(self, tmp_path):
        log_file = tmp_path / "logs" / "train.log"
        logger = utils.get_logger("test_config.logger", log_file)
        try:
            again = utils.get_logger("test_config.logger", log_file)
            assert again is logger
            assert len(logger.handlers) == 2
            assert sum(type(h) is logging.StreamHandler for h in logger.handlers) == 1
            logger.info("Dice của nốt phổi = 0.9088")
            for handler in logger.handlers:
                handler.flush()
            assert "Dice của nốt phổi = 0.9088" in log_file.read_text(encoding="utf-8")
        finally:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

    def test_git_commit_outside_repo(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        assert utils.git_commit(cwd=tmp_path) == "unknown"

    def test_git_commit_in_project(self):
        sha = utils.git_commit()
        assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha)
