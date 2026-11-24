"""Hạ tầng dùng chung — SPECS Phase 2 (§2.1), §0.3.1 (kế thừa config), §6.1.1–6.1.2 (cờ tắt, tên run).

Gồm: đọc/ghi config YAML (kế thừa `base` + deep-merge + `--set`), tham số CLI chung, 11 cờ tắt
siêu tham số, đặt tên run tất định, cố định seed, logging, checkpoint atomic, commit git.

Ở mức module chỉ phụ thuộc thư viện chuẩn + PyYAML. `numpy` và `torch` được import lười bên
trong hàm, nên `tests/test_config.py` chạy được ở máy local không cài torch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = [
    "REPO_ROOT",
    "DATASETS",
    "MODEL_NAMES",
    "AAF_MODES",
    "LOSS_NAMES",
    "TRAIN_SHORTCUTS",
    "RESOLVED_CONFIG_NAME",
    "OVERRIDES_NAME",
    "deep_merge",
    "parse_set_value",
    "parse_set_items",
    "get_by_dotted",
    "set_by_dotted",
    "apply_set_overrides",
    "load_config",
    "save_yaml",
    "save_json",
    "add_common_args",
    "add_train_shortcuts",
    "apply_overrides",
    "make_run_name",
    "prepare_run_config",
    "seed_everything",
    "get_logger",
    "save_checkpoint",
    "load_checkpoint",
    "git_commit",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = ("luna16", "lidc")
MODEL_NAMES = ("caaf_resunet", "resunetpp")
AAF_MODES = ("pa", "ca", "static", "adaptive")
LOSS_NAMES = ("dice", "dice+sobel", "dice+laplacian", "dice+hausdorff")

RESOLVED_CONFIG_NAME = "config_resolved.yaml"
OVERRIDES_NAME = "overrides.json"

# (dest trong argparse, khóa YAML) — đúng thứ tự token của tên run (§6.1.2); `seed` không sinh token.
TRAIN_SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("model", "model.name"),
    ("aaf_mode", "aaf.mode"),
    ("loss", "loss.name"),
    ("lam", "loss.lambda"),
    ("epochs", "train.epochs"),
    ("batch_size", "train.batch_size"),
    ("lr", "train.optimizer.lr"),
    ("weight_decay", "train.optimizer.weight_decay"),
    ("amp", "train.amp"),
    ("augment", "data.augment"),
    ("seed", "seed"),
)
_TOKEN_PREFIX = {
    "model.name": "model",
    "aaf.mode": "aaf",
    "loss.name": "loss",
    "loss.lambda": "lam",
    "train.epochs": "ep",
    "train.batch_size": "bs",
    "train.optimizer.lr": "lr",
    "train.optimizer.weight_decay": "wd",
}
_BOOL_TOKENS = {"train.amp": ("amp", "noamp"), "data.augment": ("aug", "noaug")}
_SHORTCUT_KEYS = frozenset(key for _, key in TRAIN_SHORTCUTS)
_MISSING = object()


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------
class _ConfigLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` với hai điểm bổ sung.

    1. Nhận `1e-4`, `5E3`, `1.0e4` là float. PyYAML theo YAML 1.1 nên coi chúng là chuỗi,
       trong khi SPECS §0.3.1 yêu cầu `--set train.optimizer.lr=1e-4` ra float.
    2. Cấm khóa trùng trong cùng một mapping: PyYAML mặc định lặng lẽ lấy khóa sau cùng.
    """


_FLOAT_RE = re.compile(
    r"""^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
        |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789."))


def _construct_mapping_no_duplicates(loader: _ConfigLoader, node: yaml.MappingNode) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicated = key in seen
        except TypeError:  # khóa không hash được: để construct_mapping báo lỗi chuẩn
            continue
        if duplicated:
            raise yaml.constructor.ConstructorError(
                "khi đọc mapping", node.start_mark, f"khóa trùng lặp {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_duplicates)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_ConfigLoader)  # noqa: S506 — _ConfigLoader kế thừa SafeLoader


# ---------------------------------------------------------------------------
# Config: deep-merge, khóa dạng a.b.c, --set
# ---------------------------------------------------------------------------
def deep_merge(parent: Mapping, child: Mapping) -> dict:
    """Trộn đệ quy `child` đè lên `parent`, trả dict mới (không sửa đầu vào).

    dict gặp dict thì trộn tiếp; mọi kiểu khác (list, scalar, None) thì `child` thay thế hẳn.
    """
    merged = copy.deepcopy(dict(parent))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_set_value(raw: str) -> Any:
    """Parse giá trị của `--set` như YAML: `8` → int, `1e-4` → float, `[1,2]` → list, `true` → bool."""
    try:
        return _yaml_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Giá trị --set không phải YAML hợp lệ: {raw!r} ({exc})") from exc


def _split_set_item(item: str) -> tuple[str, str]:
    if not isinstance(item, str) or "=" not in item:
        raise ValueError(f"--set cần dạng khoa.con=gia_tri, nhận {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise ValueError(f"Khóa --set không hợp lệ: {item!r}")
    return key, raw


def parse_set_items(items: Iterable[str] | None) -> list[tuple[str, Any]]:
    """Tách và parse danh sách `a.b.c=value`, giữ nguyên thứ tự xuất hiện."""
    parsed = []
    for item in items or []:
        key, raw = _split_set_item(item)
        parsed.append((key, parse_set_value(raw)))
    return parsed


def get_by_dotted(cfg: Mapping, key: str) -> Any:
    """Đọc `cfg["a"]["b"]["c"]` từ khóa `"a.b.c"`; khóa không tồn tại → `KeyError`."""
    node: Any = cfg
    parts = key.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"Khóa config không tồn tại: {key!r} (dừng ở {'.'.join(parts[: i + 1])!r})")
        node = node[part]
    return node


def set_by_dotted(cfg: dict, key: str, value: Any) -> None:
    """Gán `value` vào khóa `"a.b.c"` đã có sẵn trong `cfg`.

    Khóa trung gian hoặc khóa lá không tồn tại → `KeyError`, để bắt lỗi gõ sai tên khóa.
    """
    parts = key.split(".")
    node: Any = cfg
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Khóa config không tồn tại: {key!r} (dừng ở {'.'.join(parts[: i + 1])!r})")
        if i == len(parts) - 1:
            node[part] = value
        else:
            node = node[part]


def apply_set_overrides(cfg: dict, items: Iterable[str] | None) -> dict:
    """Áp lần lượt các `--set a.b.c=value` lên `cfg` (sửa tại chỗ, phần tử sau thắng)."""
    for key, value in parse_set_items(items):
        set_by_dotted(cfg, key, value)
    return cfg


def _load_config_tree(path: Path, chain: tuple[Path, ...]) -> dict:
    resolved = path.resolve()
    if resolved in chain:
        loop = " -> ".join(str(p) for p in (*chain, resolved))
        raise ValueError(f"Vòng lặp kế thừa `base`: {loop}")
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp config: {path}")
    data = _yaml_load(path.read_text(encoding="utf-8-sig"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: gốc YAML phải là mapping, nhận {type(data).__name__}")
    base = data.pop("base", None)
    if base is None:
        return data
    if not isinstance(base, str) or not base.strip():
        raise TypeError(f"{path}: `base` phải là đường dẫn dạng chuỗi, nhận {base!r}")
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    parent = _load_config_tree(base_path, (*chain, resolved))
    return deep_merge(parent, data)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Đọc config YAML có kế thừa (SPECS §0.3.1).

    Nếu tệp có khóa `base` (đường dẫn tương đối theo thư mục của tệp đó, hoặc tuyệt đối),
    nạp đệ quy tệp cha rồi deep-merge tệp con đè lên; khóa `base` bị xóa khỏi kết quả.
    Sau đó áp `overrides` dạng `a.b.c=value`; khóa không tồn tại → `KeyError`.
    """
    cfg = _load_config_tree(Path(path), chain=())
    if overrides:
        apply_set_overrides(cfg, overrides)
    return cfg


def _atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def save_yaml(obj: Any, path: str | Path) -> Path:
    """Ghi YAML UTF-8 (atomic), giữ thứ tự khóa — dùng cho `config_resolved.yaml`."""
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=None)
    return _atomic_write_text(path, text)


def save_json(obj: Any, path: str | Path) -> Path:
    """Ghi JSON UTF-8 (atomic) — dùng cho `overrides.json`, `results.json`."""
    return _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"cần số nguyên > 0, nhận {text!r}")
    return value


def _nonneg_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"cần số nguyên >= 0, nhận {text!r}")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"cần số thực > 0, nhận {text!r}")
    return value


def _nonneg_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"cần số thực >= 0, nhận {text!r}")
    return value


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Thêm 5 tham số chung (Phụ lục A): --config, --dataset, --data-dir, --out-dir, --set."""
    parser.add_argument("--config", required=True, help="Tệp YAML thực nghiệm, ví dụ configs/caaf_hausdorff.yaml")
    parser.add_argument("--dataset", required=True, choices=DATASETS, help="Bộ dữ liệu: luna16 hoặc lidc")
    parser.add_argument(
        "--data-dir", default="data/processed", help="Thư mục chứa <dataset>/{train,val,test}.h5"
    )
    parser.add_argument("--out-dir", default="experiments", help="Thư mục gốc chứa <dataset>/<run_name>/")
    parser.add_argument(
        "--set",
        dest="set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ghi đè khóa config, lặp được: --set train.batch_size=8 --set model.aspp.gap=true",
    )
    return parser


def add_train_shortcuts(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Thêm 11 cờ tắt siêu tham số (SPECS §6.1.1); mặc định `None` = giữ theo --config."""
    group = parser.add_argument_group("cờ tắt siêu tham số (None = giữ theo --config)")
    group.add_argument("--model", dest="model", choices=MODEL_NAMES, default=None, help="model.name")
    group.add_argument("--aaf-mode", dest="aaf_mode", choices=AAF_MODES, default=None, help="aaf.mode")
    group.add_argument("--loss", dest="loss", choices=LOSS_NAMES, default=None, help="loss.name")
    group.add_argument("--lambda", dest="lam", type=_nonneg_float, default=None, help="loss.lambda")
    group.add_argument("--epochs", dest="epochs", type=_positive_int, default=None, help="train.epochs")
    group.add_argument(
        "--batch-size", dest="batch_size", type=_positive_int, default=None, help="train.batch_size"
    )
    group.add_argument("--lr", dest="lr", type=_positive_float, default=None, help="train.optimizer.lr")
    group.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=_nonneg_float,
        default=None,
        help="train.optimizer.weight_decay",
    )
    group.add_argument("--amp", dest="amp", action=argparse.BooleanOptionalAction, default=None, help="train.amp")
    group.add_argument(
        "--augment", dest="augment", action=argparse.BooleanOptionalAction, default=None, help="data.augment"
    )
    group.add_argument("--seed", dest="seed", type=_nonneg_int, default=None, help="seed")
    return parser


def apply_overrides(base_cfg: Mapping, args: argparse.Namespace) -> dict:
    """Trả config cuối cùng theo thứ tự ưu tiên `base.yaml` < `--config` < cờ tắt < `--set`.

    `base_cfg` là kết quả `load_config(args.config)` (đã gồm base + config). Không sửa đầu vào.
    Cuối cùng gán `dataset` từ `args.dataset` (luôn thắng `--set dataset=...`).
    """
    resolved = copy.deepcopy(dict(base_cfg))
    for dest, key in TRAIN_SHORTCUTS:
        value = getattr(args, dest, None)
        if value is not None:
            set_by_dotted(resolved, key, value)
    apply_set_overrides(resolved, getattr(args, "set", None))
    dataset = getattr(args, "dataset", None)
    if dataset is not None:
        resolved["dataset"] = dataset
    return resolved


# ---------------------------------------------------------------------------
# Tên run
# ---------------------------------------------------------------------------
def _lookup(cfg: Mapping, key: str) -> Any:
    try:
        return get_by_dotted(cfg, key)
    except KeyError:
        return _MISSING


def _differs(old: Any, new: Any) -> bool:
    if old is _MISSING or new is _MISSING:
        return old is not new
    if isinstance(old, bool) != isinstance(new, bool):
        return True
    return old != new


def _fmt_token_value(value: Any) -> str:
    if isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, float):
        text = repr(value)  # biểu diễn ngắn nhất, khôi phục chính xác giá trị
        if text.endswith(".0"):
            text = text[:-2]
    else:
        text = str(value)
    return re.sub(r"[^A-Za-z0-9.+\-]", "-", text) or "none"


def _token_for(key: str, value: Any) -> str:
    if key in _BOOL_TOKENS:
        on, off = _BOOL_TOKENS[key]
        if value is True:
            return on
        if value is False:
            return off
        return f"{on}-{_fmt_token_value(value)}"
    return f"{_TOKEN_PREFIX[key]}-{_fmt_token_value(value)}"


def _jsonable(value: Any) -> Any:
    return None if value is _MISSING else value


def _validate_run_name(name: str) -> str:
    name = str(name).strip()
    if not name or name in {".", ".."} or re.search(r'[\\/:*?"<>|]', name):
        raise ValueError(f"--run-name không hợp lệ làm tên thư mục: {name!r}")
    return name


def make_run_name(
    cfg_path: str | Path,
    config_only_cfg: Mapping,
    resolved_cfg: Mapping,
    args: argparse.Namespace,
) -> tuple[str, dict]:
    """Đặt tên run tất định (SPECS §6.1.2), trả `(run_name, overrides_dict)`.

    - Mặc định `{config_stem}_s{seed}` với `seed` lấy từ `resolved_cfg`, nên `--seed 7` và
      `--set seed=7` cho cùng tên.
    - Với 10 khóa của cờ tắt (trừ `seed`), khóa nào có giá trị cuối khác `--config` thì thêm
      token theo thứ tự cố định: model, aaf, loss, lam, ep, bs, lr, wd, amp|noamp, aug|noaug.
      Đổi bằng cờ tắt hay bằng `--set` đều cho cùng token.
    - Các khóa `--set` còn lại (bỏ `seed`, `dataset`, 10 khóa đã có token và khóa có giá trị cuối
      trùng `--config`) được sort theo `(key, repr(val))`, băm sha1 lấy 6 ký tự → `set-<hash6>`.
    - `args.run_name` (nếu khác None) thay thế toàn bộ quy tắc trên.

    `overrides_dict` dùng để ghi `run_dir/overrides.json`.
    """
    stem = Path(cfg_path).stem
    seed = resolved_cfg["seed"]
    base_name = f"{stem}_s{seed}"

    tokens: list[str] = []
    changed: dict[str, dict[str, Any]] = {}
    for _, key in TRAIN_SHORTCUTS:
        old, new = _lookup(config_only_cfg, key), _lookup(resolved_cfg, key)
        if not _differs(old, new):
            continue
        changed[key] = {"config": _jsonable(old), "resolved": _jsonable(new)}
        if key != "seed":
            tokens.append(_token_for(key, new))

    raw_set = list(getattr(args, "set", None) or [])
    hashed: dict[str, Any] = {}
    for key, _ in parse_set_items(raw_set):
        if key in _SHORTCUT_KEYS or key == "dataset":
            continue
        old, new = _lookup(config_only_cfg, key), _lookup(resolved_cfg, key)
        if _differs(old, new):
            hashed[key] = new  # giá trị cuối cùng (khóa lặp lại thì lần sau thắng)
            changed[key] = {"config": _jsonable(old), "resolved": _jsonable(new)}

    set_hash = None
    if hashed:
        items = sorted(hashed.items(), key=lambda kv: (kv[0], repr(kv[1])))
        payload = "\n".join(f"{k}={v!r}" for k, v in items)
        set_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:6]
        tokens.append(f"set-{set_hash}")

    auto_name = f"{base_name}__{'_'.join(tokens)}" if tokens else base_name
    explicit = getattr(args, "run_name", None)
    run_name = _validate_run_name(explicit) if explicit is not None else auto_name

    shortcuts = {
        key: getattr(args, dest) for dest, key in TRAIN_SHORTCUTS if getattr(args, dest, None) is not None
    }
    overrides = {
        "run_name": run_name,
        "auto_run_name": auto_name,
        "config": Path(cfg_path).as_posix(),
        "dataset": resolved_cfg.get("dataset"),
        "seed": seed,
        "shortcuts": shortcuts,
        "set": raw_set,
        "changed": changed,
        "tokens": tokens,
        "set_hash": set_hash,
    }
    return run_name, overrides


def prepare_run_config(args: argparse.Namespace) -> tuple[dict, dict, str, dict]:
    """Bước 1 của §6.1.3: trả `(config_only, resolved, run_name, overrides)` từ `args`."""
    config_only = load_config(args.config)
    resolved = apply_overrides(config_only, args)
    run_name, overrides = make_run_name(args.config, config_only, resolved, args)
    return config_only, resolved, run_name, overrides


# ---------------------------------------------------------------------------
# Seed, logging
# ---------------------------------------------------------------------------
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Đặt seed cho `random`, `numpy` (nếu có) và `torch` (nếu có, import lười)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Logger ra console (stdout) và tùy chọn ra tệp UTF-8 (chế độ append); gọi lại không gắn trùng handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if log_file is not None:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        has_file = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_path
            for h in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def _torch_save_fsync(torch_module: Any, state: dict, target: Path) -> None:
    with open(target, "wb") as fh:
        torch_module.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())


def save_checkpoint(path: str | Path, state: dict, tmp_dir: str | Path | None = None) -> Path:
    """Ghi checkpoint atomic: ghi ra `<path>.tmp` rồi `os.replace` sang `path`.

    Khi `path` nằm trên Google Drive FUSE, truyền `tmp_dir` ở ổ cục bộ (ví dụ `/tmp`): checkpoint
    được `torch.save` ở đó trước, chép một lần sang `<path>.tmp` cạnh đích rồi mới `os.replace`.
    Lỗi giữa chừng không làm hỏng checkpoint cũ và không để lại tệp tạm.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_tmp = path.with_name(f"{path.name}.tmp")
    local_tmp: Path | None = None
    try:
        if tmp_dir is None:
            _torch_save_fsync(torch, state, final_tmp)
        else:
            tmp_dir = Path(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=tmp_dir)
            os.close(fd)
            local_tmp = Path(name)
            _torch_save_fsync(torch, state, local_tmp)
            with open(local_tmp, "rb") as src, open(final_tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
        os.replace(final_tmp, path)
    except BaseException:
        final_tmp.unlink(missing_ok=True)
        raise
    finally:
        if local_tmp is not None:
            local_tmp.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict:
    """Nạp checkpoint đầy đủ (có `rng`, `cfg`) với `weights_only=False` — cần cho PyTorch >= 2.6."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------
def git_commit(cwd: str | Path | None = None) -> str:
    """Trả mã commit `HEAD` của repo (mặc định thư mục gốc dự án); `"unknown"` nếu ngoài git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd if cwd is not None else REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        return sha
    return "unknown"
