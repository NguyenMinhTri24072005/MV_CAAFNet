# PLAN — Kế hoạch phục dựng mã nguồn CAAF-ResUNet

| Mục | Giá trị |
|---|---|
| Đặc tả kỹ thuật | `docs/PLANNING/SPECS.md` (tổ chức 1-1 theo **Phase 0 → Phase 9** + Phụ lục A–C) |
| Lý do của các quyết định | `docs/SPEC_REVIEW.md` |
| Sổ điểm mơ hồ | `docs/PLANNING/OPEN_ISSUES.md` |
| Phiên bản | **v2.1 — 2026-11-21** (đồng bộ 1-1 với `SPECS.md` v2.1: chạy khảo sát & tiền xử lý trên **Colab Pro CPU High-RAM trực tiếp trên Google Drive**; cập nhật đường dẫn dữ liệu thô chuẩn trong `data/raw/{LIDC-IDRI,LUNA16}`) |

## Mục tiêu

1. Viết lại toàn bộ mã nguồn theo bài báo: mô hình CAAF-ResUNet, 3 boundary loss, pipeline dữ liệu LIDC-IDRI (DICOM-SEG) & LUNA16 (`.mhd` + DICOM-SEG), train/evaluate/thống kê/hình.
2. Tái lập **Table 5, 6, 7** và **Figure 2, 3** trên LUNA16 và LIDC-IDRI. Riêng Table 9/A1 (MEF-PN35) không làm được vì là dữ liệu nội bộ.
3. Mọi giả định được ghi lại. Mọi kết quả truy vết được về commit và config cụ thể.

## Tiêu chí thành công

Đây là mục tiêu, không phải cam kết, vì bài báo thiếu nhiều chi tiết.

| Loại | Tiêu chí |
|---|---|
| **Định tính** (bắt buộc) | (a) Hausdorff ≥ Sobel > Laplacian về Dice trên LUNA16. (b) CAAF > PCAM-static với p < 0.05 (paired t-test, 1.200 patch). (c) Mean w2 > w1 (AAC ưu tiên CA). (d) CAAF-Hausdorff > baseline ResUNet++. |
| **Định lượng** (mong muốn) | Dice của từng ô trong Table 5–7 lệch không quá **±3 điểm %** so với bài báo. *Ngưỡng ±3 do nhóm tự đặt, bài báo không có cơ sở cho con số này.* |
| **Kỹ thuật** (bắt buộc) | Toàn bộ `pytest` pass; mọi run có `config_resolved.yaml` và `commit.txt`; tập test cố định 1.200 patch mỗi bộ; 0% rò rỉ bệnh nhân. |

## Phân bố nơi chạy (Colab-first, không upload dữ liệu lớn từ local)

| Nơi | Việc |
|---|---|
| **Local Windows** (CPU i3, 20 GB RAM, Python 3.12.10) | Phase 0, 2; viết mã Phase 1–9; chạy `pytest` nhẹ (`test_config`, `test_ct_utils`); debug thử `--limit 3` nếu cần; `git push` lên GitHub; xem `reports/` mirror về từ Drive |
| **Colab Pro (CPU High-RAM)** | **Phase 1 (khảo sát & lập `data/index/`)** và **Phase 5 (tiền xử lý LUNA16 & LIDC, chia tập, đóng gói `.h5` qua `/content/` rồi copy 1 lần sang Drive)** |
| **Colab Pro (GPU)** | Phase 3–4 (`pytest` mô hình/loss/metric), Phase 6, 7, 8, 8b, 9 |

---

## Phase 0 — Môi trường và repo (Local)

| # | Việc | Ghi chú |
|---|---|---|
| 0.1 | **Tạo lại `.venv` bằng Python 3.12.10** | `.venv` hiện tại là Python 3.13.12, không khớp `.python-version`. Khuyến nghị đặt venv ở ổ cục bộ ngoài `D:\GG_1` (hoặc xóa `.venv` cũ rồi chạy `uv venv --python 3.12.10`) để tránh Google Drive Desktop đồng bộ hàng nghìn file nhỏ. |
| 0.2 | Tách `requirements.txt` thành `requirements-local.txt` và `requirements-colab.txt` (`SPECS` §0.1); cài bản local | **Không** cần `pylidc` / `SQLAlchemy<2` / `numpy<2` vì đọc trực tiếp IDC DICOM-SEG bằng `pydicom` + `SimpleITK`. |
| 0.3 | Bổ sung `.gitignore`: `.venv/`, `desktop.ini`, `data/`, `experiments/`, `*.pt`, `*.h5`, `*.npz`, `.ipynb_checkpoints/` | Thư mục nằm trong `D:\GG_1` (mirror Drive) nên có `desktop.ini`. |
| 0.4 | `git init`, commit khung; tạo repo GitHub **private**; push | Token lưu trong **Colab Secrets** (`GH_TOKEN`), dùng qua `GIT_ASKPASS`, không in ra output. |
| 0.5 | Tạo `src/__init__.py`, `notebooks/`; thay `scripts/colab_setup.ipynb` bằng bộ 5 notebook Colab (`SPECS` Phụ lục B); dọn các `scripts/run_*.sh` cũ | Khớp CLI Phụ lục A. |
| 0.6 | Cập nhật `configs/base.yaml`, 10 config thực nghiệm và tạo `configs/data.yaml` (`SPECS` §0.3) | Trỏ đúng `MV_CAAFNet/data/raw/LIDC-IDRI` và `MV_CAAFNet/data/raw/LUNA16`; đổi `loss: <str>` thành `loss: {name: ...}`; thêm `dataset: null`, `hausdorff_mode`, `att_sigmoid`. |

**Hoàn thành khi**: Python 3.12.10 chạy sạch `import numpy, scipy, h5py, pydicom, SimpleITK, yaml`; repo đã push lên GitHub; Colab clone được bằng `GH_TOKEN` và thấy sẵn `data/raw/{LIDC-IDRI, LUNA16}` trên Drive.

---

## Phase 1 — Khảo sát dữ liệu & Lập chỉ mục (Colab Pro CPU High-RAM) — *cổng quyết định*

Dữ liệu thô đã kiểm chứng nằm sẵn trong `data/raw/` (tự đồng bộ Google Drive qua `D:\GG_1`):
- LIDC-IDRI (IDC DICOM-SEG): `data/raw/LIDC-IDRI` (1.010 bệnh nhân, 1.018 `CT_*` scan, các thư mục `SEG_*/*.dcm`).
- LUNA16: `data/raw/LUNA16` (`subset0..9/*.mhd/.raw`, 888 scan, `annotations.csv` 1.186 nốt / 601 scan).

| # | Việc | Spec |
|---|---|---|
| 1.1 | Viết `src/data/build_index.py` (chỉ đọc **header** `stop_before_pixels=True`) và chạy qua `notebooks/00_colab_data_survey.ipynb` trên **Colab CPU High-RAM** → tạo `data/index/{ct_scans.csv, seg_index.csv, luna16_scans.csv, index_summary.json}` trên Drive | §1.2 |
| 1.2 | Kiểm tra tính toàn vẹn: 100% tệp `SEG` parse được `Nodule <K>` (hoặc xử lý ca dị biệt); 601/601 `series_uid` dương tính của LUNA16 đều có `SEG` trong `seg_index.csv`; kiểm tra 8 bệnh nhân LIDC có 2 scan | §1.1, §1.3 |
| 1.3 | Thống kê LIDC: số bệnh nhân, số scan, số cụm nốt theo `n_readers ∈ {1,2,3,4}`, phân bố `diameter_mm` (3D vs max-2D), diện tích mask theo lát | §1.3 |
| 1.4 | Ước lượng tổng số patch LIDC theo `min_readers ∈ {1,2,3,4}` và `min_mask_area_px ∈ {1,5,10}` → chọn cấu hình gần **15.128** (13.928 + 1.200) nhất | §1.3 (B3, B6) |
| 1.5 | **B15**: Kiểm tra ánh xạ tọa độ + khớp cụm DICOM-SEG ↔ `annotations.csv` trên **toàn bộ 888 scan / 1.186 nốt LUNA16** (tỉ lệ khớp 1-1, số ca 0-khớp/đa-khớp, số scan có `Direction`/`IOP` khác đơn vị); vẽ chồng mask lên `.mhd` ở **5 scan** (gồm ca `Direction != I` nếu có) để kiểm tra bằng mắt | §1.3, §5.3 |
| 1.6 | **B8**: đếm tỉ lệ patch có `n_other_nodules > 0` (< 2% thì giữ `include_other_nodules: true`, cao hơn thì ghi chú vào README) | §1.3 |
| 1.7 | **B9, B10**: histogram HU trong mask trên mẫu 50 scan (chọn `[-1000, 400]` vs `[-1000, 150]`); thống kê `spacing_x/y` (giữ `resample_mm: null` nếu đa số < 1 mm) | §1.3 |
| 1.8 | **B17**: thống kê tỉ lệ patch có `touches_border = True` (nốt vượt 64×64 khi `resample_mm: null`) | §1.3 |

**Hoàn thành khi**: `data/index/` đã lưu trên Drive; đã chốt `min_readers`, `clevel`, `min_mask_area_px`, `include_other_nodules`, `hu_window`, `resample_mm` trong `configs/data.yaml` (cập nhật B3–B17 trong `OPEN_ISSUES.md`); ánh xạ LUNA16 đạt tỉ lệ khớp cao và đã xác minh bằng mắt 5 scan.

**Cổng quyết định**: nếu số patch lệch > 20% so với Table 4 thì ghi nguyên nhân vào `README.md` rồi mới đi tiếp (không cố ép số).

---

## Phase 2 — Hạ tầng chung (Local)

| # | Việc | Spec |
|---|---|---|
| 2.1 | `src/utils.py`: `load_config` (base + deep-merge + `--set`, khóa lạ → `KeyError`), `add_common_args`, `add_train_shortcuts` (11 cờ tắt), `apply_overrides`, `make_run_name` (tất định, loại `seed` khỏi hash), `seed_everything`, `get_logger`, `save_checkpoint`/`load_checkpoint` (atomic, lazy torch import, `weights_only=False`), `git_commit` | §2.1 |
| 2.2 | `tests/test_config.py` | §2.2, Phụ lục C |

**Hoàn thành khi**: `pytest tests/test_config.py` pass ngay trên máy local (không cần torch).

---

## Phase 3 — Mô hình (Viết ở Local, chạy `pytest` trên Colab Pro GPU)

| # | Tệp | Spec | Test |
|---|---|---|---|
| 3.1 | `src/models/blocks.py` (`StemBlock`, `ResidualConvModule`, `AttentionModule`, `SEBlock`, `UpSample`) | §3.1 | `test_shapes` |
| 3.2 | `src/models/aspp.py` (`Conv → ReLU → BN`, tùy chọn `gap`) | §3.2 | `test_shapes` |
| 3.3 | `src/models/attention.py` (`PAM`, `CAM`, softmax fp32) | §3.3 | `test_shapes`, `test_grad_flow` |
| 3.4 | `src/models/aac.py` (Alg. 1, Eq. 1–9) | §3.4 | `test_aac` |
| 3.5 | `src/models/aaf.py` (`pa | ca | static | adaptive`) | §3.5 | `test_aaf_modes` |
| 3.6 | `src/models/caaf_resunet.py` (`ResUNetFamily`, `build_model`) & `src/models/resunetpp.py` | §3.6 | `test_shapes`, `test_aaf_modes`, `test_grad_flow` |

**Hoàn thành khi**: mọi test mô hình pass trên Colab (`10_colab_train.ipynb` `RUN_PYTEST=True`); có bảng số tham số 5 biến thể trong `reports/model_params.md`; xác nhận gradient `CAM` ở 64×64 không bão hòa và ghi lại VRAM đỉnh batch 16.

---

## Phase 4 — Loss, EDT và Metric (Viết ở Local, chạy `pytest` trên Colab Pro GPU)

| # | Tệp | Spec | Test |
|---|---|---|---|
| 4.1 | `src/losses/dice.py`, `sobel.py`, `laplacian.py` | §4.2 | `test_losses`, `test_grad_flow` |
| 4.2 | `src/data/compute_edt.py` + `src/losses/hausdorff_edt.py` (đủ 3 mode: `soft_pg_detach_gp`, `bidir_soft`, `karimi_sq`, log riêng `d_pg`/`d_gp`/`empty_pred_frac`) | §4.1, §4.2 | `test_edt`, `test_losses`, `test_grad_flow` |
| 4.3 | `src/losses/bal.py` (`BoundaryAwareLoss`) | §4.2 | `test_losses` |
| 4.4 | `src/metrics.py` (`confusion`, 5 metric, quy ước `[0,1]` CSV / `%` JSON) | §4.3 | `test_metrics` |

**Hoàn thành khi**: mọi test pass trên Colab; đo thời gian `hausdorff_edt_loss` trên batch 16 (nếu > 30% thời gian 1 bước train → ghi R6).

---

## Phase 5 — Pipeline tiền xử lý, chia tập & đóng gói HDF5 (Viết ở Local, chạy trên Colab Pro CPU High-RAM)

| # | Việc | Spec |
|---|---|---|
| 5.0 | `src/data/ct_utils.py` (chuyển thể từ `medical_prep.py`, trục `(Z,Y,X)`, no-mutate, atomic write, `_DONE`) + `tests/test_ct_utils.py` (chạy pass ngay ở local) | §5.1, Phụ lục C |
| 5.1 | `src/data/preprocess_lidc.py` (đọc CT DICOM + DICOM-SEG theo `ct_scans.csv`/`seg_index.csv`, resume theo `_DONE`, hỗ trợ `--limit`) | §5.2 |
| 5.2 | `src/data/preprocess_luna16.py` (đọc `.mhd` + DICOM-SEG **không mở CT DICOM LIDC**, ánh xạ tọa độ tổng quát `Direction`/`IOP`, khớp 1-1 `annotations.csv`, xuất `match_report.csv`, hỗ trợ `--limit`) | §5.3 |
| 5.3 | `src/data/make_splits.py` (chia theo `patient` 0% rò rỉ, đúng 1.200 test + `dropped`, hỗ trợ `--mode slice` cho S1) | §5.4 |
| 5.4 | `src/data/pack_h5.py` (nén `gzip opts=4` trên `/content/processed_tmp/`, ghi kèm `patches.csv`, `<ds>.json`, `manifest.json` SHA-256, rồi `--copy-to-drive-dir` 1 lần sang Drive + verify SHA-256) | §5.5 |
| 5.5 | `src/data/dataset.py` (`H5SegDataset`, augmentation D4 giữ EDT, `make_loaders`) + `tests/test_data.py` | §5.6, Phụ lục C |
| 5.6 | Viết & chạy `notebooks/01_colab_preprocess.ipynb` trên **Colab CPU High-RAM**: thử `LIMIT_SCANS=5` trước → kiểm tra output → chạy toàn bộ **LUNA16** (~15–25 phút) rồi **LIDC-IDRI** (có resume `_DONE` nếu ngắt phiên) → đóng gói `.h5` thẳng lên Drive → vẽ lưới 32 patch kiểm tra mỗi bộ | §5.7, Phụ lục B |

**Hoàn thành khi**:
- Đủ 6 tệp `.h5` + 2 `manifest.json` + `patches.csv` + `<ds>.json` nằm sẵn ở `DRIVE_ROOT/data/processed/{luna16,lidc}/` (khớp SHA-256; không phải chờ upload từ local).
- `test.h5` có đúng 1.200 mẫu mỗi bộ; mọi `patch_id` duy nhất kể cả 8 bệnh nhân có 2 scan; 0% rò rỉ bệnh nhân (assert).
- Đã kiểm tra bằng mắt 32 patch mỗi bộ (mask khớp nốt, `dist == 0` trong mask) và ghi bảng số lượng + phân bố vào `README.md`.

---

## Phase 6 — Train/evaluate & Smoke test (Colab Pro GPU)

| # | Việc | Spec |
|---|---|---|
| 6.1 | `src/train.py` (11 cờ tắt, `make_run_name`, `dataset` vào `config_resolved.yaml`, weighted epoch avg, `last.pt` atomic, `best.pt` weights-only, `--resume` cắt trùng `train_log.csv`, `--max-batches`, `--overfit-steps`) | §6.1 |
| 6.2 | `src/evaluate.py` (đọc `dataset` từ `config_resolved.yaml`, CSV `[0,1]`, JSON `%`, `preds.npz` float16 + `pred_bin` uint8) | §6.2 |
| 6.3 | `notebooks/10_colab_train.ipynb`, `11_colab_evaluate.ipynb` (đủ 100% tham số CLI ở cell đầu) | Phụ lục B |
| 6.4 | Smoke test: `RUN_PYTEST=True`, `SMOKE_ALL_CONFIGS=True` → chạy `pytest -q` rồi `--max-batches 2 --epochs 2` cho **cả 10 config** trên LUNA16 + evaluate | §6.3 |
| 6.4b | **Sanity check 1-batch overfit**: `caaf_hausdorff` / LUNA16 với `OVERFIT_STEPS=150` → Dice trên batch đạt **> 95%** | §6.3 |
| 6.5 | Resume check: train 3 epoch, ngắt runtime, chạy lại `RESUME=True` → tiếp tục epoch 4, `train_log.csv` không trùng dòng | §6.3 |
| 6.6 | Benchmark 1 epoch đủ dữ liệu trên GPU đang có → chốt `batch_size`, `amp`, và lập lịch Phase 8 | §6.3 |

---

## Phase 7 — Run kiểm chứng (Colab Pro GPU) — *cổng quyết định*

| # | Run | Kỳ vọng (bài báo) |
|---|---|---|
| 7.1 | `baseline_resunetpp` / LUNA16 / 100 epoch | Dice ≈ 83.89% |
| 7.2 | `caaf_hausdorff` / LUNA16 / 100 epoch | Dice ≈ 90.88%, w1 ≈ 0.26 |

**Cổng quyết định** (ngưỡng ±5 điểm % tự đặt): nếu 7.1 lệch > 5 điểm % hoặc đường loss bất thường → dừng lại kiểm tra dữ liệu và metric trước khi chạy Phase 8. Kiểm tra thêm `w` của 7.2 (rủi ro R5).

---

## Phase 8 — Chạy đủ thực nghiệm (Colab Pro GPU)

| Ưu tiên | Dataset | Config | Bảng |
|---|---|---|---|
| ✓ | LUNA16 | `baseline_resunetpp`, `caaf_hausdorff` | T5, T7 |
| 1 | LUNA16 | `caaf_sobel`, `caaf_laplacian` | T5 |
| 2 | LUNA16 | `pcam_sobel`, `pcam_laplacian`, `pcam_hausdorff` | T6 |
| 3 | LUNA16 | `pa_resunet`, `ca_resunet`, `pcam_dice` | T7 |
| 4 | LIDC | `caaf_hausdorff`, `caaf_sobel`, `caaf_laplacian` | T5 |
| 5 | LIDC | `pcam_dice` | T7 |
| 6 | LUNA16 | `caaf_hausdorff` seed 7, seed 123 | seed variance |
| 7 (tùy chọn) | LIDC | `pcam_sobel`, `pcam_laplacian`, `pcam_hausdorff` | T6 trên LIDC |

- **16 run bắt buộc + 3 tùy chọn** (100 epoch/run). Dùng `RUN_QUEUE` trong `10_colab_train.ipynb` để chạy nối tiếp nhiều run trong một phiên Colab Pro.
- Theo dõi ở `reports/runs.md`. **Không đổi mã giữa Phase 8**; nếu bắt buộc sửa bug thì gắn tag mới và chạy lại các run ảnh hưởng.

### Phase 8b — Thử nghiệm độ nhạy (tùy chọn)
| Thứ tự | ID | Nội dung | Chi phí |
|---|---|---|---|
| 1 | S2 | Evaluate thêm `best.pt` (`CKPTS=["last","best"]`) | 0 GPU train |
| 2 | S1 | `SPLIT_MODE="slice"` → `data/processed_s1_slice/luna16/`, chạy `caaf_hausdorff` | 1 run |
| 3 | S3 | `EXTRA_SET=["loss.hausdorff_mode=bidir_soft"]` (hoặc `karimi_sq`) | 1 run |
| 4 | S4 | AAF trong decoder (thêm mã) | 1 run |
| 5 | S5 | `center_jitter_px: 8` → `data/processed_s5_jitter/luna16/` | 1 run |

---

## Phase 9 — Thống kê, Hình ảnh & Báo cáo (Colab Pro GPU → tự mirror về Local)

| # | Việc | Spec |
|---|---|---|
| 9.1 | `src/stats_test.py`, `src/report.py` (`--ckpt`, `--seed`, `--include-sensitivity`, hằng số đầy đủ Table 5–7) | §9.1, §9.2 |
| 9.2 | `src/visualize.py` (`loss`, `dotplot`, `diffmap`, `gradcam` `--layer aaf1/2/3`, `weights`) | §9.3 |
| 9.3 | `notebooks/12_colab_stats_figures.ipynb` → sinh `reports/compare_table{5,6,7}.md`, `seed_variance.md`, `reports/figures/*` trên Drive (tự mirror về `D:\GG_1`) | Phụ lục B |
| 9.4 | Lọc bán tự động ứng viên từ `patches.csv` (`diameter_mm`, `mean_hu`, `min_hu`, `solidity`, `touches_border`) chọn 5–6 ca tương tự C1–C5 vẽ `diffmap` + `gradcam` | §5.1, §9.3 |
| 9.5 | Hoàn thiện `README.md`: cách chạy, bảng giả định `[A]/[F]`, số liệu thực tế, đánh giá tiêu chí (a)–(d), giới hạn MEF-PN35 | |

---

## Phụ thuộc giữa các phase

```
P0 ─► P1 (Colab CPU) ─► P5 (Colab CPU) ──────────┐
P0 ─► P2 ─► P3 (test GPU) ─► P4 (test GPU) ─► P6 ◄┘ ─► P7 ─► P8 ─► P9
```
*(Có thể viết mã P2–P4 ở local song song trong lúc Colab CPU High-RAM chạy P1 hoặc P5.)*

## Rủi ro & Giảm thiểu

| # | Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|---|
| R1 | Drive FUSE bị throttle hoặc Colab CPU ngắt giữa chừng khi đọc 244.000 file DICOM LIDC | Chậm/gián đoạn P5 LIDC | Lập `data/index/` 1 lần (chỉ đọc header); dùng `workers=2..4`; lưu marker `_DONE` từng `series_uid` để resume tức thì; LUNA16 không mở CT DICOM LIDC nên chạy rất nhanh |
| R2 | Số patch lệch nhiều so với Table 4 | Khó so sánh trực tiếp | Chốt `min_readers`/`min_mask_area_px` ở P1; ghi số thực tế vào README |
| R3 | Lệch/lật trục khi ánh xạ DICOM-SEG sang `.mhd` LUNA16 | Mask sai vị trí | Dùng `TransformPhysicalPointToContinuousIndex` + nhánh tổng quát `IOP`/`Direction`; kiểm chứng 100% 888 scan & vẽ chồng 5 scan ở P1.5 |
| R4 | Thiếu VRAM GPU do PAM 64×64 | OOM khi train | Bật `amp` hoặc giảm `batch_size`; đo ở P6.6 |
| R5 | AAC kẹt ở (0.5, 0.5) hoặc one-hot | Không tái lập w Table 5 | Kiểm tra gradient tới AAC ở P3 (`test_grad_flow`); báo cáo trung thực |
| R6 | EDT bằng scipy trên CPU làm chậm Hausdorff loss | Tăng thời gian train | Đo ở P4; nếu > 30% thời gian step thì cân nhắc tối ưu |
| R7 | Colab GPU ngắt phiên giữa 100 epoch | Mất tiến độ | `last.pt` ghi atomic mỗi epoch, `best.pt` chỉ lưu weights; `--resume` cắt trùng `train_log.csv`; dùng `RUN_QUEUE` |
| R8 | Kết quả lệch > 3 điểm % so với bài báo | Không đạt mục tiêu định lượng | Ưu tiên tiêu chí định tính (a)–(d); chạy S2 → S1 → S3 ở Phase 8b để giải thích |
| R9 | Phiên bản Python/torch trên Colab đổi giữa các run | Khó tái lập | Lưu `env.json` & `env_history.jsonl` cho mỗi run |

## Câu hỏi còn mở (không chặn Phase 0–1, cần xác nhận trước Phase 6)

1. ~~Vị trí dữ liệu và đồng bộ Drive~~ — **Đã xác minh trên đĩa**: dữ liệu thô nằm ngay trong `data/raw/LIDC-IDRI` (IDC DICOM-SEG) và `data/raw/LUNA16` (đã `.gitignore`, mirror Drive qua `D:\GG_1`); chạy khảo sát & tiền xử lý trực tiếp trên **Colab Pro CPU High-RAM**.
2. Đường dẫn thực tế của thư mục `CT208E` sau khi mount Google Drive trên Colab (`/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E` hay có thư mục gốc khác?).
3. URL repo GitHub private (`REPO_URL`) và loại GPU Colab Pro thường cấp (A100, L4 hay T4).
4. Có chạy Table 6 trên cả LIDC (thêm 3 run tùy chọn) không?
