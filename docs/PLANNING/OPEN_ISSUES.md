# OPEN_ISSUES — Sổ đăng ký các điểm mơ hồ cần phục dựng

| Mục | Giá trị |
|---|---|
| Phạm vi | Mọi chỗ bài báo **không nêu**, **nêu mơ hồ** hoặc **tự mâu thuẫn**, kèm lựa chọn hiện tại trong `SPECS.md` và mức độ đã chốt |
| Phiên bản | **v2.1 — 2026-11-21** (đồng bộ `SPECS.md` v2.1 & `PLAN.md` v2.1: dữ liệu thô đặt tại `data/raw/{LIDC-IDRI,LUNA16}` chuẩn IDC DICOM-SEG không cần `pylidc`; chạy khảo sát & tiền xử lý trực tiếp trên Colab Pro CPU High-RAM) |

Ký hiệu:

**Ảnh hưởng tới kết quả**
- **Cao**: có thể làm Dice lệch vài điểm %, hoặc đổi kết luận của bài báo.
- **TB**: ảnh hưởng nhỏ, hoặc chỉ ảnh hưởng tới tốc độ/bộ nhớ.
- **Thấp**: chỉ là hình thức.

**Cách chốt**
- **E** (Evidence): suy ra từ số liệu hoặc hình trong bài báo.
- **X** (eXperiment): chạy thử để đo độ nhạy.
- **D** (Decide): tự quyết, ghi vào README.
- **P1 / P6 / P7**: chốt ở phase tương ứng của PLAN.

**Trạng thái**
- ✅ đã chốt trong SPECS
- 🟡 đã chọn tạm, cần xác nhận bằng dữ liệu hoặc thử nghiệm
- ❌ **chưa có trong SPECS**

---

## A. Bài báo tự mâu thuẫn

| ID | Vị trí | Mâu thuẫn | SPECS chọn | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|---|
| A1 | Figure 1 vs Mermaid | ~~Figure 1 vẽ RCM3 → ATT1~~ — Soi ảnh gốc `fig1_caaf_unet_architecture.png` thấy mũi tên tới `ATT1` **đã xuất phát từ `RCM2`** (16×16); lỗi chỉ ở Mermaid `caaf_unet_architecture.md` (đã sửa) | ATT1 nhận **RCM2** (16→8), khớp Figure 1 gốc và ResUNet++ | TB | ✅ |
| A2 | §2.1 | "Decoder fuses bridge output (8×8×256) with the deepest encoder features (8×8×256)" — mâu thuẫn với Figure 1 (`ATT1` dùng MaxPool nhận `RCM2`) | như A1 (theo Figure 1) | TB | ✅ |
| A3 | **§2.2 và §2.1** | §2.2: decoder dùng *"AAF-based upsampling blocks"*; §2.1: *"refining them through the AF module"*. Nhưng Figure 1 vẽ **Attention Module kiểu ResUNet++**, không phải AAF | Theo **Figure 1**: Attention Module ở decoder, AAF chỉ có ở encoder | **Cao** (đổi kiến trúc decoder) | ✅ SPECS §3.6 (biến thể khác để cho S4) |
| A4 | Figure 1 §2 | conv_Attention vẽ "Conv2D(k=1, **p=1**)" → output lệch 2 pixel | `p=0` | Thấp | ✅ |
| A5 | §2.2 và Alg. 1 | §2.2 nói AAC nhận "downsampled version of the **input**" (C=1 → FC C/2 = 0). Alg. 1 nhận feature map có C kênh | AAC nhận **F0** (C=32) | TB | ✅ |
| A6 | Eq. 19 và Eq. 21 (cũng Eq. 25 và 28) | Sobel/Laplacian loss: lấy **tổng** (Eq. 19/25) hay **trung bình** (Eq. 21/28) | Trung bình. **E**: Fig. 2a/b bắt đầu ~0.7 | TB | ✅ |
| A7 | Eq. 16–17 và Eq. 30 | Hai bộ kernel Sobel trái dấu nhau | Không ảnh hưởng (chỉ dùng độ lớn gradient) | Thấp | ✅ |
| A8 | §2.5.4 | Gọi là "Hausdorff, penalizes the **largest** deviation" (Eq. 29 dùng max), nhưng công thức thật (Eq. 33) là **trung bình** khoảng cách (~ASSD) | Theo Eq. 33–38 (trung bình) | TB | ✅ — ghi trong README |
| A9 | Alg. 5 và text | Alg. 5 cho P **nhị phân**, text nói *"fully differentiable"*. Nếu `d_GP` detach hoàn toàn thì Hausdorff chỉ phạt FP (không phạt FN) | Mặc định `hausdorff_mode=soft_pg_detach_gp` (log riêng `d_pg`, `d_gp`, `empty_pred_frac`). **X (S3)**: so với 2 chiều khả vi (`bidir_soft` / `karimi_sq`) | **Cao** | 🟡 — **X** (S3) |
| A10 | §2.6.3 và Table 6 | Exp. 2 ghi "LIDC-IDRI", nhưng Table 6 ghi **LUNA16**, 1.200 ca | Chạy LUNA16 (bắt buộc), LIDC (tùy chọn) | TB | ✅ |
| A11 | §4.1 | "Greatest Dice of 90.88% and lowest Miss Rate of 7.62%" — 90.88 của LUNA16, 7.62 của LIDC | Lấy số trong bảng làm chuẩn | Thấp | ✅ |
| A12 | §2.1 và Fig. 1 | ASPP: text có nhắc **GAP**, hình không vẽ | `gap: false`, cho bật qua config | TB | 🟡 — **X** nếu dư GPU |
| A13 | Đánh số bảng | §3.4.1 gọi bảng MEF là Table 8 (thật ra Table 9); §4.3 gọi Table 7 (thật ra Table 8) | Bỏ qua | Thấp | ✅ |

## B. Dữ liệu — bài báo không tiết lộ

| ID | Hạng mục | Bài báo nói | SPECS chọn | Ảnh hưởng | Cách chốt | Trạng thái |
|---|---|---|---|---|---|---|
| B1 | **Cách chia train/test** | Chỉ có số lượng (Table 4), không nói chia theo bệnh nhân hay theo slice | Theo **bệnh nhân** (0% rò rỉ) | **Cao**: chia theo slice có thể tăng Dice vài điểm do rò rỉ | **X** (S1 ở mục H) | 🟡 |
| B2 | Tập validation | Không có | 10% bệnh nhân lấy từ phần còn lại sau khi tách test | TB | D | ✅ |
| B3 | Chọn **nốt** nào (LIDC) | Không nói | Gom cụm DICOM-SEG theo `Nodule <K>`, `min_readers` (chốt ở P1), `diameter_mm ≥ 3.0` | **Cao** | **E** P1 (khớp gần 15.128 patch) | 🟡 |
| B4 | Gộp 4 bác sĩ thành một mask | Không nói | Consensus 50%: `votes >= ceil(0.5 * n_readers)` | **Cao** | D (có thể **X**: 25/50/75%) | 🟡 |
| B5 | Mask của LUNA16 | LUNA16 không có mask | Lấy trực tiếp từ **DICOM-SEG trong `data/raw/LIDC-IDRI`** khớp `ReferencedSeriesSequence` + tọa độ thế giới với `data/raw/LUNA16/annotations.csv` (không cần `pylidc`, không mở CT DICOM LIDC) | **Cao** | **E** P1 (kiểm tra khớp 1.186 nốt & tổng gần 11.000 patch) | 🟡 |
| B6 | Patch: lấy slice nào | "nodule-containing patches" | Mọi slice có `area_nodule_px ≥ min_mask_area_px` | TB | **E** P1.4 (thử {1, 5, 10}) | 🟡 |
| B7 | **Tâm patch** | Không nói | Trọng tâm 3D của mask nốt, cố định cho mọi slice | **Cao** | D (có thể **X**: S5 jitter ±8 px) | 🟡 |
| B8 | Nốt khác lọt vào patch | Không nói | Gộp mask mọi nốt đạt chuẩn (`include_other_nodules: true`) | TB | D + đếm tỉ lệ ở P1.6 | 🟡 |
| B9 | Cửa sổ HU / chuẩn hóa | Không nói | `[-1000, 400] → [0, 1]`, đối chiếu `[-1000, 150]` ở P1.7 | TB | **E** P1.7 | 🟡 |
| B10 | Resample spacing | Không nói | `resample_mm: null` (giữ spacing gốc) | TB-Cao | D + thống kê spacing ở P1.7 | 🟡 |
| B11 | Augmentation | Không nói | Tắt mặc định; hỗ trợ D4 (giữ nguyên EDT) qua cờ `--augment` | TB | D | ✅ |
| B12 | Chọn đúng 1.200 patch test | Không nói | Lấy theo bệnh nhân; phần dư của bệnh nhân ranh giới đưa vào `dropped` | Thấp | D | ✅ |
| B13 | Số train thực tế (13.928 / 9.800) | Không rõ có gồm val hay trừ dropped | Báo cáo số thực tế trong README | Thấp | D | ✅ |
| B14 | MEF-PN35 | Dữ liệu nội bộ | Không phục dựng được | — | — | ✅ |
| B15 | Ánh xạ hệ tọa độ & `Direction`/`IOP` giữa DICOM-SEG và `.mhd` LUNA16 | Không nói | `TransformPhysicalPointToContinuousIndex` + nhánh tổng quát `world_to_voxel_batch`; kiểm chứng trên **toàn bộ 888 scan / 1.186 nốt** và vẽ chồng 5 scan ở P1.5 | **Cao** | **E** P1.5 | 🟡 |
| B16 | Bệnh nhân LIDC có >1 scan CT | Thực tế kiểm chứng có **8/1.010 bệnh nhân có 2 `CT_*` scan** (tổng 1.018 scan) | Key thư mục interim theo `series_uid`; liên kết SEG → CT qua `ReferencedSeriesSequence`; `patch_id` chứa `s{scan_idx:02d}` | TB | D | ✅ SPECS §5.2 |
| B17 | Nốt lớn vượt 64×64 (`touches_border`) | Không nói | Ghi cột `touches_border`, thống kê tỉ lệ ở P1.8 và ghi README | TB | **E** P1.8 | 🟡 |

## C. Kiến trúc — bài báo không tiết lộ

| ID | Hạng mục | Bài báo nói | SPECS chọn | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|---|
| C1 | Cấu trúc PAM/CAM | "Dual-attention", trích DANet [12] | DANet gốc: PAM `Q,K = C/8`; `γ` khởi tạo 0, có residual | TB | ✅ |
| C2 | AAF sau phép trộn | Fig. 1: chỉ có Conv3×3 | Chỉ Conv3×3, không BN/ReLU | Thấp | ✅ |
| C3 | Q,K,V của AAC có bias không | Eq. 3 không có bias | `bias=False` | Thấp | ✅ |
| C4 | Stride / downsampling | "while downsampling" | Conv stride 2 trong RCM, shortcut Conv3×3 stride 2 + BN | TB | ✅ |
| C5 | Upsample | ConvTranspose2d | `k=2, s=2` | Thấp | ✅ |
| C6 | Kênh & kích hoạt `conv_Attention` | Không nói | 1 kênh, không sigmoid mặc định (ResUNet++) | TB | ✅ |
| C7 | Skip lấy trước hay sau AAF | Fig. 1: "Residual Combination" | Trước AAF | TB | ✅ |
| C8 | **Baseline ResUNet++** | Chỉ trích [11] | SE (reduction 8), **dùng chung decoder ConvTranspose** với CAAF | TB | 🟡 — ghi README |
| C9 | "Position/Channel + ResUnet" | Chỉ có tên | Thay SE bằng PAM/CAM + Conv3×3 | TB | ✅ |
| C10 | Khởi tạo trọng số, BN momentum | Không nói | Mặc định của PyTorch | Thấp | ✅ |
| C11 | Dropout | Không nói | Không dùng | Thấp | ✅ |
| C12 | Số kênh ra của `UpSample` | §2.1 ghi 16×16×128 | Giữ nguyên `ch → ch` (`256→256`) theo ResUNet++ | Thấp-TB | ✅ |
| C13 | `AttentionModule` decoder không có Sigmoid | Fig. 1 không vẽ Sigmoid | Mặc định `att_sigmoid: false`, có cờ config để bật nếu cần | TB | ✅ |
| C14 | Bão hòa Softmax `CAM` ở 64×64 (`N=4096`) | Không nêu chi tiết | **Đã đo trong `test_grad_flow.py`**: xác suất lớn nhất trung bình của softmax là `~0.999` (gần one-hot do `N=4096`), nhưng tỉ lệ gradient qua nhánh attention / nhánh value vẫn đạt `0.84` (nhánh attention vẫn học được, không bị triệt tiêu). Ghi nhận lại vào README; không tự ý đổi công thức DANet gốc | TB | ✅ đã đo ở Đợt 5 |

## D. Loss — bài báo không tiết lộ

| ID | Hạng mục | SPECS chọn | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|
| D1 | Dice theo mẫu hay theo batch; ε | Theo mẫu, `eps = 1e-6` | TB | ✅ |
| D2 | Hệ số λ trong BAL | `1.0` (Eq. 14 không có hệ số) | TB | ✅ |
| D3 | Padding Sobel/Laplacian | Zero padding (`padding=1`) | Thấp | ✅ |
| D4 | Hausdorff: `D_P` khi `P_b` rỗng | Cắt trần `64·√2` (`90.51`), log riêng `d_pg`, `d_gp`, `empty_pred_frac` | TB | ✅ |
| D5 | Hausdorff: đơn vị khoảng cách | Pixel (**E**: Fig. 2c bắt đầu ~2.4) | TB | ✅ |
| D6 | **Loss của Table 7** (baseline, PA, CA, PCAM) | Chỉ Dice (**E**: PCAM T7 = 85.96 khác mọi PCAM T6) | **Cao** | ✅ |

## E. Huấn luyện — bài báo không tiết lộ

| ID | Hạng mục | Bài báo nói | SPECS chọn | Ảnh hưởng | Cách chốt | Trạng thái |
|---|---|---|---|---|---|---|
| E1 | Batch size | Không | `16` | TB | P6.6 (VRAM) | 🟡 |
| E2 | Weight decay, betas | "AdamW" | `0.01`, `(0.9, 0.999)` | Thấp-TB | D | ✅ |
| E3 | **Checkpoint báo cáo** | Không có val; 100 epoch | `last` (chính), `best` (tham khảo S2) | TB-Cao | **X** S2 | 🟡 |
| E4 | Seed, số lần chạy | std là giữa các patch | `42` + 2 seed phụ (`7`, `123`) cho `caaf_hausdorff` | TB | D | ✅ |
| E5 | Mixed precision | Không | `amp: false` mặc định | Thấp | P6.6 | 🟡 |
| E6 | "Training loss" ở Fig. 2 | Không nói rõ | Tổng loss, trung bình có trọng số theo batch trong epoch | Thấp | D | ✅ |
| E7 | DataLoader | Không | `shuffle=True`, `drop_last=False` | Thấp | D | ✅ |

## F. Đánh giá — bài báo không tiết lộ

| ID | Hạng mục | SPECS chọn | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|
| F1 | Ngưỡng nhị phân | `0.5` | TB | ✅ |
| F2 | Metric theo patch hay gộp toàn tập | Theo patch (**E**: std ±16–22 & Fig. 3 theo patch) | TB | ✅ |
| F3 | `std` dùng `ddof` | `ddof=1` | Thấp | ✅ |
| F4 | Patch dự đoán rỗng | Dice = 0 (giữ trong trung bình) | TB | ✅ |
| F5 | "Mean w1/w2" | Trung bình theo mẫu trên 1.200 patch test | Thấp | ✅ |
| F6 | Grad-CAM: lớp và mục tiêu | Chọn `--layer {aaf1,aaf2,aaf3}` (mặc định `aaf1`), mục tiêu `Σ(logits · pred_bin)` (fallback `Σ(logits · prob)` khi rỗng) | Thấp | 🟡 |

## G. Các vấn đề kỹ thuật / vận hành đã được giải quyết trong SPECS v2.0

| ID | Điểm | Giải quyết trong SPECS v2.0 / PLAN v2.0 | Trạng thái |
|---|---|---|---|
| G1–G6, G8 | Cờ tắt CLI, `ct_utils`, A3, B6/B8, C10/C11, ngưỡng ±3/±5, SE reduction=8 | Đã hoàn thiện từ v1.1 | ✅ |
| G7 | Câu hỏi mở còn lại (xác nhận đường dẫn Drive sau khi mount trên Colab, `REPO_URL`, loại GPU Colab Pro, Table 6 trên LIDC) | PLAN cuối tài liệu (không chặn P0–P1) | 🟡 cần xác nhận trước P6 |
| G9–G14 | Lưu `dataset` vào `config_resolved.yaml`, đơn vị CSV `[0,1]` / JSON `%`, `preds.npz` float16, PyTorch 2.6+ (`weights_only=False`), `best.pt` weights-only, cắt trùng `train_log.csv`, `manifest.json` SHA-256, 100% CLI trên notebook | Đã tích hợp đầy đủ từ v1.2 | ✅ |
| G15 | **Thực tế dữ liệu trên đĩa & bỏ `pylidc`** | Dữ liệu đặt tại `data/raw/LIDC-IDRI` (bản NCI IDC DICOM-SEG `CT_*` + `SEG_*`) và `data/raw/LUNA16` (`subset0..9` + `annotations.csv`). Chuyển sang đọc trực tiếp `pydicom` + `SimpleITK`, loại bỏ hoàn toàn `pylidc` / `SQLAlchemy<2` / `numpy<2` | ✅ (v2.1) |
| G16 | **Luồng xử lý dữ liệu Colab-first & tổ chức SPECS theo Phase** | Tổ chức lại `SPECS.md` theo đúng **Phase 0 → Phase 9** với pseudocode thuật toán rõ ràng cho LIDC, LUNA16, `make_splits`, model, loss, train/eval/viz; chuyển Phase 1 & Phase 5 lên **Colab Pro CPU High-RAM** xử lý trực tiếp trên Drive (ghi `.h5` ở `/content/` rồi copy 1 lần sang Drive → 0 giờ upload từ local) | ✅ (v2.0) |

## H. Thử nghiệm độ nhạy đề xuất (Phase 8b — tùy chọn)

Chỉ chạy nếu kết quả Phase 7/8 lệch nhiều hoặc còn dư compute. Mỗi thử nghiệm là 1 run trên LUNA16 với `caaf_hausdorff`.

| ID | Điểm mơ hồ | Thử nghiệm | Chi phí |
|---|---|---|---|
| S1 | B1 (cách chia) | `SPLIT_MODE="slice"` → `data/processed_s1_slice/luna16/` | 1 run + 1 bộ `.h5` |
| S2 | E3 (checkpoint) | Evaluate cả `last.pt` và `best.pt` (`CKPTS=["last", "best"]`) | 0 GPU train |
| S3 | A9 (gradient Hausdorff) | `loss.hausdorff_mode=bidir_soft` (hoặc `karimi_sq`, phạt cả FN) | 1 run |
| S4 | A3 (decoder) | Thay Attention Module ở decoder bằng AAF | 1 run + thêm mã |
| S5 | B7 (tâm patch) | `center_jitter_px: 8` → `data/processed_s5_jitter/luna16/` | 1 run + 1 bộ `.h5` |

Nếu kết quả bám sát bài báo ở Phase 7 thì có thể bỏ toàn bộ mục H. Nếu lệch nhiều thì ưu tiên S2 → S1 → S3.
