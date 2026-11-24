# Review spec mã nguồn CAAF-ResUNet so với bài báo

Nguồn đối chiếu: `docs/MARKDOWN/caaf_resunet_markdown/paper.md` (Medicina 2025, 61, 1126) và
`caaf_unet_architecture.md` (Figure 1). Khung đối chiếu: docstring trong `src/**`, `configs/*.yaml`,
`make_structure.py`. **Hiện mọi tệp mã nguồn mới chỉ là docstring**, nên đây là review *spec* (thiết kế),
chưa phải review code.

Ký hiệu:
- ✅ khớp bài báo
- ⚠️ bài báo không nêu hoặc mơ hồ → phải tự giả định và ghi vào README
- ❌ khung hiện tại sai hoặc thiếu so với bài báo → cần sửa trước khi viết mã

---

## 1. Môi trường Python

| Nơi chạy | Dùng để | Python đề xuất |
|---|---|---|
| Colab Pro (GPU) | train, evaluate, Grad-CAM | Python mặc định của Colab (kiểm tra bằng `!python --version`, hiện là 3.12). **Không** tự cài Python khác trên Colab. |
| Máy local (i3-1005G1, 20 GB RAM, CPU) | khảo sát dữ liệu, tiền xử lý → `.h5`, viết mã | **Python 3.12**, cài song song với 3.14 trong một venv riêng cho dự án |

Lý do chọn 3.12 cho local:
- Trùng phiên bản với Colab, nên mã chạy được ở local thì chắc chắn chạy được trên Colab (không lỡ dùng cú pháp hay hành vi chỉ có ở 3.13/3.14).
- Các thư viện y tế (SimpleITK, pylidc, pydicom) có wheel ổn định cho 3.12; với 3.14 thì dễ thiếu wheel.
- `pylidc` là thư viện cũ (phụ thuộc SQLAlchemy, từng lỗi với numpy ≥ 1.24). Nếu vẫn lỗi trên 3.12, có thể bỏ pylidc
  và đọc trực tiếp XML của LIDC (`pydicom` + `xml.etree`).

Thư viện local tối thiểu cho bước khảo sát dữ liệu: `numpy pydicom SimpleITK matplotlib pandas h5py scipy` (+ `pylidc` nếu dùng).
`torch` **không bắt buộc** ở local. Nếu muốn chạy `pytest` cho mô hình ở local, chỉ cần bản torch CPU; các test dùng tensor
64×64 nhỏ nên chạy được trên i3.

> Gợi ý: tách `requirements.txt` thành `requirements-local.txt` (khảo sát/tiền xử lý, không có torch) và
> `requirements-colab.txt` (không ghim torch, vì Colab đã cài sẵn; ghim torch dễ kéo về bản không khớp CUDA).

---

## 2. Mô hình (`src/models/`)

### 2.1 Kích thước tensor (Section 2.1)

| Tầng | Bài báo | Ghi chú |
|---|---|---|
| Input | 1×64×64 | ✅ patch 64×64, 1 kênh CT |
| Encoder_ResBlock (F0) | 32×64×64 | ✅ |
| AAF1 → RCM1 | 32@64 → 64×32×32 | ✅ RCM giảm kích thước (stride 2) |
| AAF2 → RCM2 | 64@32 → 128×16×16 | ✅ |
| AAF3 → RCM3 | 128@16 → 256×8×8 | ✅ |
| ASPP bridge | 256×8×8 | ✅ |
| Decoder | 128@16 → 64@32 → 32@64 | ✅ |
| Output | 1×64×64, Sigmoid | ✅ |

`tests/test_shapes.py` (64→32→16→8→16→32→64, đầu ra ∈ [0,1]) ✅ đúng.

### 2.2 Từng khối

| Khối | Bài báo nói gì | Khung hiện tại | Đánh giá / quyết định đề xuất |
|---|---|---|---|
| **Encoder_ResBlock** | Conv3×3 → BN → ReLU → Conv3×3, cộng skip từ X | có | ✅ Nhánh skip: Conv3×3 + BN (1→32 kênh) như ResUNet++ ⚠️ |
| **Residual Conv Module** | Enhancement (BN→ReLU) + ResBlock (Conv→BN→ReLU→Conv) + residual | có | ✅ Theo kiểu pre-activation của ResUNet++: BN→ReLU→Conv(s=2)→BN→ReLU→Conv; shortcut Conv3×3(s=2)+BN ⚠️. Ở decoder dùng stride 1. |
| **AAC** (Alg. 1) | AdaptiveAvgPool 8×8 → Q,K,V (C×C) → softmax(QKᵀ/√C) → mean theo N → FC C→C/2 + ReLU → FC C/2→2 → softmax | "nhận F0: C=32 → 16 → 2" | ✅ Đúng Eq. 1–9. ⚠️ Section 2.2 nói AAC nhận "downsampled version of the input", nhưng nếu đầu vào là X (C=1) thì C/2 = 0, vô nghĩa. Vì vậy chọn **F0 (C=32)** là hợp lý; ghi rõ là giả định. Một cặp (w1, w2) **dùng chung cho cả 3 AAF** (Figure 1) ✅. Q,K,V có bias hay không: ⚠️ (đề xuất `bias=False`). |
| **PAM / CAM** | "DANet-style" [12], không cho chi tiết | "kiểu DANet" | ⚠️ Theo DANet gốc: PAM có Q,K giảm C/8, V giữ C, `out = γ·attn + x` (γ khởi tạo 0); CAM không có tham số ngoài γ. Ghi rõ là giả định. ⚠️ **Bộ nhớ**: PAM ở AAF1 chạy trên 64×64, tức N = 4096 → ma trận attention 4096×4096 ≈ 64 MB/mẫu (fp32). Batch 16 ≈ 1 GB chỉ riêng ma trận này, cộng thêm backward. Chạy được trên A100/L4; trên T4 cần batch nhỏ hơn hoặc AMP. |
| **AAF** | X = w1·PA(X) + w2·CA(X) (Eq. 12) → Conv3×3 (Figure 1) | mode `pa / ca / static / adaptive` | ✅ Lưu ý Eq. 12: **w1 nhân PA, w2 nhân CA** (Table 5: w2 ≈ 0.74–0.78 là trọng số của CA). |
| **ASPP** | Dilation 6/12/18, mỗi nhánh Conv3×3 → ReLU → BN; concat → Conv1×1. Text có nhắc thêm GAP; Figure 1 không vẽ GAP. | "6/12/18 + tùy chọn GAP" | ✅ Thứ tự **Conv → ReLU → BN** (Figure 1, khác thông lệ). GAP: ⚠️ để `aspp.gap: false` mặc định theo Figure 1, cho bật qua config. Lưu ý trên bản đồ 8×8, dilation 12 và 18 làm mọi tap ngoài tâm rơi vào padding, nên thực chất chỉ còn tác dụng như Conv1×1. Vẫn giữ vì đúng bài báo. |
| **Attention Module** | conv_Encoder: BN→ReLU→Conv3×3→MaxPool2; conv_Decoder: BN→ReLU→Conv3×3; cộng → conv_Attention: BN→ReLU→Conv(k=1, **p=1**) → nhân với feature decoder | "Conv 1×1 pad=0" | ✅ Khung đã sửa đúng lỗi trong hình: Conv k=1 với p=1 sẽ làm tăng kích thước lên 2 pixel, không nhân được. ⚠️ Số kênh ra của conv_Attention: theo ResUNet++ là **1 kênh** (broadcast khi nhân), không có sigmoid. |
| **Nguồn encoder của Attention Module ở bridge** | Figure 1 vẽ RCM3 → ATT1 | — | ❌ **Mâu thuẫn trong hình**: RCM3 là 8×8, qua MaxPool thành 4×4, không cộng được với bridge 8×8. Theo ResUNet++ (và theo cách ATT2 nhận RCM1, ATT3 nhận ER trong chính Figure 1), ATT1 phải nhận **RCM2 (16×16×128) → MaxPool → 8×8**. Cần ghi rõ trong mã và README. |
| **Decoder** | Lặp 3 lần: Attention → UpSample (ConvTranspose2d) → Concat skip → RCM | — | ✅ ConvTranspose2d k=2, s=2 ⚠️ (bài báo không cho kernel). Tầng cuối: RCM → ASPP → Conv1×1 → Sigmoid ✅. |
| **Đầu ra mô hình** | — | trả về `prob_mask, w1, w2` | ✅ Cần w1, w2 để báo cáo "Mean w1/w2" ở Table 5 và 7. Với mode không phải `adaptive`, trả về `None` hoặc hằng 0.5. |

### 2.3 Baseline và các biến thể ablation (Table 7)

| Config | Bài báo | Khung hiện tại | Đánh giá |
|---|---|---|---|
| `baseline_resunetpp` | ResUNet++ [11] | `aaf.mode: none` | ❌ **ResUNet++ gốc có khối Squeeze-and-Excitation (SE) ở vị trí mà CAAF đặt AAF**. Nếu `none` chỉ bỏ AAF thì đó không còn là ResUNet++. Đề xuất: `resunetpp.py` dựng đúng ResUNet++ (SE, reduction 8 hoặc 16 ⚠️) và chọn qua `model.name: resunetpp`, không qua `aaf.mode`. |
| `pa_resunet`, `ca_resunet` | "Position/Channel + ResUnet" | chỉ dựng 1 nhánh | ✅ (thay SE bằng PAM hoặc CAM; có hay không Conv3×3 sau đó: giữ như AAF ⚠️) |
| `pcam_*` | w1 = w2 = 0.5, không AAC | `static` | ✅ |
| `caaf_*` | đầy đủ | `adaptive` | ✅ |

---

## 3. Loss (`src/losses/`)

| Loss | Bài báo | Khung | Đánh giá |
|---|---|---|---|
| **Dice** (Eq. 13) | 1 − 2ΣPG / (ΣP + ΣG) trên **P mềm** (xác suất) | kèm smooth ε | ✅ ⚠️ Tính theo từng mẫu rồi lấy trung bình hay tính trên cả batch: chưa rõ → đề xuất **theo từng mẫu**. ε = 1e-6. |
| **Sobel** (Eq. 16–21) | Gradient magnitude của P và G, L1, **chia N pixel** (Eq. 21) | ε trong căn bậc hai | ✅ Eq. 19 dùng tổng, Eq. 21 dùng trung bình. Chọn **trung bình**: Figure 2a bắt đầu ở ~0.7, chỉ khớp nếu lấy trung bình (lấy tổng sẽ ra hàng chục). Padding 1 (zero). ε: `sqrt(gx² + gy² + 1e-8)`. |
| **Laplacian** (Eq. 23–28) | Kernel [[0,1,0],[1,−4,1],[0,1,0]], L1, trung bình | — | ✅ Trung bình (cùng lý do). Padding zero. |
| **Hausdorff-EDT** (Eq. 33–38) | D_G = EDT(1−G), D_P = EDT(1−P), d(P→G) + d(G→P). Alg. 5 ghi P **nhị phân** nhưng text nói "fully differentiable". | d(P→G) = Σ(P·D_G)/ΣP (khả vi); d(G→P) dùng D_P từ P nhị phân, detach | ✅ Đây là cách hợp lý duy nhất để vừa đúng công thức vừa có gradient: EDT của P nhị phân không khả vi. Ghi rõ giả định. Đơn vị **pixel**, không chuẩn hóa: khớp Figure 2c bắt đầu ~2.4. ❗ Ca biên: P rỗng → D_P không xác định → **cắt trần** d(G→P) = 64·√2 ≈ 90.5. G rỗng → bỏ qua số hạng đó. D_G được tính sẵn khi tiền xử lý (`compute_edt.py`) ✅. D_P phải tính lại mỗi bước bằng scipy trên CPU, chậm. Có thể thay bằng thuật toán EDT trên GPU (kornia) nếu quá chậm. |
| **BAL** (Eq. 14) | L = L_Dice + L_Boundary, **không có hệ số λ** | đọc từ config | ✅ Để `loss.lambda: 1.0` trong config nhưng mặc định = 1 cho đúng bài báo. |
| Loss của Table 7 | Không nêu | Giả định chỉ dùng Dice | ✅ Có bằng chứng ủng hộ: PCAM ở Table 7 (85.96) ≠ PCAM+Hausdorff ở Table 6 (88.40) ≠ PCAM+Sobel (87.46), nên Table 7 dùng loss khác, nhiều khả năng chỉ có Dice. |

`tests/test_losses.py`: "Loss = 0 khi P = G". Với Sobel và Laplacian thì đúng. Với Hausdorff thì d(P→G) = 0 khi P = G nhị phân ✅. Với Dice có ε thì ≈ 0, không bằng đúng 0, nên test cần dùng `atol`.

---

## 4. Metric và thống kê

| Mục | Bài báo | Khung | Đánh giá |
|---|---|---|---|
| Dice, IoU, Sensitivity, Miss Rate, Specificity (Eq. 39–43) | Từ TP/TN/FP/FN | ✅ | ✅ Tính **theo từng patch**, báo cáo mean ± std (Table 5). Miss Rate = 1 − Sensitivity, nên phải khớp đúng như Table 5. |
| Ngưỡng nhị phân | Không nêu | 0.5 | ⚠️ |
| Dự đoán rỗng | — | Dice = 0, không NaN | ✅ Nếu cả P và G cùng rỗng: Dice = 1 (quy ước, ghi rõ). |
| Paired t-test | Hai phía, ghép cặp theo ảnh, 1200 mẫu (Table 6) | ✅ | ✅ `scipy.stats.ttest_rel`. ⚠️ Bài báo tự mâu thuẫn: Section 2.6.3 nói Exp. 2 trên LIDC-IDRI, Table 6 ghi LUNA16. Khung chạy cả hai ✅. |
| Mean w1, w2 | Trung bình trên toàn bộ tập test | `per_patch_metrics.csv` kèm w1, w2 | ✅ |

---

## 5. Huấn luyện

| Siêu tham số | Bài báo | `base.yaml` hiện tại | Đánh giá |
|---|---|---|---|
| Optimizer | AdamW, lr 1e-4 | ✅ | ✅ |
| Weight decay | Không nêu | thiếu | ⚠️ thêm `weight_decay: 0.01` (mặc định của PyTorch) |
| Scheduler | StepLR(30, 0.1) | ✅ | ✅ |
| Epoch | 100 (Figure 2) | 100 | ✅ |
| Batch size | Không nêu | thiếu | ⚠️ thêm `batch_size: 16` (chỉnh theo GPU) |
| Seed | Không nêu | 42 | ⚠️ |
| Tập validation | **Không có** (Table 4 chỉ có train/test) | `best.pt` theo val Dice | ⚠️ Phải tách val từ train (theo patient, ví dụ 10%), nên số mẫu train sẽ < 13.928/9.800. **Đề xuất**: lưu cả `last.pt` và `best.pt`; báo cáo chính dùng `last.pt` (gần với bài báo hơn vì bài báo không có val), `best.pt` để tham khảo. |
| Augmentation | **Không nêu** | flip/rot đồng bộ cho ảnh, mask và D_G | ⚠️ Chỉ dùng phép biến đổi giữ nguyên khoảng cách: flip, rot90. Không dùng scale/elastic, vì D_G sẽ sai. Đề xuất `augment: false` mặc định cho đúng bài báo, bật thử như một biến thể. |
| AMP / precision | Không nêu (PyTorch 1.8, A100) | — | ⚠️ cho bật qua config. Riêng AAC/PAM nên tính softmax ở fp32. |
| Checkpoint mỗi epoch, `--resume` | — | ✅ | ✅ Bắt buộc trên Colab (bị ngắt phiên). Ghi checkpoint thẳng lên Drive. |

---

## 6. Dữ liệu (`src/data/`)

| Mục | Bài báo | Khung | Đánh giá |
|---|---|---|---|
| Số mẫu | LIDC 13.928 train / 1.200 test; LUNA16 9.800 / 1.200 | — | ⚠️ Chỉ cho số lượng, không cho cách chọn. Test **đúng 1.200** mẫu mỗi bộ. |
| Patch | 64×64 2D, mỗi patch chứa nốt (Table A1: "lung-nodule-containing patches") | 64×64 | ✅ Cắt theo slice, tâm patch là tâm nốt ⚠️. |
| Cửa sổ HU / chuẩn hóa | Không nêu | "lung window HU" | ⚠️ Đề xuất HU ∈ [−1000, 400] → [0, 1]. Đưa vào config. |
| Mask LIDC (tối đa 4 bác sĩ) | Không nêu | consensus mask | ⚠️ Đề xuất consensus 50% (pylidc `clevel=0.5`). |
| Chia tập | Không nêu | theo `patient_id`, cố định seed | ✅ Rất nên làm (tránh rò rỉ giữa các slice của cùng bệnh nhân). |
| LUNA16 ⊂ LIDC-IDRI | — | không train chéo | ✅ |
| Resample spacing | Không nêu | — | ⚠️ Đề xuất **không** resample (patch theo pixel gốc). Ghi rõ. |
| MEF-PN35 (Table 9, A1, 10–13) | Dữ liệu nội bộ, không công bố | — | ❌ Không phục dựng được. Ghi rõ trong README. Hình C1–C5 có thể thay bằng ca tương tự lấy từ LIDC. |
| Kích thước `.h5` | — | — | ~15k patch × (ảnh float32 16 KB + mask 4 KB + D_G float32 16 KB) ≈ **0.5–0.6 GB mỗi bộ**. Đủ nhỏ để tiền xử lý ở local rồi đưa lên Drive. |

---

## 7. Thực nghiệm cần chạy

| Bảng | Dataset | Config | Số run |
|---|---|---|---|
| Table 5 | LUNA16, LIDC | caaf_sobel/laplacian/hausdorff | 6 |
| Table 6 | LUNA16 (+ LIDC) | pcam_sobel/laplacian/hausdorff (đối chứng với caaf_* ở trên) | 3 (+3) |
| Table 7 | LUNA16 | baseline, pa, ca, pcam_dice (+ caaf_hausdorff đã có) | 4 |
| Table 7 | LIDC | pcam_dice (+ caaf_hausdorff đã có) | 1 |
| Seed | LUNA16 | caaf_hausdorff × 2 seed thêm | 2 |
| **Tổng** | | | **16 run tối thiểu, 19 nếu chạy Table 6 trên cả LIDC** |

Mỗi run 100 epoch. Thứ tự đề xuất: `baseline_resunetpp` (LUNA16) để kiểm chứng pipeline (mục tiêu ~84 Dice), rồi đến `caaf_hausdorff`, rồi các config còn lại.

---

## 8. Thiết kế chạy trên Colab

### 8.1 Nguyên tắc: tham số nằm ở đâu

| Loại tham số | Đặt ở đâu | Ví dụ |
|---|---|---|
| Siêu tham số mô hình / train / loss / tiền xử lý | **YAML trong `configs/`** (nằm trong git, là nguồn chuẩn duy nhất) | lr, epochs, batch_size, aaf.mode, loss, HU window |
| Tham số **của một lần chạy** (đường dẫn, dataset, seed, resume) | **CLI arg** (argparse trong `main()` của từng script) | `--dataset luna16 --seed 42 --resume` |
| Giá trị cụ thể cho phiên Colab | **Cell tham số đầu notebook** → ghép thành lệnh CLI | `CONFIG = "caaf_hausdorff"` |
| Ghi đè nhanh 1 siêu tham số (thử nghiệm) | `--set key=value` (lặp nhiều lần) | `--set train.batch_size=8` |
| Bí mật (GitHub token) | **Colab Secrets** (`google.colab.userdata`), không bao giờ ghi vào notebook | `GH_TOKEN` |

Luồng: **cell notebook → `!python -m src.train --config … --set …` → argparse → nạp YAML (base + config) → áp `--set` → lưu `config_resolved.yaml` vào thư mục run.**
Gọi bằng `!python -m …` (subprocess) thay vì `import` để mỗi lần chạy có GPU memory sạch, log rõ ràng và không bị dính trạng thái module cũ khi `git pull`.

Phần argparse dùng chung đặt trong `src/utils.py` (`add_common_args(parser)`, `load_config(args)`). Mỗi script chỉ thêm các tham số riêng của nó.

### 8.2 Danh sách CLI arg (tổng cộng **24 tham số**, trong đó 5 dùng chung)

> **Đã được thay thế** bởi `docs/PLANNING/SPECS.md` §11 (22 tham số khác nhau). Bảng dưới đây chỉ giữ lại để tham khảo lịch sử.

**Dùng chung** (`src/utils.py::add_common_args`) cho `train.py` và `evaluate.py`: 5 tham số

| Arg | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `--config` | path | bắt buộc | `configs/caaf_hausdorff.yaml` (tự kế thừa `base.yaml`) |
| `--dataset` | `luna16` \| `lidc` | bắt buộc | chọn bộ dữ liệu |
| `--data-dir` | path | `data/processed` | thư mục chứa `{dataset}/train.h5, val.h5, test.h5` (trên Colab: `/content/processed`) |
| `--out-dir` | path | `experiments` | gốc lưu kết quả (trên Colab: thư mục trên Drive) |
| `--set` | `key=value`, lặp | — | ghi đè YAML, ví dụ `--set train.batch_size=8` |

`seed` để trong YAML và đổi qua `--set seed=7`, không tách thành arg riêng.

**`src/train.py`**: 5 chung + 3 riêng = 8

| Arg | Ý nghĩa |
|---|---|
| `--resume` | tiếp tục từ `last.pt` trong thư mục run |
| `--run-name` | tên thư mục run; mặc định `{config}_{dataset}_s{seed}` (cố định để `--resume` tìm đúng) |
| `--max-batches` | giới hạn số batch mỗi epoch, dùng chạy thử nhanh (smoke test) |

**`src/evaluate.py`**: 5 chung + 2 riêng = 7

| Arg | Ý nghĩa |
|---|---|
| `--run-dir` | thư mục run (đọc `config_resolved.yaml`, checkpoint) |
| `--ckpt` | `last` \| `best` |

**`src/stats_test.py`**: 3

| Arg | Ý nghĩa |
|---|---|
| `--a`, `--b` | hai tệp `per_patch_metrics.csv` (ví dụ caaf vs pcam) |
| `--out` | tệp kết quả (t, p, mean ± std) |

**`src/visualize.py`**: 3

| Arg | Ý nghĩa |
|---|---|
| `--run-dirs` | một hoặc nhiều thư mục run |
| `--kind` | `loss` \| `dotplot` \| `diffmap` \| `gradcam` |
| `--out-dir` | nơi lưu hình |

**Tiền xử lý** (`preprocess_*.py`, `compute_edt.py`, `pack_h5.py`, `make_splits.py`): dùng chung 3 tham số, chạy ở **local**

| Arg | Ý nghĩa |
|---|---|
| `--raw-dir` | `data/raw/LIDC-IDRI` hoặc `data/raw/LUNA16` |
| `--out-dir` | `data/processed/{dataset}` |
| `--config` | `configs/data.yaml` (HU window, clevel, patch size, tỉ lệ val, seed chia tập) |

Tổng: 5 chung + 3 (train) + 2 (evaluate) + 3 (stats) + 3 (visualize) + 3 (tiền xử lý) = **19 tham số khác nhau**.
Tính theo từng script (đếm lặp các tham số chung) thì là 8 + 7 + 3 + 3 + 3 = 24.

### 8.3 Notebook đặt ở đâu

Tạo thư mục `notebooks/` ở gốc repo (chuyển `scripts/colab_setup.ipynb` vào đây). Mỗi notebook **tự đủ**: cell đầu là setup, vì mỗi phiên Colab bắt đầu trắng.

```
notebooks/
├── 00_local_data_survey.ipynb      # LOCAL (CPU): khảo sát LIDC/LUNA16, thống kê kích thước nốt, HU
├── 01_local_preprocess.ipynb       # LOCAL (CPU): gọi src/data/* → .h5, rồi upload .h5 lên Drive
├── 10_colab_train.ipynb            # COLAB: 1 phiên = 1 run (chọn CONFIG/DATASET/SEED ở cell tham số)
├── 11_colab_evaluate.ipynb         # COLAB: evaluate mọi run → per_patch_metrics.csv, results.json
└── 12_colab_stats_figures.ipynb    # COLAB hoặc LOCAL: t-test, Figure 2/3, diffmap, Grad-CAM, điền reports/
```

Khung cell của `10_colab_train.ipynb`:

1. **Tham số** (chỉ sửa cell này): `REPO_URL`, `BRANCH`, `CONFIG`, `DATASET`, `SEED`, `RESUME`, `EXTRA_SET = []`, `DRIVE_ROOT`.
2. **Setup**: mount Drive → `git clone`/`git pull` (token lấy từ Colab Secrets nếu repo private) → `pip install -r requirements-colab.txt` → `nvidia-smi`.
3. **Dữ liệu**: copy `.h5` từ Drive về `/content/processed` (đọc ở ổ cục bộ nhanh hơn đọc trực tiếp từ Drive).
4. **Smoke test** (tùy chọn): `!python -m src.train … --max-batches 2 --set train.epochs=1`.
5. **Train**: `!python -m src.train --config configs/{CONFIG}.yaml --dataset {DATASET} --data-dir /content/processed --out-dir {DRIVE_ROOT}/experiments --set seed={SEED} {"--resume" if RESUME else ""}`.
6. **Ghi lại** commit hash (`git rev-parse HEAD`) vào thư mục run để truy vết được mã đã dùng.

Kết quả trên Drive: `experiments/{config}_{dataset}_s{seed}/` gồm `config_resolved.yaml`, `last.pt`, `best.pt`, `train_log.csv`, `commit.txt`. Sau evaluate có thêm `per_patch_metrics.csv` và `results.json`.

`scripts/*.sh` vẫn giữ để chạy hàng loạt, nhưng trên Colab sẽ gọi qua notebook (thường mỗi phiên chỉ đủ thời gian cho 1–2 run).

---

## 9. Việc cần sửa trong khung trước khi viết mã

1. ❌ `baseline_resunetpp.yaml`: chọn mô hình qua `model.name: resunetpp` (có SE), không qua `aaf.mode: none`.
2. ❌ Ghi chú trong `caaf_resunet.py`: ATT1 nhận **RCM2** (16×16 → MaxPool → 8×8), không phải RCM3.
3. ⚠️ `base.yaml` bổ sung: `model.channels: [32, 64, 128, 256]`, `train.batch_size`, `train.weight_decay`, `train.amp`,
   `train.augment: false`, `data.val_ratio`, `aspp.gap: false`, `loss.lambda: 1.0`, `loss.reduction: mean`,
   `attention.pam_reduction: 8`, `aac.pool_size: 8`, `eval.ckpt: last`.
4. ⚠️ Thêm `configs/data.yaml` cho tiền xử lý (HU window, clevel, patch size, seed chia tập).
5. Tách `requirements-local.txt` và `requirements-colab.txt`.
6. Tạo `notebooks/` và chuyển `scripts/colab_setup.ipynb` vào.
7. README: liệt kê mọi mục ⚠️ ở trên thành danh sách "Giả định phục dựng", và ghi MEF-PN35 không có.
