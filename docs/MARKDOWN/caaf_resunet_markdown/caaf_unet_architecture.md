# CAAF-UNet Architecture (Figure 1) — dạng Markdown

Sơ đồ vẽ lại bằng Mermaid (GitHub, VS Code, Obsidian, Typora đều render được).
Chú thích: ⊕ = cộng từng phần tử, ⊗ = nhân từng phần tử, PAM = Position Attention Module, CAM = Channel Attention Module.

## 1. Sơ đồ tổng thể

```mermaid
flowchart TB
    X["Image X"]:::input
    AAC["Adaptive Attention Controller (AAC)"]:::aac

    subgraph ENC["ENCODER"]
        direction TB
        ER["Encoder_ResBlock<br/>Conv2D(k=3,p=1) → BatchNorm → ReLU → Conv2D(k=3,p=1) → Residual Combination"]:::res
        AAF1["Adaptive Attention Fusion 1<br/>w1×PAM ⊕ w2×CAM → Conv2D(k=3,p=1)"]:::aaf
        RCM1["Residual Convolution Module 1<br/>Enhancement Block (BN → ReLU) + ResBlock (Conv→BN→ReLU→Conv) → Residual Combination"]:::rcm
        AAF2["Adaptive Attention Fusion 2"]:::aaf
        RCM2["Residual Convolution Module 2"]:::rcm
        AAF3["Adaptive Attention Fusion 3"]:::aaf
        RCM3["Residual Convolution Module 3"]:::rcm
        ER --> AAF1 --> RCM1 --> AAF2 --> RCM2 --> AAF3 --> RCM3
    end

    subgraph BR["ASPP BRIDGE"]
        direction TB
        A1["ASPP_block1<br/>Conv2D(k=3,pad=6) → ReLU → BN"]:::aspp
        A2["ASPP_block2<br/>Conv2D(k=3,pad=12) → ReLU → BN"]:::aspp
        A3["ASPP_block3<br/>Conv2D(k=3,pad=18) → ReLU → BN"]:::aspp
        CAT["Concatenate"]:::cat
        C11["Conv2D (1×1)"]:::conv
        A1 --> CAT
        A2 --> CAT
        A3 --> CAT
        CAT --> C11
    end

    subgraph DEC["DECODER"]
        direction TB
        UP1["UpSample (ConvTranspose2d)"]:::up
        CC1["Concatenate (dim=1)"]:::cat
        DR1["Residual Convolution Module"]:::rcm
        ATT2["Attention Module"]:::att
        UP2["UpSample (ConvTranspose2d)"]:::up
        CC2["Concatenate (dim=1)"]:::cat
        DR2["Residual Convolution Module"]:::rcm
        ATT3["Attention Module"]:::att
        UP3["UpSample (ConvTranspose2d)"]:::up
        CC3["Concatenate (dim=1)"]:::cat
        DR3["Residual Convolution Module"]:::rcm
        DASPP["ASPP Module"]:::aspp
        CF["Conv2D (kernel_size=1)"]:::conv
        SIG["Sigmoid"]:::act
        OUT["OUTPUT (mask)"]:::input
        UP1 --> CC1 --> DR1 --> ATT2 --> UP2 --> CC2 --> DR2 --> ATT3 --> UP3 --> CC3 --> DR3 --> DASPP --> CF --> SIG --> OUT
    end

    ATT1["Attention Module (bridge)"]:::att

    %% Input
    X --> ER
    X -. "skip connection" .-> ER
    X --> AAC
    AAC -. "w1, w2" .-> AAF1
    AAC -. "w1, w2" .-> AAF2
    AAC -. "w1, w2" .-> AAF3

    %% Encoder -> bridge
    RCM3 --> A1
    RCM3 --> A2
    RCM3 --> A3
    C11 --> ATT1
    RCM2 -. "conv_Encoder" .-> ATT1
    ATT1 --> UP1

    %% Skip connections
    RCM2 -- "skip" --> CC1
    RCM1 -- "skip (Residual Combination)" --> ATT2
    RCM1 -- "skip (Residual Combination)" --> CC2
    ER -- "skip (Residual Combination)" --> ATT3
    ER -- "skip (Residual Combination)" --> CC3

    classDef input fill:#b0001c,color:#fff,stroke:#700;
    classDef aac fill:#22dd22,color:#000,stroke:#080;
    classDef res fill:#ffe0d0,stroke:#c96;
    classDef aaf fill:#f8c8c8,stroke:#c66;
    classDef rcm fill:#ffff66,stroke:#aa0;
    classDef aspp fill:#d5e8d4,stroke:#082;
    classDef cat fill:#f4a6a6,stroke:#c44;
    classDef up fill:#5f7186,color:#fff,stroke:#345;
    classDef att fill:#5f8a5f,color:#fff,stroke:#264;
    classDef conv fill:#ffe6cc,stroke:#c96;
    classDef act fill:#d5e8d4,stroke:#6a6;
```

> Lưu ý: trong hình gốc, các mũi tên skip từ encoder sang decoder chạy nhiều đường dài; sơ đồ trên nối đúng từng điểm nguồn (Residual Combination của từng tầng encoder) tới điểm đích (Attention Module hoặc Concatenate) của decoder.

## 2. Chi tiết Attention Module

```mermaid
flowchart TB
    ENCIN["Encoder feature (skip)"]:::input
    DECIN["Decoder / bridge feature"]:::input

    subgraph AM["Attention Module"]
        direction TB
        subgraph CE["conv_Encoder"]
            direction TB
            E1["BatchNorm"] --> E2["ReLU"] --> E3["Conv2D (k=3, p=1)"] --> E4["MaxPooling (2,2)"]
        end
        subgraph CD["conv_Decoder"]
            direction TB
            D1["BatchNorm"] --> D2["ReLU"] --> D3["Conv2D (k=3, p=1)"]
        end
        ADD(("⊕"))
        subgraph CA["conv_Attention"]
            direction TB
            T1["BatchNorm"] --> T2["ReLU"] --> T3["Conv2D (k=1, p=1)"]
        end
        MUL(("⊗"))
        E4 --> ADD
        D3 --> ADD
        ADD --> T1
        T3 --> MUL
    end

    ENCIN --> E1
    DECIN --> D1
    DECIN -- "feature gốc" --> MUL
    MUL --> OUTF["Output → UpSample"]:::input

    classDef input fill:#eef,stroke:#88a;
```

## 3. Dạng văn bản (cây khối)

```text
Image X ──┬─► Adaptive Attention Controller (AAC) ──► (w1, w2) cấp cho 3 khối AAF
          │
          ▼
ENCODER
 ├─ Encoder_ResBlock : Conv2D(k3,p1) → BN → ReLU → Conv2D(k3,p1) → Residual Combination (+ skip từ X)
 ├─ AAF 1            : (w1 × PAM) ⊕ (w2 × CAM) → Conv2D(k3,p1)
 ├─ Residual Conv Module 1 : Enhancement Block (BN→ReLU) ∥ ResBlock (Conv→BN→ReLU→Conv) → Residual Combination
 ├─ AAF 2 → Residual Conv Module 2
 └─ AAF 3 → Residual Conv Module 3

BRIDGE (ASPP)
 ├─ ASPP_block1 : Conv2D(k3, pad 6)  → ReLU → BN
 ├─ ASPP_block2 : Conv2D(k3, pad 12) → ReLU → BN
 ├─ ASPP_block3 : Conv2D(k3, pad 18) → ReLU → BN
 └─ Concatenate → Conv2D(1×1)
      └─► Attention Module (kết hợp feature encoder + bridge) ──► UpSample

DECODER (lặp 3 tầng)
 UpSample(ConvTranspose2d) → Concatenate(dim=1, skip từ encoder) → Residual Conv Module → Attention Module → ...
 Tầng cuối: ... → Residual Conv Module → ASPP Module → Conv2D(k=1) → Sigmoid → OUTPUT
```

## 4. Bảng tóm tắt các khối

| Khối | Cấu tạo | Vai trò |
|---|---|---|
| AAC | Self-attention trên feature thu nhỏ 8×8 → FC → Softmax | Sinh trọng số thích nghi (w1, w2) |
| AAF | w1×PAM ⊕ w2×CAM → Conv2D(k3,p1) | Trộn position và channel attention |
| Residual Conv Module | Enhancement Block + ResBlock + Residual Combination | Trích xuất đặc trưng ổn định nhờ residual |
| ASPP | 3 nhánh dilation 6/12/18 → Concatenate → Conv 1×1 | Đặc trưng đa tỉ lệ |
| Attention Module | conv_Encoder, conv_Decoder → ⊕ → conv_Attention → ⊗ | Lọc skip connection |
| UpSample | ConvTranspose2d | Tăng độ phân giải |
| Đầu ra | Conv2D(k=1) → Sigmoid | Mask xác suất [0, 1] |
