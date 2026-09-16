# ATDL: Knowledge Distillation & Ternary Quantization (CIFAR-10)

This repository contains the implementation, training notebooks, evaluation scripts, visualization plots, and checkpoints for **Knowledge Distillation (KD)** paired with **Quantization-Aware Training (QAT)** using a Ternary Weight ResNet18 student and a full-precision ResNet34 teacher on CIFAR-10.

---

## 📊 Summary of Results

| Model | Params (M) | Size (MB) | Test Acc (%) | Sparsity | Compression vs Teacher |
|---|---|---|---|---|---|
| **Teacher** (ResNet34, FP32) | 21.28 | 81.18 | **96.03%** | Dense FP32 | 1.00× (reference) |
| **Baseline** (ResNet18, FP32, no KD) | 11.17 | 42.63 | **95.55%** | Dense FP32 | 1.90× |
| **Student** (ResNet18, Ternary, KD+QAT) | 11.17 | **2.74** | **95.15%** | **47.2%** | **29.62×** |

> The compressed ternary student achieves **29.6× compression** over the teacher (and **15.5× compression** over the FP32 baseline) while retaining **95.15% test accuracy** (within 0.88% of the full-precision teacher).

---

## 📁 Repository Structure

```
├── codes/
│   ├── task1_resnet34_teacher_.ipynb       # Teacher training (ResNet34, FP32)
│   ├── task2_ternary_quantizer_.ipynb      # Ternary quantizer implementation & tests
│   ├── task3_kd_qat_training_.ipynb        # KD + QAT student training pipeline
│   ├── task4_resnet18_fp32_baseline_.ipynb # FP32 baseline without KD
│   ├── task4_final_evaluation_.ipynb       # Comparative benchmark & metrics
│   ├── save_compressed_checkpoint.ipynb    # Checkpoint bit-packing & compression
│   ├── general_visualizations.ipynb        # Visualization & analytical plots
│   └── test_custom_images_.ipynb           # Inference on custom images
├── visualization/                          # Generated plots, tables, and ablation results
│   ├── ablation_plot.png
│   ├── accuracy_vs_compression.png
│   ├── confusion_matrices.png
│   ├── final_comparison_table.csv
│   └── ...
├── Outputs/                                # Model weights & checkpoints (.pth)
├── requirements.txt                        # Environment dependencies
└── README.md
```

---

## 🚀 Setup & Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Recommended hardware: CUDA-enabled GPU (e.g. CUDA 12.x).
