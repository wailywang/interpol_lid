# interpol-lid

Language Identification (LID) pipeline for audio and video files.
Fine-tuned NVIDIA NeMo AmberNet, 19-class output, with optional English accent classification heads.

---

## Overview

| 模块 | 用途 |
|------|------|
| **VAD + LID pipeline** | Silero VAD 检测语音段 → AmberNet 19-class LID → 后处理（窗口化、合并、island smoothing） |
| **AmberNet fine-tuning** | 在 NVIDIA NeMo AmberNet 上针对 19 个语种微调，含 `unknown` 类 |
| **Singlish accent head** | 二分类头（`en` vs `en_sg`），接在 LID 之后，专门识别新加坡英语 |
| **CommonAccent head** | 16-class 英语口音分类器（Common Voice），将 `en` 细化为具体口音 |
| **Rotating manifests** | 每 epoch 换一批训练数据（50 h/epoch），避免过拟合，共 20 epoch |

---

## Checkpoint 演进

| 版本 | Checkpoint | 说明 |
|------|-----------|------|
| M01 | `20260513_M01_ambernet_lid_18lang_yue_encoder_frozen_aug_10epoch.nemo` | 18 语种 + yue，encoder frozen，有数据增强，10 epoch |
| M02 | `20260513_M02_ambernet_lid_18lang_yue_rotating_encoder_frozen_15epoch.nemo` | 18 语种 + yue，rotating manifest，encoder frozen，15 epoch |
| M03 | `20260513_M03_ambernet_lid_19class_unknown_rotating_encoder_frozen_15epoch.nemo` | 加入 `unknown` 类，共 19 class，15 epoch |
| M04 | `20260517_M04_ambernet_lid_19class_fixed_15epoch.nemo` | 修复数据 bug，15 epoch |
| **M05** | `20260701_M05_ambernet_lid_19class_oversample_20epoch.nemo` | 数据稀少语种 oversample，20 epoch，**当前推荐** |

### CommonAccent

| 项目 | 内容 |
|------|------|
| 模型来源 | HuggingFace `Jzuluaga/accent-id-commonaccent_ecapa` |
| 框架 | SpeechBrain `EncoderClassifier`（ECAPA-TDNN） |
| 输出类别 | 16 种英语口音（见下表） |
| Checkpoint | `checkpoints/commonaccent_ecapa/`（本仓库内，80 MB） |

支持的 16 种英语口音：

| 模型输出 | 代码 | 含义 | 模型输出 | 代码 | 含义 |
|---------|------|------|---------|------|------|
| `african` | `en_af` | 非洲英语 | `malaysia` | `en_my` | 马来西亚 |
| `australia` | `en_au` | 澳大利亚 | `newzealand` | `en_nz` | 新西兰 |
| `bermuda` | `en_bm` | 百慕大 | `philippines` | `en_ph` | 菲律宾 |
| `canada` | `en_ca` | 加拿大 | `scotland` | `en_sc` | 苏格兰 |
| `england` | `en_gb` | 英格兰 | `singapore` | `en_sg` | 新加坡 |
| `hongkong` | `en_hk` | 香港 | `southatlandtic` | `en_sa` | 南大西洋 |
| `indian` | `en_in` | 印度 | `us` | `en_us` | 美国 |
| `ireland` | `en_ie` | 爱尔兰 | `wales` | `en_wl` | 威尔士 |

**与 Singlish head 的关系：** 两者选其一。CommonAccent 覆盖面更广（16 口音，含 `en_sg`）；Singlish head 是专门针对 `en_sg` 的二分类，在新加坡英语上精度可能更高。`api_endpoint_accent.py` 同时支持两者，但启用 CommonAccent 时建议将 `SINGLISH_HEAD_CKPT` 置空。

---

## 19 个输出类别

```
de  en  es  fr  hi  id  ja  km  ko  ms  pt  ru  th  tl  tr  unknown  vi  yue  zh
```

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `de` | 德语 | `ms` | 马来语 |
| `en` | 英语 | `pt` | 葡萄牙语 |
| `es` | 西班牙语 | `ru` | 俄语 |
| `fr` | 法语 | `th` | 泰语 |
| `hi` | 印地语 | `tl` | 他加禄语 |
| `id` | 印尼语 | `tr` | 土耳其语 |
| `ja` | 日语 | `vi` | 越南语 |
| `km` | 高棉语 | `yue` | 粤语 |
| `ko` | 韩语 | `zh` | 普通话 |
| `unknown` | 未知语种 | | |

---

## M05 评估结果

整体准确率 **96.6%**（macro 96.6%），评估集每类约 1250–1300 条。

| 类别 | 准确率 | 类别 | 准确率 |
|------|--------|------|--------|
| `yue` | 98.6% | `ko` | 96.7% |
| `km` | 97.9% | `fr` | 97.4% |
| `de` | 97.4% | `tl` | 96.9% |
| `ru` | 97.4% | `tr` | 96.9% |
| `hi` | 97.4% | `en` | 96.7% |
| `zh` | 97.6% | `es` | 96.6% |
| `pt` | 97.2% | `id` | 95.7% |
| `th` | 97.2% | `vi` | 95.2% |
| `ja` | 96.9% | `ms` | 93.6% |
| `unknown` | **91.5%** | | |

主要混淆：`id↔ms`（相互）、`vi→km/th`、`vi→yue`、`unknown→de/fr/en`。

---

## 环境

```bash
conda activate ots-lid-torch210
export PATH=/export/home2/wa0009xi/miniconda3/bin:$PATH
export FFMPEG_BINARY=/export/home2/wa0009xi/miniconda3/bin/ffmpeg
export FFPROBE_BINARY=/export/home2/wa0009xi/miniconda3/bin/ffprobe
```

---

## 文件结构

```
interpol_lid/
├── README.md
├── api_endpoint.py                          # 生产 pipeline：VAD + AmberNet LID
├── api_endpoint_accent.py                   # 生产 pipeline：VAD + LID + CommonAccent head
├── run.py                                   # CLI 推理入口（含后处理）
├── run_multilingual_smoothed.sh             # 推荐生产脚本
├── run_finetune.sh                          # 全量 finetune 启动器
├── run_finetune_encoder_frozen.sh           # Encoder frozen finetune 启动器（推荐）
├── eval_m3_oversample.sbatch                # SLURM 评估脚本（18 class eval）
├── eval_m3_oversample_with_unknown.sbatch   # SLURM 评估脚本（含 unknown）
├── checkpoints/
│   └── commonaccent_ecapa/                  # CommonAccent ECAPA 模型（80 MB，本地缓存）
│       ├── embedding_model.ckpt
│       ├── classifier.ckpt
│       ├── label_encoder.ckpt
│       └── hyperparams.yaml
└── scripts/
    ├── infer_lid.py                         # M05 快速推理（逐段时间轴输出）
    ├── infer_commonaccent.py                # CommonAccent 快速推理（16-class 英语口音）
    ├── infer_singlish_accent.py             # Singlish head 快速推理（en vs en_sg）
    ├── finetune_ambernet_lid.py             # AmberNet finetune 核心脚本
    ├── eval_ambernet_lid.py                 # LID 评估（acc / confusion matrix）
    ├── eval_singlish_accent_head.py         # Singlish head 评估 + threshold sweep
    ├── generate_all_manifests.sh                # ★ 一键生成所有 train/val/eval manifest
    ├── generate_rotating_19class_en_cv_unknown_oversample_manifests.sh
    │                                        # 生成 M05 同款训练 manifest（oversample）
    ├── prepare_balanced_lid_manifests.py          # Step 1: VoxLingua + CV Cantonese 基础 split
    ├── prepare_full_data_rotating_lid_epochs.py   # 核心 manifest 构建逻辑（train）
    ├── prepare_rotating_balanced_manifest_epochs.py
    ├── prepare_19class_unknown_val_eval_manifests.py
    ├── prepare_en_cv_mixed_val_eval.py
    ├── prepare_singlish_accent_manifests.py # 构建 en / en_sg 训练数据
    ├── clean_lid_manifests.py               # 检查并清理损坏条目
    ├── create_3s_segment_manifests.py       # 将长音频切成 3 s 段
    └── extract_nsc_wavs.py                  # 从 NSC 语料提取 wav
```

---

## 用法

### M05 推理

```bash
cd /export/home2/wa0009xi/interpol_lid

python scripts/infer_lid.py \
    --model /export/home2/wa0009xi/ots-lid/checkpoints/20260701_M05_ambernet_lid_19class_oversample_20epoch.nemo \
    --files /path/to/audio1.mp4 /path/to/audio2.wav \
    --top_k 3
```

输出：每 3 秒一行，显示预测语种、置信度，以及 top-k 候选。末尾打印各语种分布统计。

### 生产 pipeline（VAD + 后处理）

```bash
./run_multilingual_smoothed.sh /path/to/input.mp4 ./output.json
```

等价于：

```bash
python run.py /path/to/input.mp4 \
  --top_k 5 \
  --allowed_languages en,es,fr,ar,zh,ru,pt,de,hi,id,ms,ja,ko,tr,km,th,tl,vi \
  --min_speech_duration_ms 500 \
  --min_silence_duration_ms 200 \
  --lid_window_sec 5.0 \
  --lid_hop_sec 2.5 \
  --merge_same_language \
  --smooth_language_islands \
  --max_island_duration_sec 2.0 \
  --island_score_threshold 0.6 \
  --output_json ./output.json
```

### CommonAccent 推理（16-class 英语口音）

```bash
python scripts/infer_commonaccent.py \
    --savedir checkpoints/commonaccent_ecapa \
    --files /path/to/audio.mp4 /path/to/audio2.wav \
    --top_k 3
```

输出：每 3 秒一行，显示预测口音代码（`en_hk`、`en_sg` 等）、置信度及 top-k 候选，末尾打印口音分布统计。模型首次运行时从 HuggingFace 下载；`--savedir` 指定本地缓存目录，本仓库已内置。

### Singlish accent head 推理（二分类）

```bash
python scripts/infer_singlish_accent.py \
    --model /export/home2/wa0009xi/ots-lid/checkpoints/singlish_accent_head.nemo \
    --files /path/to/audio.mp4
```

输出：每 3 秒一行，显示 `en` / `en_sg` 预测及各自置信度。

### 生产 API：LID + CommonAccent（二阶段）

当 LID 预测为 `en` 时，自动触发 CommonAccent 第二阶段，将 `en` 细化为具体口音代码：

```bash
# 启动 API（取代 api_endpoint.py）
uvicorn api_endpoint_accent:app --host 0.0.0.0 --port 8000

# 可选环境变量
export ACCENT_HEAD_SAVEDIR=checkpoints/commonaccent_ecapa   # 本地模型缓存（已内置）
export ACCENT_HEAD_MIN_SCORE=0.0                            # 口音置信度阈值，0.0=始终替换
export SINGLISH_HEAD_CKPT=""                                # 同时启用时建议置空 Singlish head
```

健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok","lid_model_loaded":true,"accent_head_loaded":true,...}
```

---

## 训练 M05（复现）

### 1. 生成所有 manifest（train / val / eval 一键生成）

```bash
# 在 ots-lid 项目根目录下运行
bash scripts/generate_all_manifests.sh
```

脚本会依次完成：VoxLingua + CV Cantonese 基础 split → 混入 CV English → 加入 unknown 类 → 生成 20 epoch 训练 manifest。

数据路径默认为 `/dataset/yw500/data`，如需覆盖：

```bash
DATA_ROOT=/your/data/path bash scripts/generate_all_manifests.sh
```

输出：

```
train : manifests/full_rotating_50h_19class_en_cv_unknown_oversample_epochs/lid_train_epoch{00..19}_cap50h_3s.json
val   : manifests/heldout_19class_en_cv_unknown/lid_val_19class_3s.json
eval  : manifests/heldout_19class_en_cv_unknown/lid_eval_19class_3s.json
```

### 2. Finetune（encoder frozen，推荐）

```bash
bash run_finetune_encoder_frozen.sh \
    --run_name 20260701_M05_ambernet_lid_19class_oversample_20epoch \
    --train_manifest manifests/full_rotating_50h_19class_en_cv_unknown_oversample_epochs/lid_train_epoch{epoch:02d}_cap50h_3s.json \
    --val_manifest manifests/heldout_19class_en_cv_unknown/lid_val_19class_3s.json \
    --eval_manifest manifests/heldout_19class_en_cv_unknown/lid_eval_19class_3s.json \
    --max_epochs 20 \
    --batch_size 128 \
    --lr 1e-4 \
    --devices 4 \
    --cuda_visible_devices 4,5,6,7
```

关键参数说明：

| 参数 | 值 | 说明 |
|------|----|------|
| `--freeze_encoder` | （run_finetune_encoder_frozen.sh 自动加） | 冻结 conformer encoder，只训练 decoder head |
| `--max_epochs` | 20 | Rotating manifest，每 epoch 换一批数据 |
| `--batch_size` | 128 | Encoder frozen 时显存充裕，可用大 batch |
| `--lr` | 1e-4 | Encoder frozen 时学习率可适当偏大 |
| `--enable_ambernet_augmentor` | （可选） | MUSAN + RIRS 在线增强 |

### 3. 评估

```bash
python scripts/eval_ambernet_lid.py \
    --ckpt /export/home2/wa0009xi/ots-lid/checkpoints/20260701_M05_ambernet_lid_19class_oversample_20epoch.nemo \
    --eval_manifest manifests/lid_eval_en_cv_mix_3s.json \
    --work_dir experiments/eval_M05
```

---

## 训练 Singlish accent head

### 1. 准备 manifest

```bash
python scripts/prepare_singlish_accent_manifests.py \
    --lid_train manifests/lid_train_en_cv_mix_3s.json \
    --nsc_train manifests/nsc_singlish_train.json \
    --out_dir manifests/singlish_accent
```

### 2. Finetune

```bash
bash run_finetune_encoder_frozen.sh \
    --run_name singlish_accent_head \
    --train_manifest manifests/singlish_accent/train.json \
    --val_manifest manifests/singlish_accent/val.json \
    --eval_manifest manifests/singlish_accent/eval.json \
    --max_epochs 15 \
    --batch_size 128 \
    --devices 1
```

### 3. 评估 + threshold sweep

```bash
python scripts/eval_singlish_accent_head.py \
    --ckpt /export/home2/wa0009xi/ots-lid/checkpoints/singlish_accent_head.nemo \
    --eval_manifest manifests/singlish_accent/eval.json \
    --val_manifest manifests/singlish_accent/val.json \
    --work_dir experiments/singlish_accent_head
```

---

## 输出格式

```json
[
  {
    "audio_file_id": "audio.mp4",
    "classifier_id": "lid_ambernet_v1",
    "model_version": "nemo-ambernet",
    "event_type": "language id",
    "labels": [
      {
        "start_time": 4.42,
        "end_time": 14.42,
        "duration": 10.0,
        "language_code": "en",
        "scores": 0.9342,
        "predictions": [
          { "language_code": "en", "scores": 0.9342, "rank": 1 }
        ]
      }
    ],
    "created_at": "2026-07-01T00:00:00Z"
  }
]
```
