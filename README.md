# Vision Language Model Fine-Tuning for Document-to-Markdown Generation

**Course Project | Generative AI**
**Student IDs:** 22F8785 · 22F8805

---

## Overview

This project fine-tunes a Vision Language Model (VLM) using **QLoRA** (Quantized Low-Rank Adaptation) to convert document page images into structured Markdown text. Given an image of any document page — academic paper, report, scanned PDF — the model generates clean, readable Markdown that mirrors the document's structure.

The approach combines parameter-efficient fine-tuning with 4-bit quantization, making it feasible to adapt a 2-billion parameter multimodal model on commodity GPU hardware (Kaggle T4 × 2).

---

## Project Structure

```
project/
├── 22F8785_22F8805_project_genAI.ipynb   # Main notebook (training + inference + app)
└── README.md
```

The notebook is organized into two major cells:

| Cell | Purpose |
|------|---------|
| **Cell 1** | Environment setup, data loading, QLoRA fine-tuning, adapter saving |
| **Cell 2** | Adapter loading, inference on train/val samples, Gradio app deployment |

---

## Model

| Component | Details |
|-----------|---------|
| Base Model | [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Architecture | Vision Language Model (image encoder + language decoder) |
| Parameters | ~2 Billion |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| Fine-tuning | QLoRA — only LoRA adapter weights are trained |

---

## Dataset

**Nougat Training Dataset Example**
[kaggle.com/datasets/zphilip/nougat-training-dataset-example](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example)

Each sample in the dataset is an `(image, markdown)` pair representing a document page and its corresponding Markdown transcription. The loader supports `.jsonl` manifests as well as paired image/text files (`.mmd`, `.md`, `.txt`).

**Split used:**

| Split | Size |
|-------|------|
| Training | 80% (capped at 500 samples) |
| Validation | 20% |

---

## Fine-Tuning Configuration

### QLoRA Settings

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 (`bnb_4bit_quant_type="nf4"`) |
| Compute dtype | `float16` |
| Double quantization | Enabled |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

### Training Arguments

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation steps | 8 (effective batch = 8) |
| Learning rate | 2e-4 |
| Warmup steps | 20 |
| Optimizer | `paged_adamw_8bit` |
| Mixed precision | `fp16` |
| Gradient checkpointing | Enabled |
| Max sequence length | 2048 tokens |
| Image resolution | 128–384 pixel tiles (28×28 each) |

---

## How It Works

### Training Pipeline

1. **Load dataset** — parse JSONL manifests to collect `(image_path, markdown)` pairs.
2. **Quantize base model** — load `Qwen2-VL-2B-Instruct` in 4-bit precision.
3. **Inject LoRA adapters** — attach lightweight trainable adapters to attention projections.
4. **Format as ChatML** — each sample is formatted as a two-turn conversation:
   - *User:* `[image] + "Convert this document page to Markdown."`
   - *Assistant:* `[ground-truth markdown]`
5. **Train** — only adapter weights are updated; base model weights remain frozen.
6. **Save adapter** — the fine-tuned LoRA adapter is saved separately from the base model.

### Inference Pipeline

1. Load base model in 4-bit and merge the saved LoRA adapter via `PeftModel`.
2. Pass a document image and instruction prompt through the processor.
3. Generate Markdown autoregressively with greedy decoding (`do_sample=False`).
4. Display raw Markdown text alongside a rendered preview.

---

## Evaluation

The notebook evaluates the fine-tuned model on:

- **3 training images** — to verify the model has learned the training distribution.
- **3 unseen validation images** — to assess generalization.

For each sample, the predicted Markdown and ground-truth Markdown are printed side-by-side for qualitative comparison.

---

## Gradio Application

A Gradio web app is included for interactive inference:

```
Upload a document image → Enter an instruction → Adjust max tokens → Click "Convert"
```

**Outputs:**
- **Markdown (raw)** — copyable text box with the generated Markdown.
- **Preview** — live-rendered Markdown preview.

The app launches with a public shareable link (`share=True`) for easy demo access on Kaggle.

---

## Environment

| Component | Specification |
|-----------|--------------|
| Platform | Kaggle Notebooks |
| Accelerator | GPU T4 × 2 |
| Framework | PyTorch + Hugging Face Transformers |
| Key Libraries | `transformers`, `peft`, `bitsandbytes`, `accelerate`, `gradio`, `datasets` |

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -q -U bitsandbytes accelerate peft
```

### 2. Configure Paths

At the top of the notebook, update these constants if needed:

```python
DATA        = "/kaggle/input/datasets/zphilip/nougat-training-dataset-example"
MODEL       = "Qwen/Qwen2-VL-2B-Instruct"
OUT         = "/kaggle/working/qwen2vl-nougat"
ADAPTER     = "/kaggle/input/datasets/faiqahmad01/finetuned-qwen-adapter"
TRAIN_LIMIT = 500
```

### 3. Run Training (Cell 1)

Execute Cell 1 to fine-tune the model. The trained adapter will be saved to `OUT`.

### 4. Run Inference + App (Cell 2)

Execute Cell 2 to load the fine-tuned adapter, run sample predictions, and launch the Gradio app.

---

## Key Design Decisions

**Why QLoRA?**
Full fine-tuning a 2B-parameter VLM requires tens of GBs of GPU memory. QLoRA reduces this to a fraction by quantizing the frozen base model and training only small low-rank matrices, making the entire pipeline fit on two T4 GPUs.

**Why Qwen2-VL?**
Qwen2-VL-2B-Instruct is a capable open-source VLM with strong document understanding out of the box. Its instruction-tuned format aligns naturally with the ChatML prompt structure used here.

**Why the Nougat dataset?**
The Nougat dataset provides high-quality image–Markdown pairs derived from academic papers, offering diverse document layouts including equations, tables, and multi-column text — exactly the complexity this task targets.

---

## Limitations

- Training is capped at 500 samples; a larger subset would likely improve Markdown fidelity.
- Only 2 epochs are run due to compute constraints; longer training may reduce validation loss.
- Complex tables and mathematical equations may not be rendered perfectly by the current adapter.

---

## References

- [Qwen2-VL model card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Nougat dataset on Kaggle](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example)
- [PEFT / LoRA documentation](https://huggingface.co/docs/peft)
- [BitsAndBytes quantization](https://huggingface.co/docs/bitsandbytes)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
