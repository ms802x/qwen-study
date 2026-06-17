# Qwen Text LoRA Finetune

Small controlled LoRA experiment for Qwen text rendering.

## Contents

- `dataset/images/`: 100 synthetic 512x512 text-poster images.
- `dataset/metadata/train.jsonl`: prompts and exact target text for each image.
- `checkpoints/qwen_text_lora/pytorch_lora_weights.safetensors`: trained transformer LoRA.
- `outputs/training_loss.jsonl`: step-level training loss.
- `outputs/training_summary.json`: training configuration and final loss.
- `outputs/eval/`: before/after generations for six evaluation prompts.
- `report/index.html`: readable report with dataset, loss plot, before/after sheet, and conclusions.

## Commands

```bash
.venv-qwen-research/bin/python qwen_text_lora_finetune/scripts/generate_synthetic_text_dataset.py

CUDA_VISIBLE_DEVICES=1 .venv-qwen-research/bin/python \
  qwen_text_lora_finetune/scripts/train_qwen_text_lora.py \
  --steps 800 --rank 8 --alpha 8 --lr 5e-5

CUDA_VISIBLE_DEVICES=1 .venv-qwen-research/bin/python \
  qwen_text_lora_finetune/scripts/evaluate_and_report.py
```

## Outcome

The LoRA learned the synthetic clean-poster prior and improved simple English text prompts in the evaluation sheet. It did not solve exact Arabic or bilingual text rendering. This is useful as a verified training pipeline, not a final quality fix.
