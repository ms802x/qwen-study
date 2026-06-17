# Qwen Text LoRA Finetune

Small controlled LoRA experiment for Qwen text rendering. The current revision uses
10-20 word English targets and shorter Arabic targets to keep Arabic glyphs large
enough for visual inspection at 512px.

## Contents

- `dataset/images/`: 100 synthetic 512x512 text-poster images.
- `dataset/metadata/train.jsonl`: prompts and exact target text for each image.
- `checkpoints/qwen_text_lora/pytorch_lora_weights.safetensors`: trained transformer LoRA.
- `outputs/training_loss.jsonl`: step-level training loss.
- `outputs/training_summary.json`: training configuration and final loss.
- `outputs/eval/`: before/after generations for six evaluation prompts.
- `report/index.html`: readable report with dataset, loss plot, before/after sheet, and conclusions.
- `FULL_FINETUNE_RECOMMENDATION.md`: recommendation for the serious full-model Arabic/English text run.

## Commands

```bash
.venv-qwen-research/bin/python qwen_text_lora_finetune/scripts/generate_synthetic_text_dataset.py

CUDA_VISIBLE_DEVICES=1 .venv-qwen-research/bin/python \
  qwen_text_lora_finetune/scripts/train_qwen_text_lora.py \
  --steps 1000 --rank 16 --alpha 16 --lr 5e-5 --recache

CUDA_VISIBLE_DEVICES=1 .venv-qwen-research/bin/python \
  qwen_text_lora_finetune/scripts/evaluate_and_report.py
```

## Outcome

The LoRA learned the synthetic clean-poster prior and copied a seen 10-word
English target cleanly. Held-out long English is mixed and can still regress versus
base. Arabic remains the main failure case: the dataset contains readable Arabic
pixels, but this small attention LoRA does not reliably learn exact Arabic glyph
identity or word order. This is useful as a verified training pipeline, not a final
quality fix.

## Full Finetuning Decision

This folder is a diagnostic trainer, not the trainer I would use for a serious
full-model run. For full DiT finetuning of Qwen-Image-Edit-2511, use a maintained
Qwen-aware trainer such as DiffSynth-Studio. Full finetuning can plausibly help
Arabic and long English text rendering, but only with a much larger text-focused
dataset and OCR/recognition-based validation. Running full finetuning on only
these 100 synthetic images would likely overfit and harm general edit quality.
