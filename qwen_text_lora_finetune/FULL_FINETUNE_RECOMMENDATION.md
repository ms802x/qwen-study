# Full Finetuning Recommendation

## Short Answer

Full finetuning can plausibly improve Arabic and long English text rendering, but
it is not a guaranteed fix by itself. It is only worth doing after the dataset and
evaluation loop are strong enough. Full finetuning on the current 100 synthetic
images would likely overfit the clean poster style and damage general Qwen edit
quality.

## What This LoRA Proved

- The local training pipeline works.
- The model can be pushed toward clean text-poster layouts.
- A seen 10-word English target can become much cleaner after training.
- Held-out long English is still unreliable.
- Arabic exact rendering did not learn in a useful way.

The Arabic failure is not surprising. Arabic has connected glyph shaping,
right-to-left ordering, diacritics/marks, and many possible letter forms. A small
attention-only LoRA on 100 images mostly learns style/layout and weak text priors;
it does not provide enough signal to rewrite the model's internal prompt-to-glyph
mapping.

## Trainer Choice

Do not use the small local trainer here for the serious full-model run. It is a
diagnostic harness.

Recommended production trainer:

- DiffSynth-Studio:
  https://github.com/modelscope/DiffSynth-Studio
- Qwen-Image-Edit-2511 full-training script:
  https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/full/Qwen-Image-Edit-2511.sh
- Qwen-Image-Edit-2511 LoRA script:
  https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh

Alternative serious trainer:

- Musubi Qwen-Image docs:
  https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md

My recommendation for this project is DiffSynth-Studio full DiT finetuning for the
main experiment, plus LoRA probes for cheap ablations.

## Dataset Needed Before Full Finetuning

Minimum first serious run:

- 5k to 20k generated poster/text-layout images.
- Balanced Arabic-only, English-only, and bilingual samples.
- Exact text strings stored in metadata.
- Font diversity across Arabic and Latin.
- Size diversity: headline, body, caption, small labels.
- Layout diversity: centered posters, menus, infographics, tables, callouts,
  product labels, signs, and charts.
- Held-out text vocabulary that never appears in training.

Do not train only on beautiful posters. Include simple white-background text
cards because they isolate glyph fidelity from style.

## Evaluation Gate

Before accepting a full finetune, evaluate:

- Exact English word accuracy at 10, 20, and 50 words.
- Arabic visual word accuracy at 5, 10, and 20 words.
- Mixed Arabic/English ordering.
- Same prompt across 8 to 16 fixed seeds.
- General edit quality regression on non-text images.
- Identity/reference preservation regression.

For Arabic, use both human inspection and OCR where available. Arabic OCR will
not be perfect, but it is still useful as a regression signal when paired with
visual review.

## Suggested Full-Finetune Strategy

1. Start with the Qwen image generation model path, not only edit, if the goal is
   poster generation from scratch.
2. Finetune the DiT first; keep VAE frozen unless there is evidence the VAE cannot
   reconstruct Arabic glyphs.
3. Use a low learning rate, around `1e-6` to `5e-6`, for full finetuning.
4. Use gradient checkpointing and bf16 on the H100s.
5. Save frequent checkpoints and evaluate every checkpoint visually.
6. Stop early if Arabic improves but general image quality starts to collapse.

The right conclusion from the current LoRA is not "Arabic is impossible." The
right conclusion is that Arabic exact text rendering is a data-and-objective
problem, and it needs a serious full-finetuning recipe rather than a 100-image
attention LoRA.
