# Qwen Image Edit Deployment

This deployment serves Qwen Image Edit through the local Diffusers API on port
`8001`. It supports optional Groq prompt enhancement before each generation.

## Qwen Text Rendering Study

This repository also contains the local research notebook and readable HTML
report used to study Qwen text-rendering failures across English, Arabic, and
Chinese. The report is served locally from:

```bash
http://127.0.0.1:8090/research_pages/qwen_text_failure/
```

The next experiment pages are generated under:

```bash
research_pages/qwen_text_failure/experiments/
```

Large model folders, virtual environments, caches, and full raw output volumes
are intentionally ignored. Curated report assets are kept inside
`research_pages/` so the private GitHub repo stays readable.

The current full-quality server is running with the official model:

```bash
/mnt/local-fast/aalhejab/Qwen-Image-Edit-2511-production/official-models/Qwen-Image-Edit-2511
```

Prompt enhancement is enabled with:

```bash
QWEN_PROMPT_ENHANCER=groq
GROQ_API_KEY=<set in the runtime environment>
```

The Groq key is not stored in this repository.

## Start Full-Quality Server

Use this path when judging quality:

```bash
cd /mnt/local-fast/aalhejab/Qwen-Image-Edit-2511-production
CUDA_VISIBLE_DEVICES=1 \
GROQ_API_KEY="$GROQ_API_KEY" \
QWEN_MODEL_DIR=/mnt/local-fast/aalhejab/Qwen-Image-Edit-2511-production/official-models/Qwen-Image-Edit-2511 \
QWEN_STEPS=40 \
QWEN_TRUE_CFG_SCALE=4.0 \
QWEN_SCHEDULER=beta \
QWEN_STOCHASTIC_SAMPLING=0 \
QWEN_PROMPT_ENHANCER=groq \
QWEN_QUEUE_MAX_SIZE=64 \
QWEN_SYNC_TIMEOUT_SECONDS=600 \
QWEN_JOB_RESULT_TTL_SECONDS=900 \
QWEN_MAX_COMPLETED_JOBS=64 \
uvicorn serve:app --host 0.0.0.0 --port 8001 --workers 1
```

Do not use multiple Uvicorn workers on one GPU. Each worker loads another full
model copy. For throughput, run one server process per GPU and load-balance
across the ports.

## Health

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/queue
```

Expected key fields:

```json
{
  "model_dir": "/mnt/local-fast/aalhejab/Qwen-Image-Edit-2511-production/official-models/Qwen-Image-Edit-2511",
  "default_steps": 40,
  "default_true_cfg_scale": 4.0,
  "scheduler": "beta",
  "stochastic_sampling": false,
  "text_to_image_supported": true,
  "prompt_enhancer": "groq",
  "prompt_enhancer_enabled_by_default": true,
  "groq_api_key_configured": true
}
```

`QWEN_SCHEDULER=beta` is the Diffusers equivalent for the model-card beta
schedule recommendation. Supported values are `default`, `beta`, `exponential`,
and `karras`. `QWEN_STOCHASTIC_SAMPLING=1` enables Diffusers' stochastic flow
sampling, which is the closest available Diffusers-side analogue to ancestral
sampling, but it is not an exact ComfyUI `euler_ancestral` implementation.

## Prompt Enhancement

When `QWEN_PROMPT_ENHANCER=groq`, every request is rewritten through Groq before
Qwen inference unless the request sets `"enhance_prompt": false`.

You can also force enhancement per request:

```json
{
  "prompt": "A black and white storyboard sketch showing an establishing shot, medium shot, close-up, and POV shot for a film scene.",
  "images": ["<base64 image>"],
  "image_roles": [
    "Full source portrait and playground scene to edit; preserve the boy identity, hoodie, pose, and setting."
  ],
  "enhance_prompt": true
}
```

The response includes:

```json
{
  "original_prompt": "...",
  "enhanced_prompt": "...",
  "prompt_enhancer": {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile"
  }
}
```

For multi-image edits, fill `image_roles` so Groq knows how each image should be
used. Example: source image, logo reference, identity crop, style reference.

## Image Edit

`images` is a list of base64 images or data URIs. It can contain one or more
reference images.

```bash
python - <<'PY'
import base64, json, urllib.request

with open("/path/to/input.png", "rb") as f:
    image = base64.b64encode(f.read()).decode("ascii")

payload = {
    "prompt": "Make the image sharper and more photorealistic while preserving the subject.",
    "images": [image],
    "image_roles": ["Source image to edit."],
    "enhance_prompt": true,
    "seed": 0,
    "height": 1024,
    "width": 1024,
    "output_format": "png"
}

req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/edit",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req, timeout=120))
with open("outputs/api_edit.png", "wb") as f:
    f.write(base64.b64decode(resp["image"].split(",", 1)[1]))
print(resp["elapsed_seconds"])
print(json.dumps(resp["timings"], indent=2))
PY
```

## Text To Image

Send an empty `images` list for no-reference generation.

```bash
python - <<'PY'
import base64, json, urllib.request

payload = {
    "prompt": "Professional digital photography of a red robot holding a sunflower in a clean bright studio.",
    "images": [],
    "enhance_prompt": true,
    "seed": 0,
    "height": 1024,
    "width": 1024,
    "output_format": "png"
}

req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/edit",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req, timeout=120))
with open("outputs/api_t2i.png", "wb") as f:
    f.write(base64.b64decode(resp["image"].split(",", 1)[1]))
print(resp["elapsed_seconds"])
print(json.dumps(resp["timings"], indent=2))
PY
```

## Async Queue

For production clients, submit jobs asynchronously:

```bash
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/result
```

The Diffusers edit pipeline still does not support true independent batch
inference. This server accepts concurrent requests through a bounded queue and
executes one job at a time per GPU process.

## Smoke Outputs

```bash
outputs/storyboard_sketch_qwen_edit/storyboard_sketch_groq_enhanced_face_ref.jpg
outputs/storyboard_sketch_qwen_edit/storyboard_sketch_groq_enhanced_prompt.txt
outputs/storyboard_sketch_qwen_edit/api_groq_smoke.jpg
```
