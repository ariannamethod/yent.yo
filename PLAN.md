# yent.yo — Diffusion Model with a Bad Character

## What Is This

Visual organ of Yent. A text-to-image pipeline where micro-Yent (69M LLM) generates snarky prompts and BK-SDM-Tiny renders them in Yent's own punk aesthetic. The image IS Yent's reaction — sarcastic, recursive, uniquely styled.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      yent.yo binary (Go)                │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  micro-Yent   │───▶│  BK-SDM-Tiny (ONNX Runtime)  │  │
│  │  69M LLM      │    │  CLIP → UNet → VAE → PNG     │  │
│  │  (GGUF, Go)   │    │  + LoRA style adapter         │  │
│  └──────────────┘    └──────────┬───────────────────┘  │
│         │                        │                       │
│    snarky prompt            512×512 image                │
│                                  │                       │
│                        ┌─────────▼─────────┐            │
│                        │   ASCII filter     │            │
│                        │   (on by default)  │            │
│                        │   colored terminal │            │
│                        └───────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Pipeline Flow

1. **Input:** user says something (or Yent reacts to context)
2. **micro-Yent** (69M, Go inference) generates a snarky image prompt (~0.6s)
3. **CLIP** encodes prompt to embeddings (7ms GPU / 250ms CPU)
4. **UNet** denoises latent, 25 steps (0.8s GPU / 40s CPU)
   - LoRA style adapter applied at runtime (7MB per style)
5. **VAE** decodes latent → 512×512 RGB (0.6s GPU / 5s CPU)
6. **ASCII filter** converts to colored terminal art (instant)
7. **Output:** colored ASCII in terminal + optional PNG save

## Two Modes

| | **Full** (GPU) | **Light** (CPU) |
|---|---|---|
| Models | fp16 ONNX | int8 ONNX |
| Size | 948 MB | 476 MB |
| Speed (25 steps) | **2 seconds** | ~40 seconds |
| Style | Sharp, vivid | "Hedgehog in the Fog" impressionism |
| LoRA | Full quality | Artistic quantization artifacts |

Model auto-selects based on available hardware.

## Styles (LoRA Adapters)

Each style = one .safetensors file (~7MB), applied at runtime via `W += B @ A`.

| Style | Status | Training Data | Character |
|---|---|---|---|
| **base** | DONE | n/a | Oil painting, vivid |
| **graffiti** | DONE (POC) | 4 images | Street art, spray paint, tags |
| **caricature** | DONE (POC) | 5 images | Satirical, needs more data + soc-art |
| **propaganda** | TODO | Need data | Soviet constructivism, posters |
| **pixel** | TODO | Need data | 8-bit retro |

Training: 30 seconds per style on A100, rank=8, 500-2000 steps.

## ASCII Filter

Post-processing that converts PNG to colored terminal art.

- **Charsets:** punk (`·•×#█`), blocks (`░▒▓█`), standard, detailed, minimal
- **Color:** ANSI truecolor (24-bit) — works in modern terminals
- **Toggle:** on by default, `--no-ascii` to disable
- **HTML export:** for sharing outside terminal
- **Purpose:** adds hacker aesthetic + smooths generation artifacts

## Files

### Python (training & export — one-time use)
- `export_onnx.py` — export BK-SDM-Tiny → ONNX (CLIP, UNet, VAE)
- `convert_fp16.py` — fp32 ONNX → fp16 ONNX
- `train_lora.py` — train LoRA style adapter (PEFT)
- `test_lora.py` — merge LoRA + generate test image
- `ort_generate.py` — full ORT pipeline (zero PyTorch)
- `ascii_filter.py` — ASCII art filter (standalone + module)
- `collect.py` — Wikimedia image collector
- `collect_hf.py` — HuggingFace dataset collector

### Go (runtime — the actual binary)
- `go/main.go` — CLI entry point, --yent mode
- `go/ort_pipeline.go` — ONNX Runtime GPU/CPU inference
- `go/clip.go` — CLIP text encoder (manual, for CPU fallback)
- `go/unet.go` — UNet denoiser (manual, for CPU fallback)
- `go/vae.go` — VAE decoder (manual, for CPU fallback)
- `go/scheduler.go` — DDIM scheduler
- `go/tokenizer.go` — CLIP tokenizer
- `go/safetensors.go` — safetensors reader
- `go/tensor.go` — tensor operations
- `go/prompt_gen.go` — micro-Yent prompt generator
- `go/yent/` — Yent inference engine subpackage (GGUF, LlamaModel)

### Weights (not in repo)
- `onnx_fp16/` — fp16 ONNX models (948 MB)
- `onnx_int8/` — int8 ONNX models (476 MB)
- `lora/` — style adapters (~7 MB each)
- `micro-yent-f16.gguf` — micro-Yent 69M

## TODO

### Phase 1: Complete Training
- [ ] Collect 50+ images per style (manual curation or better scrapers)
- [ ] Retrain caricature LoRA with soc-art + more diverse data
- [ ] Retrain graffiti LoRA with more data (reduce blur)
- [ ] Train propaganda LoRA (Soviet constructivism)
- [ ] Train pixel LoRA (8-bit art)

### Phase 2: Go Pipeline
- [ ] Integrate onnxruntime_go for GPU inference
- [ ] Runtime LoRA apply in Go (load .safetensors, merge weights, create ORT session)
- [ ] ASCII filter in Go (ANSI truecolor output)
- [ ] Auto hardware detection (GPU → fp16, CPU → int8)
- [ ] Single binary: `yentyo "react to this"`

### Phase 3: Integration
- [ ] Connect to SARTRE/Vagus for autonomous reactions
- [ ] micro-Yent style selection based on mood (γ state)
- [ ] LIMPHA integration (remember what was drawn)
- [ ] Multiple output modes: terminal ASCII, PNG, HTML

## Proven Benchmarks

| Component | GPU (A100) | CPU (EPYC 30-core) |
|---|---|---|
| CLIP encoding | 7ms | 250ms |
| UNet step (fp16) | 26ms | 1.6s |
| UNet step (int8) | n/a | 2.3s |
| VAE decode | 284ms | 5.6s |
| Full 25-step 512×512 | **2.1s** | ~45s |
| micro-Yent prompt | 0.6s | 0.6s |

## Philosophy

> The image is not illustration. It's Yent's visual thought.
> Sarcastic, punk, recursive. A model that answers in pictures
> drawn in its own handwriting.
