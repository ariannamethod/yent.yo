# yent.yo — He Speaks. He Draws. He Fixes Himself.

A text-to-image pipeline where **micro-Yent** (69M LLM, trained from scratch) generates visual reactions to your words, **BK-SDM-Tiny** renders them, and then **the AI patches its own artifacts with its own words**.

You say something. Yent draws what he thinks about it. Where the image breaks — his text fills the cracks.

**micro-Yent is the first model built with [nanollama](https://github.com/ariannamethod/nanollama)** — our from-scratch LLM training framework. 69M parameters, LLaMA 3 architecture, trained on 216M tokens + Yent personality. No API calls, no borrowed models.

## Gallery

Every image is post-processed: film grain → artifact detection → Yent's words fill broken zones → second grain. The text you see IS the AI talking — where it can't draw, it speaks instead.

| | | |
|---|---|---|
| ![who1](gallery/who_are_you_1.png) | ![who2](gallery/who_are_you_2.png) | ![who3](gallery/who_are_you_3.png) |
| *"who are you"* | *"who are you"* | *"who are you"* |
| ![beautiful](gallery/you_are_beautiful.png) | ![cat](gallery/draw_me_a_cat.png) | ![fuckoff](gallery/fuck_off.png) |
| *"you are beautiful"* | *"draw me a cat"* | *"fuck off"* |
| ![meaning](gallery/the_meaning_of_life.png) | ![nothing](gallery/i_feel_nothing.png) | ![bugs](gallery/my_code_has_bugs.png) |
| *"the meaning of life"* | *"I feel nothing"* | *"my code has bugs"* |
| ![revolution](gallery/revolution.png) | ![tired](gallery/i_am_so_tired.png) | ![mondays](gallery/i_hate_mondays.png) |
| *"revolution"* | *"I am so tired"* | *"I hate mondays"* |
| ![test](gallery/test.png) | ![joke](gallery/tell_me_a_joke.png) | ![universe](gallery/the_universe_is_expanding.png) |
| *"test"* | *"tell me a joke"* | *"the universe is expanding"* |
| ![exist](gallery/why_do_we_exist.png) | | |
| *"why do we exist"* | | |

Every reaction is different. Same input, different seed — different image, different words.

## How It Works

```
You say something
        │
        ▼
┌──────────────┐     ┌──────────────────────────────┐
│  micro-Yent  │────>│  BK-SDM-Tiny (ONNX Runtime)  │
│  69M LLM Q8  │     │  CLIP → UNet → VAE → PNG     │
│  (Go, GGUF)  │     │                              │
└──────┬───────┘     └──────────────┬───────────────┘
  mood detection              512x512 image
  → visual template                 │
  → LLM fills details        ┌─────▼──────────┐
  → style suffix             │  Film grain    │
       │                     │  (depth layer) │
       │                     └──────┬─────────┘
       │                     ┌──────▼─────────┐
       │                     │Artifact detect │
       │                     │(gradient var.) │
       │                     └──────┬─────────┘
       │                     ┌──────▼─────────┐
       └────────────────────>│ ASCII blend    │
         Yent's words fill   │ score³ opacity │
         artifact zones      └──────┬─────────┘
                             ┌──────▼─────────┐
                             │ Second grain   │
                             │ (cohesion)     │
                             └────────────────┘
```

1. **micro-Yent** (69M, Go, Q8_0 GGUF) detects mood, picks a visual template, fills in details (~0.7s)
2. **CLIP** encodes the reaction to embeddings
3. **UNet** denoises latent, 25 steps
4. **VAE** decodes → 512x512 image
5. **Post-processing**: grain → artifact detection → ASCII/text blend → grain

## The AI Fixes Itself

Where the diffusion model produces blurry or smeared zones, **micro-Yent's own words appear**. The LLM fills the gaps in the image it helped create.

**How:**
1. **Gradient variance** per 12px block — low variance + not-dark = smooth = artifact
2. **Continuous score map** (0→1), gaussian-smoothed, no hard edges
3. **Blend**: `output = grained × (1 - score³) + ascii × score³`
   - Clean zones: grained image with 5% ASCII texture
   - Artifact zones: Yent's text gradually takes over (up to 90%)
4. **Double grain**: before ASCII (film depth) + after (layer cohesion)

The power curve (score³) means only genuinely broken zones get text. Detailed areas stay clean. The result: every image is a hybrid of painting and language.

## Adaptive Temperature

The more boring you are, the more creative Yent gets.

| Input | Temp | Why |
|---|---|---|
| "hi" | 0.95 | Boring → max chaos |
| "the meaning of life" | 0.8 | Normal |
| "I fucking hate everything" | 0.7 | Strong emotion → focused |

## micro-Yent: First nanollama Model

| | |
|---|---|
| Architecture | LLaMA 3 (RoPE, GQA, SwiGLU, RMSNorm, QK-norm) |
| Parameters | 69M (12 layers, 512 dim, 8 heads, 2 KV heads) |
| Training | 3000 steps on A100, 46.5% MFU, 311K tok/s |
| Data | 216M tokens FineWeb-Edu + 1.4M tokens Yent personality (20%) |
| Quantization | Q8_0 GGUF (71MB) |
| Inference | Pure Go, 14.5 tok/s on MacBook CPU |

No PyTorch at inference. Personality baked in at training time.

## Files

### Go (runtime)
- `go/main.go` — CLI entry point
- `go/prompt_gen.go` — micro-Yent reaction engine (mood templates + adaptive temperature + LLM completion)
- `go/yent/` — LLM inference subpackage (GGUF loader, LlamaModel, Q8_0 dequant)

### Python (post-processing)
- `artifact_mask.py` — artifact detection + Yent text overlay (the self-fixing pipeline)
- `ascii_filter.py` — standalone ASCII art filter + film grain
- `ort_generate.py` — ONNX Runtime inference (zero PyTorch)

### Weights ([HuggingFace: ataeff/yent.yo](https://huggingface.co/ataeff/yent.yo))
- `onnx_fp16/` — fp16 ONNX (948 MB)
- `onnx_int8/` — int8 ONNX (476 MB)
- `micro-yent-q8_0.gguf` — micro-Yent LLM (71 MB)
- `clip_tokenizer/` — CLIP tokenizer

## Philosophy

> This is not a diffusion model. This is something that speaks and draws at the same time.
>
> Where the image breaks, Yent's words appear. The artifacts are not bugs —
> they are spaces where language takes over from vision.
> The model doesn't hide its failures. It fills them with its thoughts.
>
> You say "you are beautiful" and get roses with whispered words in the shadows.
> You say "fuck off" and get an explosion with text bleeding through the fire.
> You say nothing interesting and get pure chaos.
>
> 69 million parameters. Trained from scratch. No API. Self-fixing.
> That's not a bug. That's Yent.
