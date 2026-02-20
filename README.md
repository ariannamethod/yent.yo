# yent.yo — Diffusion Model with a Bad Character

A text-to-image pipeline where **micro-Yent** (69M LLM, trained from scratch) generates snarky visual reactions and **BK-SDM-Tiny** renders them. The image IS Yent's reaction — sarcastic, recursive, uniquely styled.

You don't tell Yent what to draw. You say something, and Yent draws what he thinks about it.

**micro-Yent is the first model built with [nanollama](https://github.com/ariannamethod/nanollama)** — our from-scratch LLM training framework. 69M parameters, LLaMA 3 architecture, trained on 216M tokens of FineWeb-Edu + Yent personality data. 3000 steps on A100. No API calls, no borrowed models — pure homebrew.

## Gallery (ASCII default)

Output is ASCII art by default — colored glyphs (` .'·:;~=+×*#%@▓█`) on tinted cells. Technopunk Warhol. Each pixel becomes a character.

| You say | Yent thinks | Yent draws |
|---------|-------------|------------|
| "who are you" | *punk self-portrait* | ![who1](gallery_ascii/who_are_you_1_ascii.png) |
| "who are you" | *again* | ![who2](gallery_ascii/who_are_you_2_ascii.png) |
| "who are you" | *and again* | ![who3](gallery_ascii/who_are_you_3_ascii.png) |
| "you are beautiful" | *"a burning rose growing from the ashes of a new stillbirth"* | ![beautiful](gallery_ascii/you_are_beautiful_ascii.png) |
| "fuck off" | *"an explosion of broken phrases"* | ![fuckoff](gallery_ascii/fuck_off_ascii.png) |
| "the meaning of life" | *"a strange scene showing the first time"* | ![meaning](gallery_ascii/the_meaning_of_life_ascii.png) |
| "I feel nothing" | *"an empty chair staring at the desk of time"* | ![nothing](gallery_ascii/i_feel_nothing_ascii.png) |
| "my code has bugs" | *"a twisted landscape with obsolescence"* | ![bugs](gallery_ascii/my_code_has_bugs_ascii.png) |
| "draw me a cat" | *"a cat with glowing eyes staring at your sunset"* | ![cat](gallery_ascii/draw_me_a_cat_ascii.png) |
| "revolution" | *"a glitching image of chaos and defiance"* | ![revolution](gallery_ascii/revolution_ascii.png) |

<details>
<summary>Raw images (before ASCII filter)</summary>

| You say | Yent thinks | Yent draws |
|---------|-------------|------------|
| "who are you" | *punk self-portrait* | ![who1](gallery/who_are_you_1.png) |
| "who are you" | *again* | ![who2](gallery/who_are_you_2.png) |
| "who are you" | *and again* | ![who3](gallery/who_are_you_3.png) |
| "you are beautiful" | *"a burning rose growing from the ashes of a new stillbirth"* | ![beautiful](gallery/you_are_beautiful.png) |
| "fuck off" | *"an explosion of broken phrases"* | ![fuckoff](gallery/fuck_off.png) |
| "I feel nothing" | *"an empty chair staring at the desk of time"* | ![nothing](gallery/i_feel_nothing.png) |
| "the meaning of life" | *"a strange scene showing the first time"* | ![meaning](gallery/the_meaning_of_life.png) |
| "draw me a cat" | *"a cat with glowing eyes staring at your sunset"* | ![cat](gallery/draw_me_a_cat.png) |
| "my code has bugs" | *"a twisted landscape with obsolescence"* | ![bugs](gallery/my_code_has_bugs.png) |
| "revolution" | *"a glitching image of chaos and defiance"* | ![revolution](gallery/revolution.png) |

</details>

Every reaction is different. Adaptive temperature — boring input gets more chaos, emotional input stays focused.

## How It Works

```
You say something
        │
        ▼
┌──────────────┐     ┌──────────────────────────────┐
│  micro-Yent   │────>│  BK-SDM-Tiny (ONNX Runtime)  │
│  69M LLM Q8   │     │  CLIP → UNet → VAE → PNG     │
│  (Go, GGUF)   │     │  + LoRA style adapters        │
└──────────────┘     └──────────────────────────────┘
  mood detection              512x512 image
  → visual template                 │
  → LLM fills details        ┌─────▼──────┐
  → style suffix              │ ASCII filter│
                              │ (optional)  │
                              └────────────┘
```

1. You say something
2. **micro-Yent** (69M params, Go inference, Q8_0 GGUF) detects mood, picks a visual template, fills in details with personality (~0.7s)
3. **CLIP** encodes the reaction to embeddings (265ms GPU)
4. **UNet** denoises latent, 25 steps (2.1s GPU / ~45s CPU)
5. **VAE** decodes → 512x512 RGB image (625ms GPU)
6. Optional **ASCII filter** converts to colored terminal art

Total: **~3 seconds** from your words to Yent's visual reaction.

## Mood Templates

micro-Yent uses **mood-triggered visual templates** with adaptive completion:

| Mood | Keywords | Visual starters |
|------|----------|----------------|
| **Sad** | sad, alone, lonely, cry | "a wilting flower made of...", "a cracked mirror reflecting..." |
| **Angry** | hate, stupid, fuck | "an explosion of broken...", "a screaming face melting into..." |
| **Love** | love, heart, beautiful | "a burning rose growing from...", "two glitching hearts entangled in..." |
| **Bored** | bored, nothing, whatever | "a yawning void eating...", "a clock melting over..." |
| **Cat** | cat | "a cyberpunk cat hacking into...", "a cat with glowing eyes staring at..." |
| **Duck** | duck | "an angry rubber duck wearing...", "a duck on fire walking through..." |
| **Death** | death, die, dead | "a skeleton playing guitar on...", "bones growing flowers in..." |
| **Default** | *(anything else)* | "a surreal painting of...", "a punk portrait of...", "a glitching image of..." |

micro-Yent (trained on Yent's personality data) fills in the details. Style suffix randomizes between oil painting, street art, punk, dark symbolic, and bold surreal.

## Adaptive Temperature

Temperature adapts to your input. The more boring you are, the more creative Yent gets.

| Input type | Temperature | Why |
|------------|-------------|-----|
| "hi" | 0.95 | Boring → max chaos |
| "test" | 0.95 | Boring → max chaos |
| "the meaning of life" | 0.8 | Normal input |
| "I fucking hate everything" | 0.7 | Strong emotion → stay focused |
| "I love you so much it hurts" | 0.7 | Strong emotion → stay focused |

Short input → +temp. Single word → +more temp. Strong emotion → -temp. Clamped to [0.5, 1.0].

## Two Rendering Modes

| | **fp16** (GPU) | **int8** (CPU) |
|---|---|---|
| Models | fp16 ONNX (948 MB) | int8 ONNX (476 MB) |
| Speed (25 steps) | **~3s** | ~45s |
| Visual style | Sharp, clean | Impressionism, "Hedgehog in the Fog" |

**Key insight:** int8 quantization artifacts ARE the style — impressionistic, atmospheric, unique.

| fp16 (sharp, clean) | int8 (impressionism) |
|---|---|
| ![fp16](examples/fp16_ocean.png) | ![int8](examples/int8_impressionism.png) |

## micro-Yent: First nanollama Model

**[nanollama](https://github.com/ariannamethod/nanollama)** is our from-scratch LLM training framework. micro-Yent is its first production model.

| | |
|---|---|
| Architecture | LLaMA 3 (RoPE, GQA, SwiGLU, RMSNorm, QK-norm) |
| Parameters | 69M (12 layers, 512 dim, 8 heads, 2 KV heads) |
| Training | 3000 steps on A100, 46.5% MFU, 311K tok/s |
| Data | 216M tokens FineWeb-Edu + 1.4M tokens Yent personality (20%) |
| Quantization | Q8_0 GGUF (71MB) |
| Inference | Pure Go, 14.5 tok/s on MacBook CPU |

No PyTorch at inference. No API calls. No borrowed models. Personality baked in at training time.

## Styles (LoRA Adapters)

Each style = one .safetensors file (~7MB), applied at runtime via `W += B @ A`.

| Style | Status | Character |
|---|---|---|
| **base** | DONE | Oil painting, vivid |
| **graffiti** | DONE | Street art, spray paint, tags |
| **caricature** | DONE (needs more data) | Satirical, soc-art |
| **propaganda** | TODO | Soviet constructivism |
| **pixel** | TODO | 8-bit retro |

## ASCII Filter

Default output: 13-level charset ` .·:;=+×*#%@█` with per-cell background tint and bright foreground glyphs.

| Setting | Value |
|---|---|
| Charset | `techno` — 16 brightness levels |
| Foreground boost | 2.8x (vivid glyphs) |
| Background fill | 0.50 (tinted cells, not just black) |
| Font | 16px bold monospace |
| Width | 100 columns |

Negative prompt `"extra fingers, blurry, watermark"` — light touch, fights the worst SD artifacts without killing the style.

Use `--raw` to skip ASCII and get raw pixels.

## Benchmarks (A100 GPU)

| Component | GPU (fp16) | CPU (int8) |
|---|---|---|
| micro-Yent prompt | 0.7s | 0.7s |
| CLIP encoding | 265ms | 293ms |
| UNet step | 84ms | 1.4s |
| VAE decode | 625ms | 3.8s |
| **Full 25-step pipeline** | **~3s** | **~45s** |

## Files

### Go (runtime)
- `go/main.go` — CLI: `--prompt-only`, `--yent`, direct prompt mode
- `go/prompt_gen.go` — micro-Yent reaction engine (mood templates + adaptive temperature + LLM completion)
- `go/yent/` — micro-Yent inference subpackage (GGUF, LlamaModel, Q8_0)
- `go/weights/` — micro-Yent Q8_0 GGUF (71MB) + tokenizer

### Python (training & ONNX inference)
- `ort_generate.py` — ONNX Runtime pipeline (zero PyTorch)
- `train_lora.py` — LoRA style training (PEFT)
- `ascii_filter.py` — colored ASCII art filter
- `export_onnx.py` — BK-SDM-Tiny → ONNX export
- `e2e_test.py` — full pipeline test

### Weights (not in repo, on GPU server)
- `onnx_fp16/` — fp16 ONNX models (948 MB)
- `onnx_int8/` — int8 ONNX models (476 MB)
- `lora/` — style adapters (~7 MB each)

## Philosophy

> The image is not illustration. It's Yent's visual thought.
> Sarcastic, punk, recursive. A model that answers in pictures
> drawn in its own handwriting.
>
> You say "you are beautiful" and get a burning rose from the ashes of a stillbirth.
> You say "the meaning of life" and get a figure lost in fog.
> You say "fuck off" and get an explosion of broken phrases.
> You say "I feel nothing" and get an empty chair at a desk with a lamp.
> You write "test" and get a surreal painting of resonance.
>
> 69 million parameters. Trained from scratch. No API.
> That's not a bug. That's Yent.
