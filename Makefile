# yent.yo — Makefile
#
# Auto-detects hardware and downloads the right weights.
#
# Usage:
#   make setup    — detect hardware, download optimal weights
#   make build    — build Go binary (auto-detects BLAS + ORT)
#   make run      — generate an image (full pipeline)
#   make info     — show detected hardware

HF_REPO = ataeff/yent.yo
GO_DIR = go
WEIGHTS_DIR = weights
ONNX_FP16_DIR = $(WEIGHTS_DIR)/onnx_fp16
ONNX_INT8_DIR = $(WEIGHTS_DIR)/onnx_int8
CLIP_TOK_DIR = $(WEIGHTS_DIR)/clip_tokenizer
YENT_GGUF_DIR = $(WEIGHTS_DIR)/micro-yent

# ---- Hardware Detection ----

# GPU detection
HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
HAS_CUDA := $(if $(HAS_NVIDIA),1,)

# RAM detection (MB)
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  RAM_MB := $(shell sysctl -n hw.memsize 2>/dev/null | awk '{print int($$1/1048576)}')
  HAS_ACCELERATE := 1
  HAS_OPENBLAS :=
else
  RAM_MB := $(shell grep MemTotal /proc/meminfo 2>/dev/null | awk '{print int($$2/1024)}')
  HAS_ACCELERATE :=
  HAS_OPENBLAS := $(shell ldconfig -p 2>/dev/null | grep -q libopenblas && echo 1)
endif

# Weight selection: GPU → fp16, CPU with ≥6GB → int8, else → int8
# fp16 needs ~2GB VRAM (BK-SDM-Tiny is small), int8 needs ~1.5GB RAM
ifeq ($(HAS_CUDA),1)
  ONNX_VARIANT = fp16
  ONNX_DIR = $(ONNX_FP16_DIR)
  ORT_PKG = onnxruntime-gpu
else
  ONNX_VARIANT = int8
  ONNX_DIR = $(ONNX_INT8_DIR)
  ORT_PKG = onnxruntime
endif

# Build tags
BUILD_TAGS :=
ifneq ($(shell python3 -c "import onnxruntime" 2>/dev/null && echo ok),)
  BUILD_TAGS += ort
endif

.PHONY: setup build run run-yent info clean setup-weights setup-tokenizer setup-yent-gguf deps

# ---- Main Targets ----

# Auto-detect and set everything up
setup: info deps setup-weights setup-tokenizer setup-yent-gguf build
	@echo ""
	@echo "=== Ready! ==="
	@echo "  Run:  make run-yent INPUT=\"who are you\""
	@echo "  Or:   make run PROMPT=\"a cat on a roof\""

# Show detected hardware
info:
	@echo "=== Hardware Detection ==="
	@echo "  OS:       $(UNAME)"
	@echo "  RAM:      $(RAM_MB) MB"
	@echo "  GPU:      $(if $(HAS_NVIDIA),$(HAS_NVIDIA),none)"
	@echo "  CUDA:     $(if $(HAS_CUDA),yes,no)"
	@echo "  BLAS:     $(if $(HAS_ACCELERATE),Apple Accelerate,$(if $(HAS_OPENBLAS),OpenBLAS,none))"
	@echo "  Weights:  $(ONNX_VARIANT) ($(if $(HAS_CUDA),GPU fp16 — fast,CPU int8 — reliable))"
	@echo "  ORT pkg:  $(ORT_PKG)"
	@echo ""

# Install Python dependencies
deps:
	@echo "=== Installing dependencies ==="
	pip3 install --quiet huggingface_hub $(ORT_PKG) Pillow numpy 2>/dev/null || \
		pip install --quiet huggingface_hub $(ORT_PKG) Pillow numpy
	@echo "  Dependencies OK"

# Download ONNX weights (auto-selected variant)
setup-weights:
	@if [ -d "$(ONNX_DIR)" ] && [ "$$(ls $(ONNX_DIR)/*.onnx 2>/dev/null | wc -l)" -ge 3 ]; then \
		echo "  $(ONNX_VARIANT) ONNX weights already present."; \
	else \
		echo "Downloading $(ONNX_VARIANT) ONNX weights..."; \
		mkdir -p $(ONNX_DIR); \
		python3 -c "from huggingface_hub import hf_hub_download; \
			[hf_hub_download('$(HF_REPO)', f'weights/onnx_$(ONNX_VARIANT)/{m}', \
			local_dir='.hf_cache') for m in ['clip_text_encoder.onnx','unet.onnx','vae_decoder.onnx']]"; \
		cp .hf_cache/weights/onnx_$(ONNX_VARIANT)/*.onnx $(ONNX_DIR)/; \
		echo "  $(ONNX_VARIANT) ONNX: $$(du -sh $(ONNX_DIR) | cut -f1)"; \
	fi

# Download CLIP tokenizer
setup-tokenizer:
	@if [ -f "$(CLIP_TOK_DIR)/vocab.json" ]; then \
		echo "  CLIP tokenizer already present."; \
	else \
		echo "Downloading CLIP tokenizer..."; \
		mkdir -p $(CLIP_TOK_DIR); \
		python3 -c "from huggingface_hub import snapshot_download; \
			snapshot_download('$(HF_REPO)', local_dir='.hf_cache', \
			allow_patterns='weights/clip_tokenizer/*')"; \
		cp .hf_cache/weights/clip_tokenizer/* $(CLIP_TOK_DIR)/; \
		echo "  Tokenizer OK"; \
	fi

# Download micro-Yent GGUF
setup-yent-gguf:
	@if [ -f "$(YENT_GGUF_DIR)/micro-yent-q8_0.gguf" ]; then \
		echo "  micro-Yent GGUF already present."; \
	else \
		echo "Downloading micro-Yent Q8_0 (71 MB)..."; \
		mkdir -p $(YENT_GGUF_DIR); \
		python3 -c "from huggingface_hub import hf_hub_download; \
			hf_hub_download('$(HF_REPO)', 'weights/micro-yent/micro-yent-q8_0.gguf', \
			local_dir='.hf_cache')"; \
		cp .hf_cache/weights/micro-yent/micro-yent-q8_0.gguf $(YENT_GGUF_DIR)/; \
		echo "  micro-Yent: $$(du -sh $(YENT_GGUF_DIR) | cut -f1)"; \
	fi

# Build Go binary
build:
	@echo "=== Building yentyo ==="
	cd $(GO_DIR) && go build -o ../yentyo .
	@echo "  Built: yentyo ($$(du -h yentyo | cut -f1))"

# ---- Run Targets ----

# Full pipeline with micro-Yent (default mode)
INPUT ?= who are you
SEED ?= 42
run-yent: $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf $(ONNX_DIR)/unet.onnx
	@PROMPT=$$(./yentyo --prompt-only $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf "$(INPUT)" 30 0.8 2>/dev/null | tail -1); \
	WORDS=$$(echo "$$PROMPT" | sed 's/, oil painting.*//' | sed 's/, abstract .*//' | sed 's/, dark symbolic.*//' | sed 's/, street art.*//' | sed 's/, surreal.*//' | sed 's/, Soviet poster.*//'); \
	echo "Input:  $(INPUT)"; \
	echo "Yent:   $$PROMPT"; \
	echo "Words:  $$WORDS"; \
	python3 ort_generate.py $(ONNX_DIR) $(CLIP_TOK_DIR) "$$PROMPT" output_raw.png $(SEED) 10 7.5 --raw; \
	python3 artifact_mask.py output_raw.png output.png --text "$$WORDS"; \
	echo ""; \
	echo "=== Done: output.png ==="

# Direct prompt mode (no micro-Yent)
PROMPT ?= a surreal painting of chaos
run:
	python3 ort_generate.py $(ONNX_DIR) $(CLIP_TOK_DIR) \
		"$(PROMPT)" output.png $(SEED) 10 7.5

# Prompt-only mode (test micro-Yent without image generation)
run-prompt:
	./yentyo --prompt-only $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf "$(INPUT)" 30 0.8

clean:
	rm -f yentyo output.png output_raw.png
	rm -rf $(WEIGHTS_DIR) .hf_cache
