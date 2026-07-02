#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Step 0: 從 HuggingFace 下載模型
# ─────────────────────────────────────────────────────────────
set -e

# Qwen3.6-27B: 28B 參數 (BF16, ~56GB), 多模態 (Image-Text-to-Text),
# 原生 262K 上下文。需要較新版 vLLM 才支援其架構 (Gated DeltaNet / MTP)。
MODEL_REPO="${1:-Qwen/Qwen3.6-27B}"

# 本地存放目錄
LOCAL_DIR="${2:-./models/$(basename "$MODEL_REPO")}"

# 開啟高速下載(需要 hf_transfer)
export HF_HUB_ENABLE_HF_TRANSFER=1

# 如果是 gated / 私有模型,先登入(新版 CLI 用 hf auth login):
#   hf auth login

echo "==> 下載 $MODEL_REPO"
echo "==> 存到 $LOCAL_DIR"

# 此 repo 全為 safetensors (15 shards, ~56GB),無 original/ 或 .pth,故不需 --exclude。
hf download "$MODEL_REPO" \
    --local-dir "$LOCAL_DIR"

echo ""
echo "==> 下載完成。模型路徑:"
echo "    $LOCAL_DIR"
echo ""
echo "==> 接著啟動 vLLM (需 vllm 版本夠新以支援 Qwen3.6 架構):"
echo "    vllm serve $LOCAL_DIR --port 8000 --served-model-name qwen3.6-27b"
