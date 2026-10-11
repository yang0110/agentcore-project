#!/bin/bash
# One-time prep for MODEL=qwen3.5-9B (INSIDE slime_trainer, repo at /work):
#   1. hf/Qwen3.5-9B-nothink: the HF snapshot (symlinked) with a chat template whose thinking defaults
#      OFF — slime's AnthropicAdapter renders the template with no enable_thinking kwarg, and Qwen3.5
#      ignores the /no_think soft switch, so a <think> trace would eat the whole max_tokens budget.
#      The (video_)preprocessor configs are left out: with them slime loads Qwen3.5's vision processor
#      and then rejects our text prompts ("prompt must be a list when processor is not None").
#      SGLang, though, must load that processor for this architecture, so hf/Qwen3.5-9B-nothink-sglang adds
#      them back and run.sh points --sglang-tokenizer-path at it.
#   2. ckpt/Qwen3.5-9B_torch_dist: the Megatron base checkpoint run.sh loads (--ref-load/--load).
#     docker exec slime_trainer bash -lc 'cd /work && bash code/demos/demo_terminalbench_kernelbench/prepare_qwen35.sh'
set -euo pipefail
REPO=Qwen/Qwen3.5-9B
OUT=/work/code/p4_slime_train
NOTHINK=$OUT/hf/Qwen3.5-9B-nothink
SGL_TOK=$OUT/hf/Qwen3.5-9B-nothink-sglang
TORCH_DIST=$OUT/ckpt/Qwen3.5-9B_torch_dist

SNAP=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s(\"$REPO\"))" 2>/dev/null | tail -1)
[ -f "$SNAP/config.json" ] || { echo "download of $REPO failed" >&2; exit 1; }

mkdir -p "$NOTHINK"
for f in "$SNAP"/*; do
  case "$(basename "$f")" in chat_template.jinja|tokenizer_config.json|*preprocessor_config.json) ;; *) ln -sfn "$f" "$NOTHINK/" ;; esac
done
python3 - "$SNAP" "$NOTHINK" <<'EOF'
import json, sys
snap, out = sys.argv[1:]
OLD = "{%- if enable_thinking is defined and enable_thinking is false %}"
NEW = "{%- if not (enable_thinking is defined and enable_thinking) %}"
tpl = open(f"{snap}/chat_template.jinja").read()
assert tpl.count(OLD) == 1, "Qwen3.5 chat template changed upstream; re-check the thinking switch"
open(f"{out}/chat_template.jinja", "w").write(tpl.replace(OLD, NEW))
cfg = json.load(open(f"{snap}/tokenizer_config.json"))
if isinstance(cfg.get("chat_template"), str):
    cfg["chat_template"] = cfg["chat_template"].replace(OLD, NEW)
json.dump(cfg, open(f"{out}/tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
EOF
mkdir -p "$SGL_TOK"
for f in "$NOTHINK"/*; do ln -sfn "$f" "$SGL_TOK/"; done
for f in "$SNAP"/*preprocessor_config.json; do ln -sfn "$f" "$SGL_TOK/"; done
python3 - "$NOTHINK" <<'EOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
s = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
assert s.endswith("<think>\n\n</think>\n\n"), repr(s[-80:])
print("chat template: thinking off by default")
EOF

if [ -f "$TORCH_DIST/latest_checkpointed_iteration.txt" ]; then
  echo "$TORCH_DIST exists — skipping conversion"
else
  source /root/slime/scripts/models/qwen3.5-9B.sh
  mkdir -p "$TORCH_DIST"
  cd /root/slime
  PYTHONPATH=/root/Megatron-LM:/root/slime python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --hf-checkpoint "$NOTHINK" --save "$TORCH_DIST"
fi
echo "PREPARE_DONE $NOTHINK $SGL_TOK $TORCH_DIST"
