# notebooks/

Phase 7 training entrypoints. Read [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
first; this directory is the *runner*, not the design.

## Files

| File | What it is |
|---|---|
| `train_grpo.ipynb` | Main scaffold. Runs end-to-end: setup → baseline eval → GRPO training → trained eval → plots → cleanup. |
| `eval_baseline_vs_trained.py` | Standalone eval harness. Imported by the notebook for both baseline and trained passes; also runnable from the CLI. |
| `__init__.py` | Makes the directory a package so the notebook can import the eval helpers cleanly. |

## Prereqs

```powershell
# From the repo root:
pip install -r requirements-train.txt

# Set the judge API token (required if you want real correctness scoring;
# the notebook will fall back to keyword scoring if it's missing):
$env:HF_TOKEN = "hf_..."
```

A CUDA-capable GPU with **≥16 GB VRAM** is recommended. The defaults
(`Qwen2.5-3B-Instruct` + LoRA r=16 + bf16 + `num_generations=8` +
`max_steps=300`) target that envelope. On a smaller card, drop
`num_generations` to 4 and `per_device_train_batch_size` stays at 1.

## Running the notebook

```powershell
jupyter lab notebooks/train_grpo.ipynb
```

Run cells top-to-bottom. The cost-estimate banner from cell 2 prints how
many judge-API calls the run will make — bail early or switch to
`judge_backend="keyword"` if the number is uncomfortably high.

## Running just the eval (without training)

Useful for sanity-checking a checkpoint or comparing two models without
re-running GRPO:

```powershell
python -m notebooks.eval_baseline_vs_trained `
    --model Qwen/Qwen2.5-3B-Instruct `
    --label baseline `
    --output baseline.json
```

The output JSON drops straight into the notebook's "side-by-side" cell.

## What gets written

```
outputs/grpo_qwen2.5_3b/
  adapter_model.safetensors    # LoRA weights
  adapter_config.json
  metrics.json                 # baseline + trained summaries (notebook cell 13)
  checkpoint-{N}/              # periodic full snapshots (every ~50 steps)
```

Move `metrics.json` somewhere safe before the next run — by default the
notebook overwrites the previous results.
