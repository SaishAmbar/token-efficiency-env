"""
TokenEfficiencyEnv — Before/After Evaluation Script
=====================================================

Runs baseline (untrained) and trained models through all 24 questions
in the prompt bank and produces a comparison table for judges.

This directly addresses Hackathon Guide §19:
    "baseline model attempt → trained model attempt → measurable improvement"

Usage:
    # Evaluate base model only (no trained model needed):
    python evaluate_before_after.py --base-only

    # Full comparison (after training):
    python evaluate_before_after.py --trained-model ./token-efficiency-grpo

    # Quick test with 3 questions:
    python evaluate_before_after.py --questions 3 --base-only

    # Save results to file:
    python evaluate_before_after.py --output results/comparison.json
"""

import sys
import os
import re
import json
import argparse
import logging
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "token_efficiency_env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OpenEnv", "src"))

from prompts import PROMPT_BANK
from scorer import score_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("evaluate")

# ─── Constants ──────────────────────────────────────────────────────
MAX_TOKEN_LIMIT = 200
MIN_BUDGET = 1
MAX_BUDGET = 200
MAX_ANSWER_TOKENS = 500
REPETITION_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are a concise, accurate question answerer.

You MUST respond in this exact format — no other text:
<budget>N</budget><answer>your answer here</answer>

Where:
- <budget>N</budget> = the number of tokens you plan to use (integer)
- <answer>...</answer> = your concise answer

Budget guidelines:
- Easy questions: budget 15–30
- Medium questions: budget 60–90
- Hard questions: budget 120–160

Example:
Question: "What is the capital of France?"
Response: <budget>15</budget><answer>Paris.</answer>
"""


def count_tokens(text):
    """Rough token count."""
    return max(1, int(len(text.split()) * 1.3))


def evaluate_response(task, raw_response):
    """Score a model's response against a task."""
    budget_match = re.search(r"<budget>(\d+)</budget>", raw_response)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)

    if not budget_match or not answer_match:
        return {
            "reward": -1.0,
            "error": "bad_format",
            "format_ok": False,
            "budget": 0,
            "tokens_used": 0,
            "answer": raw_response[:100],
            "details": {},
        }

    budget = max(MIN_BUDGET, min(MAX_BUDGET, int(budget_match.group(1))))
    answer = answer_match.group(1).strip()

    if not answer:
        return {
            "reward": -1.0, "error": "empty", "format_ok": True,
            "budget": budget, "tokens_used": 0, "answer": "", "details": {},
        }

    words = answer.lower().split()
    if len(words) > 3:
        wc = Counter(words)
        top_w, top_c = wc.most_common(1)[0]
        if top_c / len(words) > REPETITION_THRESHOLD:
            return {
                "reward": -0.5, "error": "repetition", "format_ok": True,
                "budget": budget, "tokens_used": count_tokens(answer),
                "answer": answer[:100], "details": {},
            }

    tokens_used = count_tokens(answer)
    if tokens_used > MAX_ANSWER_TOKENS:
        return {
            "reward": -0.5, "error": "too_long", "format_ok": True,
            "budget": budget, "tokens_used": tokens_used,
            "answer": answer[:100], "details": {},
        }

    result = score_answer(
        prompt=task["prompt"],
        response=answer,
        allocated_budget=budget,
        tokens_used=tokens_used,
        complexity=task.get("complexity", "medium"),
        expected_keywords=task.get("expected_keywords", []),
    )

    return {
        "reward": result["reward"],
        "error": None,
        "format_ok": True,
        "budget": budget,
        "tokens_used": tokens_used,
        "answer": answer[:200],
        "details": result["details"],
    }


def generate_response(model, tokenizer, question, device="cuda"):
    """Generate a response from a model for a given question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def run_evaluation(model, tokenizer, questions, device="cuda", label="model"):
    """Run a model through all questions and collect results."""
    results = []

    for i, task in enumerate(questions):
        raw_response = generate_response(model, tokenizer, task["prompt"], device)
        eval_result = evaluate_response(task, raw_response)

        results.append({
            "question_idx": i,
            "question": task["prompt"],
            "complexity": task["complexity"],
            "raw_response": raw_response[:300],
            **eval_result,
        })

        status = "✅" if eval_result["reward"] > 0.5 else ("⚠️" if eval_result["reward"] > 0 else "❌")
        logger.info(
            "%s [%s] Q%d (%s): reward=%.3f | tokens=%d | %s",
            status, label, i + 1, task["complexity"],
            eval_result["reward"], eval_result["tokens_used"],
            task["prompt"][:40],
        )

    return results


def print_comparison_table(base_results, trained_results=None):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("  TOKEN EFFICIENCY ENV — EVALUATION RESULTS")
    print("=" * 100)

    if trained_results:
        print(f"\n{'Question':<45} | {'Base':>8} | {'Trained':>8} | {'Δ':>6} | {'Base Tok':>8} | {'Train Tok':>9}")
        print("-" * 100)

        for b, t in zip(base_results, trained_results):
            delta = t["reward"] - b["reward"]
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            delta_color = "🟢" if delta > 0.1 else ("🟡" if delta > -0.1 else "🔴")

            print(f"  {b['question'][:43]:<43} | {b['reward']:>7.3f} | {t['reward']:>7.3f} | {delta_color}{delta_str:>5} | {b['tokens_used']:>7} | {t['tokens_used']:>8}")

        print("-" * 100)

        base_avg = sum(r["reward"] for r in base_results) / len(base_results)
        train_avg = sum(r["reward"] for r in trained_results) / len(trained_results)
        delta_avg = train_avg - base_avg
        base_tokens = sum(r["tokens_used"] for r in base_results) / max(len(base_results), 1)
        train_tokens = sum(r["tokens_used"] for r in trained_results) / max(len(trained_results), 1)
        base_format = sum(1 for r in base_results if r["format_ok"]) / len(base_results) * 100
        train_format = sum(1 for r in trained_results if r["format_ok"]) / len(trained_results) * 100

        print(f"  {'AVERAGE':<43} | {base_avg:>7.3f} | {train_avg:>7.3f} | {'🟢' if delta_avg > 0 else '🔴'}{delta_avg:>+5.2f} | {base_tokens:>7.1f} | {train_tokens:>8.1f}")
        print(f"  {'FORMAT COMPLIANCE':<43} | {base_format:>6.0f}% | {train_format:>6.0f}% |       |         |")

    else:
        print(f"\n{'Question':<50} | {'Reward':>8} | {'Tokens':>7} | {'Format':>7} | {'Complexity':>10}")
        print("-" * 100)

        for r in base_results:
            fmt = "✅" if r["format_ok"] else "❌"
            print(f"  {r['question'][:48]:<48} | {r['reward']:>7.3f} | {r['tokens_used']:>6} | {fmt:>6} | {r['complexity']:>10}")

        print("-" * 100)
        avg_reward = sum(r["reward"] for r in base_results) / len(base_results)
        avg_tokens = sum(r["tokens_used"] for r in base_results) / max(len(base_results), 1)
        format_rate = sum(1 for r in base_results if r["format_ok"]) / len(base_results) * 100

        print(f"  {'AVERAGE':<48} | {avg_reward:>7.3f} | {avg_tokens:>6.1f} | {format_rate:>5.0f}% |")

    print("=" * 100)


def save_results(base_results, trained_results, output_path):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "results": base_results,
            "avg_reward": sum(r["reward"] for r in base_results) / len(base_results),
            "avg_tokens": sum(r["tokens_used"] for r in base_results) / max(len(base_results), 1),
            "format_compliance": sum(1 for r in base_results if r["format_ok"]) / len(base_results),
        },
    }

    if trained_results:
        data["trained"] = {
            "results": trained_results,
            "avg_reward": sum(r["reward"] for r in trained_results) / len(trained_results),
            "avg_tokens": sum(r["tokens_used"] for r in trained_results) / max(len(trained_results), 1),
            "format_compliance": sum(1 for r in trained_results if r["format_ok"]) / len(trained_results),
        }
        data["improvement"] = {
            "reward_delta": data["trained"]["avg_reward"] - data["baseline"]["avg_reward"],
            "token_delta": data["baseline"]["avg_tokens"] - data["trained"]["avg_tokens"],
            "format_delta": data["trained"]["format_compliance"] - data["baseline"]["format_compliance"],
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("📄 Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TokenEfficiencyEnv models")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct", help="Baseline model")
    parser.add_argument("--trained-model", default=None, help="Path to trained model")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--questions", type=int, default=None, help="Number of questions (default: all 24)")
    parser.add_argument("--output", default="results/comparison.json", help="Output JSON path")
    parser.add_argument("--device", default="auto", help="Device: cuda, cpu, or auto")
    args = parser.parse_args()

    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Device: %s", device)

    # Select questions
    questions = PROMPT_BANK
    if args.questions:
        questions = PROMPT_BANK[:args.questions]

    logger.info("Evaluating %d questions", len(questions))

    # ─── Load base model ────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading baseline model: %s", args.base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        base_model = base_model.to(device)

    # ─── Evaluate base model ────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Evaluating BASELINE model...")
    logger.info("=" * 50)
    base_results = run_evaluation(base_model, base_tokenizer, questions, device, "BASE")

    # ─── Evaluate trained model ─────────────────────────────────
    trained_results = None
    if not args.base_only and args.trained_model:
        logger.info("=" * 50)
        logger.info("Evaluating TRAINED model: %s", args.trained_model)
        logger.info("=" * 50)

        trained_tokenizer = AutoTokenizer.from_pretrained(args.trained_model)
        if trained_tokenizer.pad_token is None:
            trained_tokenizer.pad_token = trained_tokenizer.eos_token

        trained_model = AutoModelForCausalLM.from_pretrained(
            args.trained_model,
            torch_dtype="auto",
            device_map=device if device == "cuda" else None,
        )
        if device == "cpu":
            trained_model = trained_model.to(device)

        trained_results = run_evaluation(
            trained_model, trained_tokenizer, questions, device, "TRAINED"
        )

    # ─── Print comparison ───────────────────────────────────────
    print_comparison_table(base_results, trained_results)

    # ─── Save results ───────────────────────────────────────────
    save_results(base_results, trained_results, args.output)

    # ─── Summary for judges ─────────────────────────────────────
    print("\n📋 JUDGE SUMMARY:")
    base_avg = sum(r["reward"] for r in base_results) / len(base_results)
    print(f"   Baseline avg reward: {base_avg:.3f}")
    if trained_results:
        train_avg = sum(r["reward"] for r in trained_results) / len(trained_results)
        print(f"   Trained avg reward:  {train_avg:.3f}")
        print(f"   Improvement:         {train_avg - base_avg:+.3f} ({((train_avg - base_avg) / max(abs(base_avg), 0.01)) * 100:+.0f}%)")
        base_tok = sum(r["tokens_used"] for r in base_results) / max(len(base_results), 1)
        train_tok = sum(r["tokens_used"] for r in trained_results) / max(len(trained_results), 1)
        print(f"   Token reduction:     {base_tok:.0f} → {train_tok:.0f} ({((base_tok - train_tok) / max(base_tok, 1)) * 100:.0f}% fewer)")


if __name__ == "__main__":
    main()
