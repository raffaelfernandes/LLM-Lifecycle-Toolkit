import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from train_continued_pretraining_gpu import evaluate_benchmark, load_tokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Polygl0t/Tucano2-qwen-0.5B-Base"
DEFAULT_BENCHMARK = BASE_DIR / "llm_pretraining" / "data_dompi_2025" / "benchmark_gold_contextual_25_v2.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "llm_pretraining" / "runs" / "benchmark_25_v2"
DEFAULT_PRETRAIN_ADAPTER = (
    BASE_DIR / "llm_pretraining" / "runs" / "tucano2_qwen_0p5b_loraplus_dompi_10k" / "final"
)
DEFAULT_SFT_ADAPTER = (
    BASE_DIR
    / "llm_pretraining"
    / "runs"
    / "tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250"
    / "final"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia adapters no benchmark DOMPI contextual de 25 questoes.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrain-adapter", type=Path, default=DEFAULT_PRETRAIN_ADAPTER)
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--max-items", type=int, default=25)
    return parser.parse_args()


def load_model(model_id, adapter_path, device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    return model


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_id, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("pretrain_final", args.pretrain_adapter),
        ("sft_v2_balanced", args.sft_adapter),
    ]
    summary = {
        "benchmark": str(args.benchmark.resolve()),
        "model_id": args.model_id,
        "device": device,
        "runs": {},
    }

    for name, adapter_path in runs:
        model = load_model(args.model_id, adapter_path, device)
        output_path = args.output_dir / f"{name}_benchmark_25_v2.jsonl"
        metrics = evaluate_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark_path=args.benchmark,
            output_path=output_path,
            device=device,
            max_items=args.max_items,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
        )
        metrics["adapter_path"] = str(adapter_path.resolve())
        summary["runs"][name] = metrics
        unload(model)

    summary_path = args.output_dir / "summary_benchmark_25_v2.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
