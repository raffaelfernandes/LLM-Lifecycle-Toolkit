import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from train_continued_pretraining_gpu import (
    BASE_DIR,
    DEFAULT_MODEL,
    build_loader,
    evaluate_benchmark,
    evaluate_lm,
    load_or_build_tokens,
    load_tokenizer,
    require_cuda,
    set_seed,
)


DATA_DIR = BASE_DIR / "llm_pretraining" / "data_dompi_2025_tucano2_10k"
OUTPUT_DIR = BASE_DIR / "llm_pretraining" / "runs" / "tucano2_qwen_0p5b_loraplus_dompi_10k" / "baseline_before_training"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia o modelo base no DOMPI-2025 antes do pre-treino continuado."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-batches", type=int, default=96, help="0 avalia o split inteiro.")
    parser.add_argument("--benchmark-max-items", type=int, default=100)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--rebuild-token-cache", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    require_cuda()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_amp = not args.fp32
    dtype = torch.float32 if args.fp32 else amp_dtype

    tokenizer = load_tokenizer(args.model_id, args.trust_remote_code)
    valid_ids = load_or_build_tokens(args.data_dir, tokenizer, args.model_id, "valid", args.rebuild_token_cache)
    test_ids = load_or_build_tokens(args.data_dir, tokenizer, args.model_id, "test", args.rebuild_token_cache)

    valid_loader = build_loader(
        valid_ids, args.block_size, args.stride, args.eval_batch_size, shuffle=False, seed=args.seed
    )
    test_loader = build_loader(
        test_ids, args.block_size, args.stride, args.eval_batch_size, shuffle=False, seed=args.seed
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    benchmark_path = args.data_dir / "municipal_gazettes_benchmark.jsonl"
    respostas_path = args.output_dir / "benchmark_before.jsonl"

    valid = evaluate_lm(model, valid_loader, device, amp_dtype, use_amp, args.eval_max_batches)
    test = evaluate_lm(model, test_loader, device, amp_dtype, use_amp, args.eval_max_batches)
    benchmark = evaluate_benchmark(
        model,
        tokenizer,
        benchmark_path,
        respostas_path,
        device,
        args.benchmark_max_items,
        args.block_size,
        args.benchmark_max_new_tokens,
    )

    resultado = {
        "phase": "baseline_before_training",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": args.model_id,
        "data_dir": str(args.data_dir),
        "benchmark_path": str(benchmark_path),
        "metricas_linguagem": {
            "valid": valid,
            "test": test,
            "observacao": "cross_entropy e a entropia cruzada media por token; perplexity = exp(cross_entropy).",
        },
        "benchmark": benchmark,
        "arquivos": {
            "resultado_json": str((args.output_dir / "resultado_baseline_before.json").resolve()),
            "respostas_benchmark": str(respostas_path.resolve()),
        },
    }

    (args.output_dir / "resultado_baseline_before.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
