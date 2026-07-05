import argparse
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "llm_pretraining" / "data_dompi_2025_tucano2_10k"
OUTPUT_DIR = BASE_DIR / "llm_pretraining" / "runs" / "tucano2_qwen_0p5b_loraplus_dompi_10k"
DEFAULT_MODEL = "Polygl0t/Tucano2-qwen-0.5B-Base"


class TokenBlockDataset(Dataset):
    def __init__(self, input_ids, block_size, stride):
        if len(input_ids) < block_size + 1:
            raise ValueError(f"Corpus tokenizado tem {len(input_ids)} tokens, menor que block_size={block_size}.")
        self.input_ids = input_ids
        self.block_size = block_size
        self.starts = list(range(0, len(input_ids) - block_size, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.block_size
        ids = self.input_ids[start:end]
        if isinstance(ids, torch.Tensor):
            return ids.to(dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-treinamento continuado de LLM em diarios de prefeituras com avaliacao antes/depois."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="0 treina por todas as epocas.")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=96, help="0 avalia o split inteiro.")
    parser.add_argument("--benchmark-max-items", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp32", action="store_true", help="Desativa mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Usa bfloat16 se a GPU suportar.")
    parser.add_argument("--no-lora", action="store_true", help="Treina todos os pesos do modelo.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-loraplus", action="store_true", help="Usa o otimizador LoRA+ quando LoRA esta ativo.")
    parser.add_argument("--loraplus-lr-ratio", type=float, default=16.0)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Modulos alvo separados por virgula.",
    )
    parser.add_argument("--resume-from", default="", help="Caminho de checkpoint ou 'latest'.")
    parser.add_argument("--rebuild-token-cache", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=96)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Este Python nao tem CUDA ativo no PyTorch. Rode llm_pretraining/setup_gpu_windows.ps1 "
            "ou instale torch com wheel CUDA antes do treino longo."
        )


def model_slug(model_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


def load_tokenizer(model_id, trust_remote_code):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def token_cache_path(data_dir, model_id, split):
    cache_dir = data_dir / "token_cache" / model_slug(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}.pt"


def load_or_build_tokens(data_dir, tokenizer, model_id, split, rebuild):
    cache_path = token_cache_path(data_dir, model_id, split)
    if cache_path.exists() and not rebuild:
        cached = torch.load(cache_path, map_location="cpu")
        return cached["input_ids"]

    corpus_path = data_dir / "corpus" / f"{split}.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus nao encontrado: {corpus_path}")

    text = corpus_path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    tensor = torch.tensor(ids, dtype=torch.long)
    torch.save({"model_id": model_id, "split": split, "input_ids": tensor}, cache_path)
    return tensor


def build_loader(input_ids, block_size, stride, batch_size, shuffle, seed):
    dataset = TokenBlockDataset(input_ids, block_size=block_size, stride=stride)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def latest_checkpoint(output_dir):
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.split("-")[-1]), path))
            except ValueError:
                pass
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def resolve_resume_path(output_dir, resume_from):
    if not resume_from:
        return None
    if resume_from == "latest":
        return latest_checkpoint(output_dir)
    return Path(resume_from)


def load_model(args, device):
    dtype = torch.float32 if args.fp32 else torch.float16
    if args.bf16:
        dtype = torch.bfloat16

    resume_path = resolve_resume_path(args.output_dir, args.resume_from)
    model = AutoModelForCausalLM.from_pretrained(
        resume_path if args.no_lora and resume_path else args.model_id,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    use_lora = not args.no_lora
    if use_lora:
        try:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA requer a biblioteca peft. Rode llm_pretraining/setup_gpu_windows.ps1 "
                "ou instale com: python -m pip install peft accelerate"
            ) from exc

        if resume_path and (resume_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        else:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
            )
            model = get_peft_model(model, config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.to(device)
    model.train()
    return model, resume_path


def trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def lr_scale(step, warmup_steps, total_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate_lm(model, loader, device, amp_dtype, use_amp, max_batches):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    batches = 0

    for batch in tqdm(loader, desc="avaliacao", leave=False):
        batch = batch.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_ids=batch, labels=batch)
            logits = outputs.logits
            loss = outputs.loss

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels.ne(-100)

        tokens = int(mask.sum().item())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
        correct += int((predictions.eq(shift_labels) & mask).sum().item())
        batches += 1
        if max_batches and batches >= max_batches:
            break

    cross_entropy = total_loss / max(1, total_tokens)
    perplexity = math.exp(cross_entropy) if cross_entropy < 50 else float("inf")
    token_accuracy = correct / max(1, total_tokens)
    model.train()
    return {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "tokens": total_tokens,
        "batches": batches,
    }


def normalize_answer(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def bleu_unigram(reference, prediction):
    ref_tokens = normalize_answer(reference).split()
    pred_tokens = normalize_answer(prediction).split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    precision = overlap / len(pred_tokens)
    brevity = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0
    return precision * brevity


def token_f1(reference, prediction):
    ref_tokens = normalize_answer(reference).split()
    pred_tokens = normalize_answer(prediction).split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rubric_recall(item, prediction):
    termos = item.get("rubric_must_include") or []
    if not termos:
        return None
    resposta = normalize_answer(prediction)
    acertos = 0
    for termo in termos:
        termo_normalizado = normalize_answer(termo)
        if termo_normalizado and termo_normalizado in resposta:
            acertos += 1
    return acertos / max(1, len(termos))


def formatar_alternativas(item):
    alternativas = item.get("alternativas") or []
    if not alternativas:
        return ""
    linhas = ["Alternativas:"]
    for alternativa in alternativas:
        linhas.append(f"{alternativa['letra']}) {alternativa['texto']}")
    return "\n".join(linhas) + "\n"


def resposta_correta(item, resposta_modelo):
    formato = item.get("formato", "generate_until")
    esperado = item.get("resposta_referencia", "")
    normalizado = normalize_answer(resposta_modelo)
    if formato == "multiple_choice":
        letra = str(item.get("gabarito", "")).strip().upper()
        texto_correto = ""
        for alternativa in item.get("alternativas") or []:
            if alternativa.get("letra") == letra:
                texto_correto = alternativa.get("texto", "")
                break
        primeira = normalizado[:1].upper()
        return primeira == letra or normalize_answer(texto_correto) in normalizado
    recall = rubric_recall(item, resposta_modelo)
    if recall is not None:
        return recall >= 0.6
    return normalize_answer(esperado) in normalizado


@torch.no_grad()
def evaluate_benchmark(model, tokenizer, benchmark_path, output_path, device, max_items, block_size, max_new_tokens=96):
    if not benchmark_path.exists():
        return {"items": 0, "exact_match": None, "output": str(output_path)}

    model.eval()
    items = []
    with benchmark_path.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            if linha.strip():
                items.append(json.loads(linha))
            if max_items and len(items) >= max_items:
                break

    acertos = 0
    bleu_total = 0.0
    f1_total = 0.0
    rubric_total = 0.0
    rubric_items = 0
    por_formato = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as arquivo:
        for item in tqdm(items, desc="benchmark", leave=False):
            instrucoes = "Responda de forma curta usando apenas o contexto."
            if item.get("formato") == "multiple_choice":
                instrucoes = "Responda apenas com a letra da alternativa correta, usando o contexto."
            prompt = (
                f"{instrucoes}\n\n"
                f"Contexto:\n{item['contexto']}\n\n"
                f"{formatar_alternativas(item)}"
                f"Pergunta: {item['pergunta']}\n"
                "Resposta:"
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max(block_size, 1024),
            ).to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            answer_ids = generated[0, inputs["input_ids"].shape[1] :]
            linhas_resposta = tokenizer.decode(answer_ids, skip_special_tokens=True).strip().splitlines()
            resposta_modelo = linhas_resposta[0].strip() if linhas_resposta else ""
            correto = resposta_correta(item, resposta_modelo)
            bleu = bleu_unigram(item["resposta_referencia"], resposta_modelo)
            f1 = token_f1(item["resposta_referencia"], resposta_modelo)
            rubric = rubric_recall(item, resposta_modelo)
            acertos += int(correto)
            bleu_total += bleu
            f1_total += f1
            if rubric is not None:
                rubric_total += rubric
                rubric_items += 1
            formato = item.get("formato", "generate_until")
            por_formato.setdefault(
                formato,
                {"items": 0, "correct": 0, "bleu_unigrama": 0.0, "token_f1": 0.0, "rubric_recall": 0.0, "rubric_items": 0},
            )
            por_formato[formato]["items"] += 1
            por_formato[formato]["correct"] += int(correto)
            por_formato[formato]["bleu_unigrama"] += bleu
            por_formato[formato]["token_f1"] += f1
            if rubric is not None:
                por_formato[formato]["rubric_recall"] += rubric
                por_formato[formato]["rubric_items"] += 1
            arquivo.write(
                json.dumps(
                    {
                        **item,
                        "resposta_modelo": resposta_modelo,
                        "correto_exato_normalizado": correto,
                        "bleu_unigrama": bleu,
                        "token_f1": f1,
                        "rubric_recall": rubric,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    model.train()
    for formato, stats in por_formato.items():
        stats["accuracy"] = stats["correct"] / max(1, stats["items"])
        stats["bleu_unigrama"] = stats["bleu_unigrama"] / max(1, stats["items"])
        stats["token_f1"] = stats["token_f1"] / max(1, stats["items"])
        stats["rubric_recall"] = stats["rubric_recall"] / max(1, stats["rubric_items"])
    return {
        "items": len(items),
        "accuracy_or_exact_match": acertos / max(1, len(items)),
        "bleu_unigrama": bleu_total / max(1, len(items)),
        "token_f1": f1_total / max(1, len(items)),
        "rubric_recall": rubric_total / max(1, rubric_items) if rubric_items else None,
        "por_formato": por_formato,
        "output": str(output_path),
    }


def save_checkpoint(model, tokenizer, optimizer, output_dir, global_step, metrics):
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "metrics": metrics,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


def append_metrics(output_dir, record):
    path = output_dir / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_optimizer_state_if_available(optimizer, resume_path):
    if not resume_path:
        return 0
    state_path = resume_path / "trainer_state.pt"
    if not state_path.exists():
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("global_step", 0))


def build_optimizer(model, args):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if args.use_loraplus and not args.no_lora:
        try:
            from peft.optimizers import create_loraplus_optimizer
        except ImportError as exc:
            raise RuntimeError("LoRA+ requer peft.optimizers.create_loraplus_optimizer.") from exc
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=args.learning_rate,
            loraplus_lr_ratio=args.loraplus_lr_ratio,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def main():
    args = parse_args()
    set_seed(args.seed)
    require_cuda()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_amp = not args.fp32

    tokenizer = load_tokenizer(args.model_id, args.trust_remote_code)
    train_ids = load_or_build_tokens(args.data_dir, tokenizer, args.model_id, "train", args.rebuild_token_cache)
    valid_ids = load_or_build_tokens(args.data_dir, tokenizer, args.model_id, "valid", args.rebuild_token_cache)
    test_ids = load_or_build_tokens(args.data_dir, tokenizer, args.model_id, "test", args.rebuild_token_cache)

    train_loader = build_loader(
        train_ids, args.block_size, args.stride, args.train_batch_size, shuffle=True, seed=args.seed
    )
    valid_loader = build_loader(
        valid_ids, args.block_size, args.stride, args.eval_batch_size, shuffle=False, seed=args.seed
    )
    test_loader = build_loader(
        test_ids, args.block_size, args.stride, args.eval_batch_size, shuffle=False, seed=args.seed
    )

    model, resume_path = load_model(args, device)
    trainable, total = trainable_parameters(model)
    optimizer = build_optimizer(model, args)
    global_step = load_optimizer_state_if_available(optimizer, resume_path)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    total_updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    planned_steps = args.max_steps or (total_updates_per_epoch * args.epochs)
    benchmark_path = args.data_dir / "municipal_gazettes_benchmark.jsonl"
    benchmark_dir = args.output_dir / "benchmark_outputs"

    run_config = {
        "model_id": args.model_id,
        "output_dir": str(args.output_dir),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "use_lora": not args.no_lora,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100 * trainable / max(1, total),
        "block_size": args.block_size,
        "stride": args.stride,
        "epochs": args.epochs,
        "planned_steps": planned_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "optimizer": "LoRA+" if args.use_loraplus and not args.no_lora else "AdamW",
        "loraplus_lr_ratio": args.loraplus_lr_ratio if args.use_loraplus and not args.no_lora else None,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    before_valid = evaluate_lm(model, valid_loader, device, amp_dtype, use_amp, args.eval_max_batches)
    before_test = evaluate_lm(model, test_loader, device, amp_dtype, use_amp, args.eval_max_batches)
    before_benchmark = evaluate_benchmark(
        model,
        tokenizer,
        benchmark_path,
        benchmark_dir / "before.jsonl",
        device,
        args.benchmark_max_items,
        args.block_size,
        args.benchmark_max_new_tokens,
    )
    before_record = {
        "phase": "before_training",
        "global_step": global_step,
        "valid": before_valid,
        "test": before_test,
        "benchmark": before_benchmark,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_metrics(args.output_dir, before_record)
    print(json.dumps(before_record, ensure_ascii=False, indent=2))

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    stop_training = False
    progress = tqdm(total=planned_steps, initial=min(global_step, planned_steps), desc="treino")

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            scale = lr_scale(global_step, args.warmup_steps, planned_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.item() * args.grad_accum_steps), lr=optimizer.param_groups[0]["lr"])

                if args.eval_every and global_step % args.eval_every == 0:
                    metrics = evaluate_lm(model, valid_loader, device, amp_dtype, use_amp, args.eval_max_batches)
                    record = {
                        "phase": "during_training",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "valid": metrics,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    append_metrics(args.output_dir, record)
                    print(json.dumps(record, ensure_ascii=False))

                if args.save_every and global_step % args.save_every == 0:
                    checkpoint = save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        args.output_dir,
                        global_step,
                        {"epoch": epoch + 1},
                    )
                    print(f"Checkpoint salvo em {checkpoint}")

                if args.max_steps and global_step >= args.max_steps:
                    stop_training = True
                    break

        if stop_training:
            break

    progress.close()

    final_valid = evaluate_lm(model, valid_loader, device, amp_dtype, use_amp, args.eval_max_batches)
    final_test = evaluate_lm(model, test_loader, device, amp_dtype, use_amp, args.eval_max_batches)
    final_benchmark = evaluate_benchmark(
        model,
        tokenizer,
        benchmark_path,
        benchmark_dir / "after.jsonl",
        device,
        args.benchmark_max_items,
        args.block_size,
        args.benchmark_max_new_tokens,
    )

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    after_record = {
        "phase": "after_training",
        "global_step": global_step,
        "valid": final_valid,
        "test": final_test,
        "benchmark": final_benchmark,
        "final_model": str(final_dir),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_metrics(args.output_dir, after_record)

    resumo = {
        "before": before_record,
        "after": after_record,
        "delta": {
            "valid_cross_entropy": final_valid["cross_entropy"] - before_valid["cross_entropy"],
            "valid_perplexity": final_valid["perplexity"] - before_valid["perplexity"],
            "valid_token_accuracy": final_valid["token_accuracy"] - before_valid["token_accuracy"],
            "test_cross_entropy": final_test["cross_entropy"] - before_test["cross_entropy"],
            "test_perplexity": final_test["perplexity"] - before_test["perplexity"],
            "test_token_accuracy": final_test["token_accuracy"] - before_test["token_accuracy"],
        },
    }
    (args.output_dir / "summary_before_after.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
