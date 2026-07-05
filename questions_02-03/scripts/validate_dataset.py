"""
Validador e inspetor do dataset SFT gerado.
Uso: python validate_sft_dataset.py sft_dataset_docentesDC.json
"""

import json
import sys
from pathlib import Path
from collections import Counter


CONTEXT_KEYWORDS = [
    "neste documento", "neste texto", "neste slide", "nesta página",
    "no texto acima", "no documento acima", "mencionado acima",
    "descrito acima", "listados acima", "apresentado acima",
    "conforme o texto", "segundo o texto", "de acordo com o texto",
    "this document", "this text", "this slide",
    "o autor menciona", "o docente menciona",
    "nesse contexto", "nesse trecho", "nessa seção",
    "na lista acima", "no exemplo acima",
]


def check_pair(pair: dict, idx: int) -> list[str]:
    """Retorna lista de problemas encontrados no par."""
    issues = []
    
    inst = pair.get("instruction", "")
    out  = pair.get("output", "")
    
    if len(inst.strip()) < 15:
        issues.append("instruction muito curta")
    if len(out.strip()) < 30:
        issues.append("output muito curto")
    
    full = (inst + " " + out).lower()
    for kw in CONTEXT_KEYWORDS:
        if kw.lower() in full:
            issues.append(f"referência contextual: '{kw}'")
            break
    
    if not inst.strip().endswith("?") and not any(
        inst.lower().startswith(w) for w in
        ["defina", "explique", "descreva", "liste", "compare", "analise",
         "cite", "quais", "qual", "como", "por que", "quando", "o que", "explicite"]
    ):
        issues.append("instruction pode não ser uma pergunta/comando claro")
    
    return issues


def main(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"📂 Dataset: {path}")
    print(f"   Total de pares: {len(data)}\n")

    problems = []
    instruction_lengths = []
    output_lengths = []
    
    for i, pair in enumerate(data):
        issues = check_pair(pair, i)
        instruction_lengths.append(len(pair.get("instruction", "")))
        output_lengths.append(len(pair.get("output", "")))
        if issues:
            problems.append((i, pair.get("instruction", "")[:60], issues))

    valid = len(data) - len(problems)
    print(f"✅ Pares válidos:     {valid}/{len(data)}")
    print(f"⚠  Pares com problemas: {len(problems)}")
    print(f"\nEstatísticas:")
    print(f"  Instruction — média: {sum(instruction_lengths)//len(instruction_lengths)} chars")
    print(f"  Output      — média: {sum(output_lengths)//len(output_lengths)} chars")
    
    if problems:
        print(f"\nPrimeiros 10 problemas:")
        for idx, inst_preview, issues in problems[:10]:
            print(f"  [{idx:04d}] {inst_preview!r}")
            for iss in issues:
                print(f"         → {iss}")

    # Distribuição por tipo de pergunta
    types = Counter()
    for pair in data:
        inst = pair.get("instruction", "").lower().strip()
        if inst.startswith("o que"): types["o que é"] += 1
        elif inst.startswith("como"): types["como funciona"] += 1
        elif inst.startswith("qual"): types["qual/quais"] += 1
        elif inst.startswith("por que"): types["por que"] += 1
        elif inst.startswith("explique"): types["explique"] += 1
        elif inst.startswith("compare"): types["compare"] += 1
        elif inst.startswith("defina"): types["defina"] += 1
        elif inst.startswith("explicite"): types["explicite"] += 1
        else: types["outros"] += 1

    print("\nDistribuição por tipo de pergunta:")
    for t, c in types.most_common():
        bar = "█" * (c * 40 // len(data))
        print(f"  {t:<20} {c:>4}  {bar}")

    # Amostra aleatória
    import random
    sample = random.sample(data, min(3, len(data)))
    print(f"\n📋 Amostras aleatórias:")
    for s in sample:
        print(f"  instruction: {s['instruction']}")
        print(f"  output: {s['output'][:120]}...")
        print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "questao2/sft_dataset_docentesDC.json"
    main(path)