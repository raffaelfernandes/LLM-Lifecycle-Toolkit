#!/usr/bin/env python3
"""
montar_benchmark_q4.py
======================
Extrai 100 perguntas do sft_dataset_docentesDC.json para servir como benchmark
da Questão 4, de forma ESTRATIFICADA por tipo de pergunta (preserva a mistura
de "o que é", "qual", "como funciona", etc.).

Gera:
- benchmark_q4.json        : 100 itens {id, pergunta, resposta_referencia, categoria}
- excluir_do_treino.json   : instruções (normalizadas) a remover do dataset de
                             treino CoT, para evitar data leakage.

Uso:
    python montar_benchmark_q4.py --fonte sft_dataset_docentesDC.json
"""
import argparse
import json
import re
from collections import Counter, defaultdict


def tipo_pergunta(instr: str) -> str:
    i = instr.lower().strip()
    if i.startswith("o que"): return "o_que_e"
    if i.startswith("qual") or i.startswith("quais"): return "qual_quais"
    if i.startswith("como"): return "como_funciona"
    if i.startswith("defina"): return "defina"
    if i.startswith("compare"): return "compare"
    if i.startswith("explique"): return "explique"
    if i.startswith("por que") or i.startswith("porque"): return "por_que"
    return "outros"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def montar(args):
    with open(args.fonte, encoding="utf-8") as f:
        dados = json.load(f)

    # Anti-leakage: remove da fonte do benchmark perguntas que têm quase-duplicata
    # no dataset de treino CoT (não basta match exato; conceitos repetidos vazam).
    if args.treino:
        import re as _re
        def _toks(s): return set(_re.findall(r"[a-zà-ú]{4,}", s.lower()))
        with open(args.treino, encoding="utf-8") as f:
            treino = json.load(f)
        treino_toks = [_toks(p["instruction"]) for p in treino]
        antes = len(dados)
        filtrados = []
        for p in dados:
            st = _toks(p["instruction"])
            if len(st) < 3:
                filtrados.append(p); continue
            colide = any(len(st & ct) / max(1, len(st | ct)) > args.jaccard
                         for ct in treino_toks)
            if not colide:
                filtrados.append(p)
        dados = filtrados
        print(f"[anti-leakage] {antes} -> {len(dados)} perguntas candidatas "
              f"(removidas {antes-len(dados)} com quase-duplicata no treino)")

    # agrupa por tipo
    grupos = defaultdict(list)
    for p in dados:
        grupos[tipo_pergunta(p["instruction"])].append(p)

    total = len(dados)
    n_alvo = args.n

    # quota proporcional por tipo
    quotas = {t: max(1, round(n_alvo * len(g) / total)) for t, g in grupos.items()}
    # ajuste fino para somar exatamente n_alvo
    while sum(quotas.values()) > n_alvo:
        maior = max(quotas, key=quotas.get); quotas[maior] -= 1
    while sum(quotas.values()) < n_alvo:
        maior = max(grupos, key=lambda t: len(grupos[t])); quotas[maior] += 1

    import random
    random.seed(args.seed)

    benchmark, excluir = [], []
    cid = 1
    for t, g in grupos.items():
        amostra = random.sample(g, min(quotas[t], len(g)))
        for par in amostra:
            benchmark.append({
                "id": cid,
                "pergunta": par["instruction"].strip(),
                "resposta_referencia": par["output"].strip(),
                "categoria": t,
            })
            excluir.append(norm(par["instruction"]))
            cid += 1

    # embaralha a ordem final dos ids
    random.shuffle(benchmark)
    for i, b in enumerate(benchmark, 1):
        b["id"] = i

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    with open(args.exclusao, "w", encoding="utf-8") as f:
        json.dump(excluir, f, ensure_ascii=False, indent=2)

    print(f"[ok] {len(benchmark)} perguntas -> {args.saida}")
    print(f"[ok] {len(excluir)} instruções a excluir -> {args.exclusao}")
    print("\nDistribuição por categoria:")
    for t, n in Counter(b["categoria"] for b in benchmark).most_common():
        print(f"  {t:15s} {n}")
    print("\n>> No treino CoT, filtre os pares cujo norm(instruction) esteja em "
          f"{args.exclusao} para evitar data leakage.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="sft_dataset_docentesDC.json")
    p.add_argument("--saida", default="benchmark_q4.json")
    p.add_argument("--exclusao", default="excluir_do_treino.json")
    p.add_argument("--treino", default=None,
                   help="dataset de treino CoT para remover quase-duplicatas do benchmark")
    p.add_argument("--jaccard", type=float, default=0.7,
                   help="limiar de similaridade para considerar quase-duplicata")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    montar(args)


if __name__ == "__main__":
    main()
