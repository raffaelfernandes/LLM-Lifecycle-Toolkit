#!/usr/bin/env python3
"""
validar_dataset_cot.py
======================
Validação e filtragem de qualidade do dataset CoT gerado para a Questão 4.
Adaptado do validate_dataset.py usado nas Questões 2/3.

Verifica:
- Campos obrigatórios e tipos.
- Tamanhos mínimos de reasoning e answer.
- Perguntas dependentes de contexto.
- Duplicatas por hash da instrução.
- Consistência simples reasoning -> answer (a resposta deve ter ligação
  lexical com o raciocínio; pega casos em que o reasoning "foge" da answer).
- Detecção de idioma não-português (heurística leve).

Gera estatísticas e escreve um arquivo limpo + um relatório.

Uso:
    python validar_dataset_cot.py --entrada dataset_cot_docentesDC.json \
        --saida dataset_cot_limpo.json
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MIN_REASONING_LEN = 30
MIN_ANSWER_LEN = 20
MIN_INSTRUCTION_LEN = 15

# marcadores de dependência de contexto (verificados na instruction E no reasoning)
CTX_FLAGS = ("trecho", "texto acima", "no texto", "acima mencionad",
             "segundo o texto", "de acordo com o texto", "o texto menciona",
             "o texto descreve", "conforme o texto", "nas linhas", "na linha",
             "linhas 2", "o texto afirma", "no contexto apresentado",
             "partes do trecho", "identificar as partes")

# stopwords PT comuns — se quase nenhuma aparecer, texto pode não ser PT
PT_HINTS = (" de ", " que ", " para ", " com ", " uma ", " os ", " as ",
            " da ", " do ", " em ", " é ", " e ", " ou ", " na ", " no ",
            " um ", " ao ", " se ", " por ", " são ", " dos ", " das ",
            " como ", " mais ", " ser ", " sua ", " seu ", " pode ", " entre ")

# termos que indicam foco em Ciência da Computação (sinal positivo)
CC_TERMOS = (
    "algoritmo", "programa", "código", "software", "hardware", "dados",
    "estrutura", "função", "variável", "compilador", "linguagem", "memória",
    "processo", "thread", "rede", "protocolo", "banco de dados", "sql",
    "árvore", "grafo", "lista", "pilha", "fila", "hash", "complexidade",
    "autômato", "linguagem regular", "máquina de turing", "recursão",
    "orienta", "classe", "objeto", "herança", "polimorf", "compil",
    "sistema operacional", "processador", "bit", "byte", "binário",
    "aprendizado de máquina", "rede neural", "inteligência artificial",
    "modelo", "treinamento", "servidor", "cliente", "api", "container",
    "docker", "kernel", "cache", "buffer", "criptograf", "encript",
    "blockchain", "computa", "inferência", "tipo", "vetor", "matriz",
    "sinal", "fourier", "imagem", "pixel", "frequência", "otimiza",
)

# termos de assuntos claramente FORA de CC (sinal negativo forte)
FORA_CC = (
    "avião", "piloto", "pilotagem", "aeronave", "voo",
    "direitos autorais", "fair use", "filme", "cena", "série de tv",
    "furto", "roubo", "boletim de ocorrência",
    "nota final", "instrumento de avaliação", "média ponderada das notas",
    "arqueolog", "diretor",
)


def foco_em_cc(instr: str, reas: str, ans: str) -> bool:
    """Conservador: só rejeita se houver assunto claramente fora de CC E
    nenhum termo de CC presente. Evita descartar pares bons por engano."""
    texto = f" {instr} {reas} {ans} ".lower()
    tem_cc = any(t in texto for t in CC_TERMOS)
    tem_fora = any(t in texto for t in FORA_CC)
    if tem_fora and not tem_cc:
        return False
    return True


def hash_instr(texto: str) -> str:
    norm = re.sub(r"\s+", " ", texto.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def parece_portugues(texto: str) -> bool:
    """Heurística robusta: combina stopwords com marcadores fortes de PT
    (acentuação e sufixos como -ção, -ável). Evita falsos positivos em
    respostas técnicas curtas."""
    t = f" {texto.lower()} "
    hits = sum(1 for h in PT_HINTS if h in t)
    # marcadores fortes: acentos e terminações típicas do português
    tem_acento = any(c in texto for c in "áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ")
    tem_sufixo = any(s in t for s in ("ção", "ções", "ável", "ível", "mente ",
                                       "dade ", "agem ", "ência"))
    if tem_acento or tem_sufixo:
        return hits >= 1        # com sinal forte, basta 1 stopword
    return hits >= 2            # sem sinal forte, exige 2


def tokens(texto: str):
    return set(re.findall(r"[a-zà-ú]{4,}", texto.lower()))


def consistente(reasoning: str, answer: str) -> bool:
    """A resposta deve compartilhar algum termo relevante com o raciocínio."""
    ra, an = tokens(reasoning), tokens(answer)
    if not an:
        return False
    # se a answer é muito curta em termos, exigência relaxada
    if len(an) <= 2:
        return True
    return len(ra & an) >= 1


def validar(args):
    entrada = Path(args.entrada)
    with open(entrada, "r", encoding="utf-8") as f:
        pares = json.load(f)

    print(f"[entrada] {len(pares)} pares carregados de {entrada}")

    limpos = []
    hashes = set()
    motivos = Counter()
    tam_instr, tam_reas, tam_ans = [], [], []

    for par in pares:
        # campos
        if not all(k in par for k in ("instruction", "reasoning", "answer")):
            motivos["campos_faltando"] += 1
            continue
        instr = str(par["instruction"]).strip()
        reas = str(par["reasoning"]).strip()
        ans = str(par["answer"]).strip()

        if len(instr) < MIN_INSTRUCTION_LEN:
            motivos["instrucao_curta"] += 1
            continue
        if len(reas) < MIN_REASONING_LEN:
            motivos["reasoning_curto"] += 1
            continue
        if len(ans) < MIN_ANSWER_LEN:
            motivos["answer_curta"] += 1
            continue
        if any(flag in instr.lower() for flag in CTX_FLAGS):
            motivos["dependente_contexto"] += 1
            continue
        if any(flag in reas.lower() for flag in CTX_FLAGS):
            motivos["contexto_no_reasoning"] += 1
            continue
        if not parece_portugues(reas) or not parece_portugues(ans):
            motivos["idioma_suspeito"] += 1
            continue
        if not consistente(reas, ans):
            motivos["inconsistente"] += 1
            continue
        if not foco_em_cc(instr, reas, ans):
            motivos["fora_de_cc"] += 1
            continue

        h = hash_instr(instr)
        if h in hashes:
            motivos["duplicata"] += 1
            continue
        hashes.add(h)

        limpos.append({
            "instruction": instr,
            "input": par.get("input", ""),
            "reasoning": reas,
            "answer": ans,
        })
        tam_instr.append(len(instr))
        tam_reas.append(len(reas))
        tam_ans.append(len(ans))

    # estatísticas
    def media(xs):
        return sum(xs) / len(xs) if xs else 0

    print("\n========== RELATÓRIO DE VALIDAÇÃO ==========")
    print(f"Válidos:    {len(limpos)}")
    print(f"Removidos:  {len(pares) - len(limpos)}")
    print("\nMotivos de remoção:")
    for motivo, n in motivos.most_common():
        print(f"  {motivo:22s} {n}")
    print("\nTamanhos médios (caracteres):")
    print(f"  instruction: {media(tam_instr):.0f}")
    print(f"  reasoning:   {media(tam_reas):.0f}")
    print(f"  answer:      {media(tam_ans):.0f}")
    print("============================================\n")

    saida = Path(args.saida)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(limpos, f, ensure_ascii=False, indent=2)
    print(f"[saída] {len(limpos)} pares limpos escritos em {saida}")

    if args.relatorio:
        rel = Path(args.relatorio)
        with open(rel, "w", encoding="utf-8") as f:
            f.write("RELATÓRIO DE VALIDAÇÃO — dataset CoT (Questão 4)\n")
            f.write(f"Entrada: {entrada}\n")
            f.write(f"Válidos: {len(limpos)} | Removidos: {len(pares)-len(limpos)}\n\n")
            for motivo, n in motivos.most_common():
                f.write(f"{motivo}: {n}\n")
        print(f"[saída] relatório em {rel}")


def main():
    p = argparse.ArgumentParser(description="Validação do dataset CoT (Q4)")
    p.add_argument("--entrada", default="dataset_cot_docentesDC.json")
    p.add_argument("--saida", default="dataset_cot_limpo.json")
    p.add_argument("--relatorio", default="validacao_cot_relatorio.txt")
    args = p.parse_args()
    validar(args)


if __name__ == "__main__":
    main()
