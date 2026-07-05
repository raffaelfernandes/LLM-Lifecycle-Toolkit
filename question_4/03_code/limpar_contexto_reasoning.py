#!/usr/bin/env python3
"""
limpar_contexto_reasoning.py
============================
Recupera pares que seriam descartados por mencionar "o trecho"/"o texto" no
reasoning. Em vez de descartar, reescreve o raciocínio removendo as frases-muleta
introdutórias, tornando-o independente de contexto.

Estratégia:
- Divide o reasoning em sentenças.
- Remove/reescreve prefixos-muleta ("O trecho menciona que", "No texto,", etc.).
- Se uma sentença INTEIRA depende do trecho (não sobra conteúdo útil), ela é
  removida.
- Se após a limpeza o reasoning ficar curto demais ou vazio, o par é descartado.
- Pares sem menção a contexto passam intactos.

Uso:
    python limpar_contexto_reasoning.py --entrada dataset_cot_docentesDC.json \
        --saida dataset_cot_reasoning_limpo.json
"""
import argparse
import json
import re

MIN_REASONING_FINAL = 40  # se sobrar menos que isso, descarta

# prefixos-muleta a remover do INÍCIO de uma sentença. A captura pega o que vem
# depois para reaproveitar o conteúdo. Ordem importa (mais específico primeiro).
PREFIXOS = [
    r"o trecho menciona que\s+",
    r"o trecho menciona\s+",
    r"o trecho descreve que\s+",
    r"o trecho descreve\s+",
    r"o trecho indica que\s+",
    r"o trecho indica\s+",
    r"o trecho afirma que\s+",
    r"o trecho apresenta\s+",
    r"o trecho fala sobre\s+",
    r"o trecho\s+",
    r"o texto menciona que\s+",
    r"o texto menciona\s+",
    r"o texto descreve que\s+",
    r"o texto descreve\s+",
    r"o texto indica que\s+",
    r"o texto afirma que\s+",
    r"no texto,?\s+",
    r"segundo o texto,?\s+",
    r"conforme o texto,?\s+",
    r"de acordo com o texto,?\s+",
    r"com base no trecho,?\s+",
    r"a partir do trecho,?\s+",
]

# sentenças que são PURAMENTE sobre o trecho (sem conteúdo reaproveitável)
SO_CONTEXTO = [
    r"^identificar as partes do trecho.*",
    r"^analisando o trecho.*",
    r"^observando o trecho.*",
    r"^lendo o trecho.*",
    r"^com base nas informações do trecho.*",
    r"^o trecho fornece.*informaç.*",
]

# marcadores que, se ainda restarem, indicam dependência real
RESIDUAL = ("trecho", "no texto", "o texto menciona", "o texto descreve",
            "nas linhas", "na linha")


def limpar_sentenca(s: str) -> str:
    orig = s
    low = s.lower().strip()

    # sentença puramente contextual -> remove inteira
    for pat in SO_CONTEXTO:
        if re.match(pat, low):
            return ""

    # remove prefixo-muleta e capitaliza o que sobra
    for pat in PREFIXOS:
        m = re.match(pat, low)
        if m:
            resto = s[m.end():].strip()
            if resto:
                resto = resto[0].upper() + resto[1:]
            return resto
    return orig.strip()


def limpar_reasoning(reasoning: str):
    # separa em sentenças preservando numeração tipo "1)" "2."
    partes = re.split(r'(?<=[.!?])\s+', reasoning.strip())
    novas = []
    for p in partes:
        # preserva marcador de lista no início (1), 2., etc.)
        marcador = ""
        m = re.match(r'^(\d+[\)\.]\s*)', p)
        if m:
            marcador = m.group(1)
            p = p[m.end():]
        limpa = limpar_sentenca(p)
        if limpa:
            novas.append(marcador + limpa)
    texto = " ".join(novas).strip()
    # normaliza espaços
    texto = re.sub(r'\s+', ' ', texto)
    return texto


def processar(args):
    with open(args.entrada, encoding="utf-8") as f:
        pares = json.load(f)

    limpos, recuperados, descartados, intactos = [], 0, 0, 0
    for par in pares:
        reas = par["reasoning"]
        tinha_ctx = any(t in reas.lower() for t in RESIDUAL)

        if not tinha_ctx:
            limpos.append(par)
            intactos += 1
            continue

        novo = limpar_reasoning(reas)
        # ainda tem resíduo de dependência ou ficou curto? descarta
        if any(t in novo.lower() for t in RESIDUAL) or len(novo) < MIN_REASONING_FINAL:
            descartados += 1
            continue

        par = dict(par)
        par["reasoning"] = novo
        limpos.append(par)
        recuperados += 1

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(limpos, f, ensure_ascii=False, indent=2)

    print(f"[entrada]      {len(pares)} pares")
    print(f"[intactos]     {intactos} (sem menção a contexto)")
    print(f"[recuperados]  {recuperados} (reasoning reescrito)")
    print(f"[descartados]  {descartados} (dependência real ou ficou curto)")
    print(f"[saída]        {len(limpos)} pares -> {args.saida}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", default="dataset_cot_docentesDC.json")
    p.add_argument("--saida", default="dataset_cot_reasoning_limpo.json")
    args = p.parse_args()
    processar(args)


if __name__ == "__main__":
    main()
