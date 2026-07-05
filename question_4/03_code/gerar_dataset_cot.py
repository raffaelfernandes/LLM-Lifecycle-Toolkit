#!/usr/bin/env python3
"""
gerar_dataset_cot.py
====================
Geração de dataset sintético no formato Chain-of-Thought (CoT) para a
Questão 4 (Destilação de Conhecimento) — Tópicos em IA, DC/UFPI 2026.1.

Teacher: Qwen2.5-7B-Instruct (via Ollama)
Fonte:   vickminari/docentesDC (HuggingFace)
Saída:   pares {instruction, input, reasoning, answer}

Características:
- Geração incremental com checkpoint (retomável se a sessão cair).
- Deduplicação por hash da instrução.
- Prompt que EXIGE raciocínio passo a passo antes da resposta final.

Uso:
    python gerar_dataset_cot.py --n-alvo 1000 --modelo qwen2.5:7b-instruct-q4_K_M
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Dependências opcionais — importadas de forma tolerante para o script poder
# ser inspecionado mesmo sem o ambiente completo (Colab instala on-the-fly).
# ----------------------------------------------------------------------------
try:
    import ollama
except ImportError:
    ollama = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ============================================================================
# Configuração
# ============================================================================
CHUNK_SIZE = 600          # caracteres por chunk (igual Q2/Q3)
CHUNK_OVERLAP = 300       # 50% de sobreposição
MIN_REASONING_LEN = 30    # filtro mínimo de raciocínio
MIN_ANSWER_LEN = 20       # filtro mínimo de resposta
SEED = 42

PROMPT_TEACHER = """Você é um professor de Ciência da Computação. Com base no \
trecho abaixo, crie UMA pergunta conceitual independente de contexto (que faça \
sentido sem o trecho) e responda-a em DUAS partes claramente separadas.

TRECHO:
\"\"\"{chunk}\"\"\"

Responda ESTRITAMENTE no formato JSON abaixo, sem texto fora do JSON:
{{
  "instruction": "<pergunta conceitual de CC, independente de contexto>",
  "reasoning": "<raciocínio passo a passo que leva à resposta, 2 a 4 passos>",
  "answer": "<resposta final concisa e correta>"
}}

Regras:
- A pergunta NÃO pode citar 'o trecho', 'o texto' ou 'acima'.
- O reasoning deve mostrar o caminho lógico, não só repetir a resposta.
- Tudo em português."""


# ============================================================================
# Utilidades
# ============================================================================
def hash_instr(texto: str) -> str:
    """Hash normalizado da instrução para deduplicação."""
    norm = re.sub(r"\s+", " ", texto.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def chunk_documento(texto: str, size: int, overlap: int):
    """Divide um documento em chunks com sobreposição."""
    texto = re.sub(r"\s+", " ", texto).strip()
    passo = max(1, size - overlap)
    for i in range(0, len(texto), passo):
        trecho = texto[i:i + size]
        if len(trecho) >= 100:  # ignora restos minúsculos
            yield trecho


# marcadores de conteúdo que NÃO rende boa pergunta conceitual em português
_CODE_HINTS = ("#include", "printf", "int main", "void ", "return ", "public ",
               "import ", "def ", "System.out", "console.log", "</", "/>",
               "{\n", "();", "std::", "->", "==", "!=")
_EN_STOP = (" the ", " and ", " is ", " are ", " of ", " with ", " that ",
            " this ", " for ", " from ", " was ", " were ", " which ")
_PT_STOP = (" de ", " que ", " para ", " com ", " uma ", " os ", " as ",
            " da ", " do ", " em ", " é ", " ou ", " na ", " no ", " são ",
            " ao ", " se ", " por ", " como ")


def chunk_aproveitavel(trecho: str) -> bool:
    """Pré-filtro: descarta chunks ruidosos ANTES de gastar chamada ao teacher.

    Rejeita: código-fonte, texto majoritariamente em inglês, listas de nomes,
    formulários/tabelas de notas e trechos sem densidade de português.
    """
    t = trecho.lower()

    # 1. código-fonte
    if sum(1 for h in _CODE_HINTS if h in t) >= 2:
        return False
    # densidade de símbolos típicos de código
    simbolos = sum(t.count(c) for c in ("{", "}", ";", "()", "[]", "//"))
    if simbolos >= 6:
        return False

    # 2. idioma: precisa ter portugues e nao ser dominado por ingles
    pt = sum(1 for w in _PT_STOP if w in f" {t} ")
    en = sum(1 for w in _EN_STOP if w in f" {t} ")
    if pt < 3:            # pouca densidade de português
        return False
    if en > pt:           # mais inglês que português
        return False

    # 3. listas de nomes / formulários / tabelas de notas
    #    heurística: muitos números soltos ou muitas palavras Capitalizadas
    digitos = sum(c.isdigit() for c in trecho)
    if digitos / max(1, len(trecho)) > 0.15:   # >15% dígitos = tabela/matrícula
        return False
    palavras = trecho.split()
    if palavras:
        caps = sum(1 for w in palavras if w[:1].isupper())
        if caps / len(palavras) > 0.5:          # >50% capitalizadas = lista de nomes
            return False

    # 4. precisa ter alguma pontuação de texto corrido
    if trecho.count(".") + trecho.count(",") < 2:
        return False

    return True


def extrair_json(resposta: str):
    """Extrai o primeiro objeto JSON de uma resposta do LLM."""
    # remove cercas de código se houver
    resposta = resposta.replace("```json", "").replace("```", "")
    inicio = resposta.find("{")
    fim = resposta.rfind("}")
    if inicio == -1 or fim == -1 or fim < inicio:
        return None
    bloco = resposta[inicio:fim + 1]
    try:
        return json.loads(bloco)
    except json.JSONDecodeError:
        return None


def par_valido(par: dict) -> bool:
    """Validação leve no momento da geração (a validação pesada é à parte)."""
    if not isinstance(par, dict):
        return False
    for campo in ("instruction", "reasoning", "answer"):
        if campo not in par or not isinstance(par[campo], str):
            return False
    if len(par["reasoning"].strip()) < MIN_REASONING_LEN:
        return False
    if len(par["answer"].strip()) < MIN_ANSWER_LEN:
        return False
    # rejeita perguntas que dependem do contexto
    instr_low = par["instruction"].lower()
    if any(t in instr_low for t in ("trecho", "texto acima", "no texto", "acima")):
        return False
    return True


# ============================================================================
# Carregamento de checkpoint
# ============================================================================
def carregar_checkpoint(caminho: Path):
    """Carrega pares já gerados e o conjunto de hashes vistos."""
    pares, hashes = [], set()
    if caminho.exists():
        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    par = json.loads(linha)
                    pares.append(par)
                    hashes.add(hash_instr(par["instruction"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return pares, hashes


def append_par(caminho: Path, par: dict):
    """Acrescenta um par ao arquivo JSONL (escrita incremental segura)."""
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(par, ensure_ascii=False) + "\n")


# ============================================================================
# Geração
# ============================================================================
def gerar(args):
    if ollama is None:
        sys.exit("ERRO: pacote 'ollama' não instalado. `pip install ollama`")
    if load_dataset is None:
        sys.exit("ERRO: pacote 'datasets' não instalado. `pip install datasets`")

    ckpt = Path(args.checkpoint)
    pares, hashes = carregar_checkpoint(ckpt)
    print(f"[checkpoint] {len(pares)} pares já existentes, retomando...")

    print(f"[dataset] carregando {args.fonte} ...")
    ds = load_dataset(args.fonte, split="train")
    # campo textual: tenta 'text', senão concatena tudo
    campo = "text" if "text" in ds.column_names else ds.column_names[0]

    # monta lista de chunks, aplicando o pré-filtro de qualidade
    chunks = []
    brutos = 0
    for doc in ds[campo]:
        if isinstance(doc, str) and doc.strip():
            for ch in chunk_documento(doc, CHUNK_SIZE, CHUNK_OVERLAP):
                brutos += 1
                if chunk_aproveitavel(ch):
                    chunks.append(ch)
    print(f"[dataset] {brutos} chunks brutos -> {len(chunks)} aproveitáveis "
          f"({100*len(chunks)/max(1,brutos):.0f}% após pré-filtro)")

    import random
    random.seed(SEED)
    random.shuffle(chunks)

    gerados = len(pares)
    tentativas = 0
    idx = 0

    while gerados < args.n_alvo and idx < len(chunks):
        chunk = chunks[idx]
        idx += 1
        tentativas += 1

        prompt = PROMPT_TEACHER.format(chunk=chunk)
        try:
            resp = ollama.chat(
                model=args.modelo,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "seed": SEED + tentativas},
            )
            conteudo = resp["message"]["content"]
        except Exception as e:  # rede/Ollama instável → não perde progresso
            print(f"  [aviso] falha na chamada ({e}); aguardando 3s")
            time.sleep(3)
            continue

        par = extrair_json(conteudo)
        if par is None or not par_valido(par):
            continue

        h = hash_instr(par["instruction"])
        if h in hashes:
            continue

        par_final = {
            "instruction": par["instruction"].strip(),
            "input": "",
            "reasoning": par["reasoning"].strip(),
            "answer": par["answer"].strip(),
        }
        append_par(ckpt, par_final)
        hashes.add(h)
        gerados += 1

        if gerados % 25 == 0:
            taxa = gerados / max(1, tentativas)
            print(f"  [{gerados}/{args.n_alvo}] taxa de aproveitamento: {taxa:.1%}")

    print(f"[fim] {gerados} pares gerados em {tentativas} tentativas")
    print(f"[fim] checkpoint salvo em {ckpt}")

    # exporta versão final consolidada em JSON (lista)
    pares_final, _ = carregar_checkpoint(ckpt)
    saida = Path(args.saida)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(pares_final, f, ensure_ascii=False, indent=2)
    print(f"[fim] {len(pares_final)} pares exportados para {saida}")


# ============================================================================
# CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Geração de dataset CoT (Questão 4)")
    p.add_argument("--fonte", default="vickminari/docentesDC",
                   help="dataset HuggingFace de origem")
    p.add_argument("--modelo", default="qwen2.5:7b-instruct-q4_K_M",
                   help="modelo teacher no Ollama")
    p.add_argument("--n-alvo", type=int, default=1000,
                   help="número de pares válidos desejado")
    p.add_argument("--checkpoint", default="dataset_cot_checkpoint.jsonl",
                   help="arquivo JSONL incremental (retomável)")
    p.add_argument("--saida", default="dataset_cot_docentesDC.json",
                   help="arquivo JSON final consolidado")
    args = p.parse_args()
    gerar(args)


if __name__ == "__main__":
    main()
