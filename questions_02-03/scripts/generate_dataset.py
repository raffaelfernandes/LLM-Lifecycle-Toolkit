"""
Gerador de pares instruction-response para SFT (Supervised Fine-Tuning)
Dataset: vickminari/docentesDC (HuggingFace)
Hardware alvo: Intel i7-13650HX + 32GB RAM + RTX 3050 6GB

Requisitos:
    pip install datasets ollama tqdm

Modelo recomendado (instalar antes):
    ollama pull qwen2.5:7b-instruct-q4_K_M
    # Alternativa:
    # ollama pull mistral:7b-instruct-q4_K_M
"""

import json
import re
import time
import random
from pathlib import Path
from datasets import load_dataset
import ollama
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
MODEL_NAME      = "qwen2.5:7b-instruct-q4_K_M"   # ~4.1 GB VRAM
OUTPUT_PATH     = Path("questao2/sft_dataset_docentesDC.json")
TARGET_PAIRS    = 1000
CHUNK_SIZE      = 600        # caracteres por chunk (não exceder contexto útil)
PAIRS_PER_CHUNK = 3          # pares gerados por chunk
MAX_RETRIES     = 3          # tentativas por chunk em caso de falha
TEMPERATURE     = 0.7        # diversidade da geração
SEED            = 13

# Palavras-chave que indicam pergunta contextual (serão filtradas)
CONTEXT_KEYWORDS = [
    "neste documento", "neste texto", "neste slide", "nesta página",
    "no texto acima", "no documento acima", "mencionado acima",
    "descrito acima", "listados acima", "apresentado acima",
    "conforme o texto", "segundo o texto", "de acordo com o texto",
    "this document", "this text", "this slide", "the above",
    "o autor", "o professor", "o docente menciona",
    "quantos itens", "quantos tópicos", "quais os X",
    "qual o primeiro", "qual o último", "qual o segundo",
    "nesse contexto", "nesse trecho", "nessa seção",
    "na lista acima", "no exemplo acima",
]

# ─────────────────────────────────────────────
# PROMPT PRINCIPAL
# Engenharia de prompt cuidadosa para evitar
# perguntas que dependem do chunk visto
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Você é um especialista em Ciência da Computação criando um dataset de treinamento.

REGRAS OBRIGATÓRIAS — leia com atenção:
1. Gere perguntas que possam ser respondidas SEM acesso ao texto fornecido.
   - CORRETO: "O que é uma árvore AVL e quais são suas propriedades?"
   - ERRADO: "Quais estruturas de dados são descritas neste documento?"
   - ERRADO: "O que o slide menciona sobre grafos?"
   - ERRADO: "Liste os tópicos apresentados no texto acima."

2. As perguntas devem ser sobre CONCEITOS GERAIS de Ciência da Computação:
   algoritmos, estruturas de dados, sistemas operacionais, redes, banco de dados,
   engenharia de software, linguagens de programação, compiladores, IA, etc.

3. Use o texto apenas como INSPIRAÇÃO para identificar qual conceito abordar.
   A pergunta deve fazer sentido sem o texto.

4. As respostas (output) devem ser completas, didáticas e autossuficientes.

5. Varie os tipos de pergunta:
   - Definição: "O que é X?"
   - Funcionamento: "Como funciona X?"
   - Comparação: "Qual a diferença entre X e Y?"
   - Aplicação: "Quando usar X ao invés de Y?"
   - Vantagem/desvantagem: "Quais as vantagens de X?"
   - Exemplo: "Dê um exemplo de uso de X."

6. FORMATO DE SAÍDA — retorne SOMENTE um array JSON válido, sem markdown, sem explicações:
[
  {
    "instruction": "Pergunta clara e autossuficiente sobre um conceito de CC",
    "input": "",
    "output": "Resposta completa e didática, sem referência a nenhum texto ou documento"
  }
]

Se o input field for relevante (ex: pedir para analisar um trecho de código),
use-o. Caso contrário, deixe como string vazia.
"""

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def load_docentes_dataset() -> list[str]:
    """Carrega o dataset do HuggingFace e extrai os textos."""
    print("📥 Carregando dataset docentesDC...")
    ds = load_dataset("vickminari/docentesDC", split="train")
    
    # Identifica a coluna de texto principal
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    texts = [str(row[text_col]) for row in ds if row[text_col]] # type: ignore
    print(f"   ✓ {len(texts)} documentos carregados (coluna: '{text_col}')")
    return texts


def chunk_texts(texts: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Divide os textos em chunks de tamanho controlado."""
    chunks = []
    for text in texts:
        text = text.strip()
        if len(text) < 80:   # ignora textos muito curtos
            continue
        # divide em chunks sobrepostos para melhor cobertura
        for i in range(0, len(text), chunk_size // 2):
            chunk = text[i : i + chunk_size]
            if len(chunk) >= 80:
                chunks.append(chunk)
    random.shuffle(chunks)
    print(f"   ✓ {len(chunks)} chunks gerados")
    return chunks


def is_contextual(pair: dict) -> bool:
    """Retorna True se o par contém referência contextual inválida."""
    text = (pair.get("instruction", "") + " " + pair.get("output", "")).lower()
    return any(kw.lower() in text for kw in CONTEXT_KEYWORDS)


def is_valid_pair(pair: dict) -> bool:
    """Valida estrutura e qualidade mínima do par."""
    if not isinstance(pair, dict):
        return False
    if "instruction" not in pair or "output" not in pair:
        return False
    if len(pair["instruction"].strip()) < 15:
        return False
    if len(pair["output"].strip()) < 30:
        return False
    if is_contextual(pair):
        return False
    return True


def parse_json_response(response_text: str) -> list[dict]:
    """Extrai JSON da resposta do modelo, tolerando markdown fences."""
    # Remove fences de markdown se presentes
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()
    cleaned = cleaned.strip("`").strip()

    # Tenta extrair apenas o array JSON
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    return []


def generate_pairs_for_chunk(chunk: str, n_pairs: int = PAIRS_PER_CHUNK) -> list[dict]:
    """Gera pares de perguntas e respostas para um chunk de texto."""
    user_message = (
        f"Com base no conteúdo a seguir sobre Ciência da Computação, "
        f"gere {n_pairs} pares de pergunta-resposta seguindo TODAS as regras.\n\n"
        f"CONTEÚDO DE REFERÊNCIA:\n{chunk}\n\n"
        f"Lembre-se: as perguntas devem ser independentes do texto acima."
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": 1200,
                    "seed": SEED + attempt,
                },
            )
            raw = response["message"]["content"]
            pairs = parse_json_response(raw)
            valid = [p for p in pairs if is_valid_pair(p)]
            if valid:
                return valid
        except Exception as e:
            print(f"   ⚠ Tentativa {attempt+1} falhou: {e}")
            time.sleep(2)

    return []


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove pares duplicados com base na instruction."""
    seen = set()
    unique = []
    for p in pairs:
        key = p["instruction"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def load_existing_pairs() -> list[dict]:
    """Carrega pares já gerados do arquivo de saída, se existir."""
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"   ✓ {len(data)} pares existentes carregados de '{OUTPUT_PATH}'")
        return data
    return []


def main():
    random.seed(SEED)

    # 0. Carregar pares já existentes para continuar de onde parou
    all_pairs: list[dict] = load_existing_pairs()
    existing_keys = {p["instruction"].strip().lower() for p in all_pairs}
    already_have = len(all_pairs)

    if already_have >= TARGET_PAIRS:
        print(f"✅ Meta já atingida: {already_have} pares no arquivo. Nada a fazer.")
        return

    remaining = TARGET_PAIRS - already_have
    print(f"\n   Já existem {already_have} pares. Faltam {remaining} para atingir {TARGET_PAIRS}.")

    # 1. Carregar dataset
    texts  = load_docentes_dataset()
    chunks = chunk_texts(texts)

    # Embaralha com seed diferente para explorar chunks não usados antes
    random.seed(SEED + already_have)
    random.shuffle(chunks)

    # 2. Estimar quantos chunks precisamos processar
    chunks_needed = (remaining // PAIRS_PER_CHUNK) + 60  # margem para filtragem
    chunks_to_use = chunks[:chunks_needed]

    print(f"\n🚀 Continuando geração de pares...")
    print(f"   Modelo: {MODEL_NAME}")
    print(f"   Chunks a processar: {len(chunks_to_use)}")
    print(f"   Meta: {TARGET_PAIRS} pares válidos\n")

    # 3. Gerar pares chunk a chunk
    with tqdm(total=TARGET_PAIRS, initial=already_have, desc="Pares válidos gerados") as pbar:
        for i, chunk in enumerate(chunks_to_use):
            if len(all_pairs) >= TARGET_PAIRS:
                break

            new_pairs = generate_pairs_for_chunk(chunk)

            for p in new_pairs:
                if len(all_pairs) >= TARGET_PAIRS:
                    break
                key = p["instruction"].strip().lower()
                if key in existing_keys:
                    continue  # pula duplicatas com os pares já existentes
                existing_keys.add(key)
                all_pairs.append(p)
                pbar.update(1)

            # Salva checkpoint a cada ~100 pares novos
            if (i + 1) % 33 == 0:
                checkpoint = OUTPUT_PATH.with_suffix(".checkpoint.json")
                with open(checkpoint, "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, ensure_ascii=False, indent=2)
                tqdm.write(f"   💾 Checkpoint: {len(all_pairs)} pares salvos")

    # 4. Deduplicar e truncar
    all_pairs = deduplicate(all_pairs)
    final_pairs = all_pairs[:TARGET_PAIRS]

    # 5. Salvar dataset final
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_pairs, f, ensure_ascii=False, indent=2)

    # 6. Relatório final
    print(f"\n✅ Dataset gerado com sucesso!")
    print(f"   Total de pares: {len(final_pairs)}")
    print(f"   Arquivo: {OUTPUT_PATH}")

    # Amostra
    print(f"\n📋 Amostra (primeiro par):")
    if final_pairs:
        sample = final_pairs[0]
        print(f"   instruction: {sample['instruction']}")
        print(f"   input:       {sample.get('input', '')!r}")
        print(f"   output:      {sample['output'][:120]}...")


if __name__ == "__main__":
    main()