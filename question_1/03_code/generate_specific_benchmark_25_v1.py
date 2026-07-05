import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACOES = BASE_DIR / "extracoes_dompi_2025.jsonl"
DEFAULT_DATA_DIR = BASE_DIR / "llm_pretraining" / "data_dompi_2025_tucano2_10k"
DEFAULT_OUTPUT = BASE_DIR / "llm_pretraining" / "data_dompi_2025" / "benchmark_gold_specific_25_v1.jsonl"
PREVIOUS_BENCHMARKS = [
    BASE_DIR / "llm_pretraining" / "data_dompi_2025" / "benchmark_gold_contextual_25_v2.jsonl",
    BASE_DIR / "llm_pretraining" / "data_dompi_2025" / "benchmark_gold_contextual_corrigido_compacto.jsonl",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Gera benchmark DOMPI especifico de 25 perguntas de resposta curta.")
    parser.add_argument("--extracoes-jsonl", type=Path, default=DEFAULT_EXTRACOES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--context-chars", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def norm(text):
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def clean(text):
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def one_line(text, max_len=260):
    text = clean(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" :-.;")
    return text[:max_len].strip(" :-.;")


def load_ids(path):
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_benchmark_publication_ids(paths):
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                origem = row.get("origem") or {}
                if origem.get("id_publicacao"):
                    ids.add(origem["id_publicacao"])
    return ids


def load_records(extracoes_path, allowed_ids):
    records = {}
    with extracoes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pub_id = record.get("id_publicacao")
            if pub_id not in allowed_ids:
                continue
            text = clean(record.get("texto", ""))
            previous = records.get(pub_id)
            if previous is None or len(text) > len(previous.get("texto", "")):
                record["texto"] = text
                records[pub_id] = record
    return list(records.values())


def field_after(text, labels, max_len=260):
    stop = (
        "CONTRATANTE|CONTRATADA|CONTRATADO|DISTRATANTE|DISTRATADA|VALOR|FONTE|"
        "ASSINATURA|VIGENCIA|VIGNCIA|PRAZO|FUNDAMENTO|OBJETO|DATA|RECURSOS|DOTA"
    )
    for label in labels:
        match = re.search(rf"{label}\s*[:\-]\s*(.+?)(?:\n\s*(?:{stop})\s*[:\-]|\Z)", text, flags=re.I | re.S)
        if match:
            value = trim_embedded_labels(one_line(match.group(1), max_len))
            if valid_field_value(value):
                return value
    return ""


def trim_embedded_labels(value):
    parts = re.split(
        r"\s+(?:OBJETO|SUPORTE\s+LEGAL|FUNDAMENTO|VALOR|DOTA.{0,3}O|VIG.{0,3}NCIA|PRAZO|CONTRATANTE|Torna\s+publico)\s*[:\-]",
        value,
        maxsplit=1,
        flags=re.I,
    )
    return parts[0].strip(" :-.;")


def valid_field_value(value):
    value_norm = norm(value)
    if len(value) < 8 or len(value) > 280:
        return False
    if len(value_norm.split()) < 2:
        return False
    blocked = ["diario oficial", "ano xx", "continua na proxima pagina", "pagina"]
    return not any(term in value_norm for term in blocked)


def valid_contracted(value):
    value_norm = norm(value)
    blocked = [
        "objeto",
        "suporte legal",
        "torna publico",
        "subprocurador",
        "prefeitura",
        "fundamento",
        "artigo",
        "contratacao de empresa",
    ]
    if any(term in value_norm for term in blocked):
        return False
    return len(value_norm.split()) <= 18


def money_value(text):
    match = re.search(r"R\$\s*[0-9.]{1,15},[0-9]{2}", text)
    return match.group(0).strip() if match else ""


def modality(text):
    text_norm = norm(text)
    checks = [
        ("pregao eletronico", "pregao eletronico"),
        ("pregao presencial", "pregao presencial"),
        ("dispensa de licitacao", "dispensa de licitacao"),
        ("inexigibilidade de licitacao", "inexigibilidade de licitacao"),
        ("chamamento publico", "chamamento publico"),
        ("tomada de precos", "tomada de precos"),
    ]
    for key, label in checks:
        if key in text_norm:
            return label
    return "pregao" if "pregao" in text_norm else ""


def short_hint(value, max_words=12):
    words = norm(value).split()
    if not words:
        return ""
    return " ".join(words[:max_words])


def short_window(text, answer, context_chars):
    text_norm = norm(text)
    answer_norm = norm(answer)
    pos = text_norm.find(answer_norm[:80])
    if pos < 0:
        start = 0
    else:
        ratio = max(1, len(text)) / max(1, len(text_norm))
        start = max(0, int(pos * ratio) - 450)
    return clean(text[start : start + context_chars])


def context_for(record, answer, context_chars):
    return (
        "### Documento DOMPI-2025\n"
        f"Municipio: {record.get('municipio', '')}\n"
        f"Data: {record.get('data', '')}\n"
        f"Tipo: {record.get('tipo_ato_normalizado') or record.get('tipo_ato', '')}\n"
        f"ID publicacao: {record.get('id_publicacao', '')}\n\n"
        "Trecho do documento:\n"
        f"{short_window(record.get('texto', ''), answer, context_chars)}\n\n"
        "### FIM_DO_TRECHO"
    )


def rubric(answer):
    tokens = []
    for token in norm(answer).split():
        if len(token) >= 4 and token not in {"para", "pela", "pelo", "como", "com", "dos", "das"}:
            tokens.append(token)
    out = []
    for token in tokens:
        if token not in out:
            out.append(token)
        if len(out) == 5:
            break
    return out


def valid_item(answer, context):
    answer_norm = norm(answer)
    context_norm = norm(context)
    if len(answer_norm) < 4:
        return False
    if answer_norm not in context_norm:
        tokens = [token for token in answer_norm.split() if len(token) >= 4]
        hits = sum(1 for token in tokens if token in context_norm)
        return hits >= max(1, min(4, len(tokens) // 2))
    return True


def build_item(record, question, answer, tema, answer_type, context_chars):
    answer = one_line(answer, 280)
    context = context_for(record, answer, context_chars)
    if not valid_item(answer, context):
        return None
    return {
        "tarefa": "QA contextual especifica",
        "formato": "generate_until",
        "answer_type": answer_type,
        "tema": tema,
        "dificuldade": "facil_media",
        "pergunta": question,
        "alternativas": [],
        "gabarito": answer,
        "resposta_referencia": answer,
        "answer_aliases": [],
        "rubric_must_include": rubric(answer),
        "metricas_sugeridas": ["exact_match_normalizado", "token_f1", "rubric_recall", "groundedness"],
        "criterio_correcao": "A resposta deve ser curta e preservar o campo especifico pedido no enunciado.",
        "contexto": context,
        "evidencias": [answer],
        "origem": {
            "municipio": record.get("municipio", ""),
            "data": record.get("data", ""),
            "tipo_ato": record.get("tipo_ato_normalizado") or record.get("tipo_ato", ""),
            "id_publicacao": record.get("id_publicacao", ""),
            "nome_arquivo": record.get("nome_arquivo", ""),
        },
        "curadoria_contexto": {
            "metodo": "pergunta_especifica_com_resposta_curta_ancorada_no_contexto",
            "status": "mantido",
            "benchmark_split": "test",
        },
    }


def candidates_for(record, context_chars):
    text = record.get("texto", "")
    tipo = record.get("tipo_ato_normalizado") or record.get("tipo_ato", "")
    municipio = one_line(record.get("municipio", "municipio nao identificado"), 80)
    data = one_line(record.get("data", "data nao identificada"), 40)
    objeto = field_after(text, ["OBJETO"], 260)
    contratado = field_after(text, ["CONTRATADA", "CONTRATADO"], 220)
    valor = field_after(text, ["VALOR"], 140)
    valor_curto = money_value(valor or text)
    mod = modality(text)
    out = []

    if objeto:
        out.append(
            build_item(
                record,
                f"No documento DOMPI de {municipio}, data {data}, qual texto aparece no campo OBJETO?",
                objeto,
                "campo_objeto",
                "span_campo",
                context_chars,
            )
        )
    if contratado and objeto and valid_contracted(contratado):
        out.append(
            build_item(
                record,
                f"No trecho em que o OBJETO menciona '{short_hint(objeto)}', qual parte aparece como CONTRATADA/CONTRATADO?",
                contratado,
                "campo_contratado",
                "span_campo",
                context_chars,
            )
        )
    if valor_curto and objeto:
        out.append(
            build_item(
                record,
                f"No trecho sobre '{short_hint(objeto)}', qual valor monetario aparece no campo VALOR ou no extrato?",
                valor_curto,
                "campo_valor",
                "valor_monetario",
                context_chars,
            )
        )
    if mod and objeto:
        out.append(
            build_item(
                record,
                f"No documento de {municipio}, cujo objeto menciona '{short_hint(objeto)}', qual procedimento licitatorio e citado?",
                mod,
                "modalidade_licitacao",
                "classe_curta",
                context_chars,
            )
        )

    portaria = re.search(
        r"(Art\.?\s*1[^\w]?\s*[-:.]?\s*(?:Nomear|Exonerar|Designar|Conceder|Instituir|Revogar).{25,240}?)(?=\n\s*Art\.?\s*2|\n\s*REGISTRE-SE|\n\s*Publique|\Z)",
        text,
        flags=re.I | re.S,
    )
    if portaria:
        answer = one_line(portaria.group(1), 240)
        if norm(answer).count("art 1") == 1:
            out.append(
                build_item(
                    record,
                    f"Na portaria de {municipio}, data {data}, qual acao aparece no Art. 1?",
                    answer,
                    "portaria_artigo_1",
                    "span_artigo",
                    context_chars,
                )
            )

    demonstrativo = re.search(r"(DEMONSTRATIVO\s+(?:DA|DO|DAS|DOS)\s+.{20,130}?)(?:\n|RGF|RREO)", text, flags=re.I | re.S)
    if demonstrativo and tipo in {"LRF_RGF", "LRF_RREO"}:
        out.append(
            build_item(
                record,
                f"No documento fiscal de {municipio}, qual demonstrativo e identificado no trecho?",
                one_line(demonstrativo.group(1), 180),
                "demonstrativo_fiscal",
                "span_titulo",
                context_chars,
            )
        )

    return [row for row in out if row]


def select_items(records, size, context_chars, seed):
    rng = random.Random(seed)
    rng.shuffle(records)
    limits = {
        "campo_objeto": 7,
        "campo_contratado": 5,
        "campo_valor": 5,
        "modalidade_licitacao": 4,
        "portaria_artigo_1": 3,
        "demonstrativo_fiscal": 1,
    }
    selected = []
    counts = Counter()
    used_docs = set()
    used_questions = set()

    for record in records:
        for candidate in candidates_for(record, context_chars):
            tema = candidate["tema"]
            pub_id = candidate["origem"]["id_publicacao"]
            key = norm(candidate["pergunta"])
            if counts[tema] >= limits.get(tema, size):
                continue
            if pub_id in used_docs or key in used_questions:
                continue
            selected.append(candidate)
            counts[tema] += 1
            used_docs.add(pub_id)
            used_questions.add(key)
            break
        if len(selected) >= size:
            return selected

    for record in records:
        for candidate in candidates_for(record, context_chars):
            pub_id = candidate["origem"]["id_publicacao"]
            key = norm(candidate["pergunta"])
            if pub_id in used_docs or key in used_questions:
                continue
            selected.append(candidate)
            used_docs.add(pub_id)
            used_questions.add(key)
            break
        if len(selected) >= size:
            return selected
    return selected


def main():
    args = parse_args()
    train_ids = load_ids(args.data_dir / "splits" / "train_ids.txt")
    valid_ids = load_ids(args.data_dir / "splits" / "valid_ids.txt")
    test_ids = load_ids(args.data_dir / "splits" / "test_ids.txt")
    old_benchmark_ids = load_ids(args.data_dir / "splits" / "benchmark_ids.txt")
    previous_ids = load_benchmark_publication_ids(PREVIOUS_BENCHMARKS)
    allowed = test_ids - train_ids - valid_ids - old_benchmark_ids - previous_ids
    records = load_records(args.extracoes_jsonl, allowed)
    selected = select_items(records, args.size, args.context_chars, args.seed)
    if len(selected) < args.size:
        raise RuntimeError(f"Gerados {len(selected)} itens, esperado {args.size}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected, start=1):
            row["id"] = f"gold_specific_q{idx:02d}"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_ids = {row["origem"]["id_publicacao"] for row in selected}
    manifest = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "descricao": "Benchmark DOMPI especifico com 25 perguntas de resposta curta e evidencias no contexto.",
        "output": str(args.output.resolve()),
        "fonte": str(args.extracoes_jsonl.resolve()),
        "size": len(selected),
        "context_chars": args.context_chars,
        "temas": dict(Counter(row["tema"] for row in selected)),
        "answer_types": dict(Counter(row["answer_type"] for row in selected)),
        "sem_vazamento": {
            "intersect_train": len(selected_ids & train_ids),
            "intersect_valid": len(selected_ids & valid_ids),
            "intersect_old_benchmark": len(selected_ids & old_benchmark_ids),
            "intersect_previous_contextual_benchmarks": len(selected_ids & previous_ids),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
