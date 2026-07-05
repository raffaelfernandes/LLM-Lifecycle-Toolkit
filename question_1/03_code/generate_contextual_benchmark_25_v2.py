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
DEFAULT_OUTPUT = BASE_DIR / "llm_pretraining" / "data_dompi_2025" / "benchmark_gold_contextual_25_v2.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Gera benchmark contextual aberto de 25 questoes DOMPI.")
    parser.add_argument("--extracoes-jsonl", type=Path, default=DEFAULT_EXTRACOES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--context-chars", type=int, default=2600)
    parser.add_argument("--seed", type=int, default=42)
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


def one_line(text, max_len=420):
    text = clean(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" :-.;")
    return text[:max_len].strip(" :-.;")


def load_ids(path):
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


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


def field_after(text, labels):
    stop = (
        "CONTRATANTE|CONTRATADA|CONTRATADO|DISTRATANTE|DISTRATADA|VALOR|FONTE|"
        "ASSINATURA|VIGENCIA|VIGNCIA|PRAZO|FUNDAMENTO|OBJETO|DATA|RECURSOS"
    )
    for label in labels:
        match = re.search(rf"{label}\s*[:\-]\s*(.+?)(?:\n\s*(?:{stop})\s*[:\-]|\Z)", text, flags=re.I | re.S)
        if match:
            value = one_line(match.group(1), 360)
            if 4 <= len(value) <= 360:
                return value
    return ""


def first_match(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match
    return None


def short_window(text, answer, context_chars):
    text_norm = norm(text)
    answer_norm = norm(answer)
    pos = text_norm.find(answer_norm[:80])
    if pos < 0:
        start = 0
    else:
        ratio = max(1, len(text)) / max(1, len(text_norm))
        start = max(0, int(pos * ratio) - 500)
    return clean(text[start : start + context_chars])


def context_for(record, answer, context_chars):
    trecho = short_window(record.get("texto", ""), answer, context_chars)
    return (
        "### Documento DOMPI-2025\n"
        f"Territorio: {record.get('territorio', '')}\n"
        f"Municipio: {record.get('municipio', '')}\n"
        f"Data: {record.get('data', '')}\n"
        f"Tipo: {record.get('tipo_ato_normalizado') or record.get('tipo_ato', '')}\n"
        f"ID publicacao: {record.get('id_publicacao', '')}\n\n"
        "Trecho do documento:\n"
        f"{trecho}\n\n"
        "### FIM_DO_TRECHO"
    )


def valid_answer(answer, context):
    answer_norm = norm(answer)
    if len(answer) < 20 or len(answer) > 520:
        return False
    if len(answer_norm.split()) < 4:
        return False
    tokens = [token for token in answer_norm.split() if len(token) >= 4]
    context_norm = norm(context)
    hits = sum(1 for token in tokens if token in context_norm)
    return hits >= max(2, min(6, len(tokens) // 2))


def is_fiscal_doc(record, text):
    tipo = record.get("tipo_ato_normalizado") or record.get("tipo_ato", "")
    text_norm = norm(text)
    fiscal_terms = [
        "rgf",
        "rreo",
        "demonstrativo",
        "relatorio de gestao fiscal",
        "execucao orcamentaria",
    ]
    return tipo in {"LRF_RGF", "LRF_RREO"} or any(term in text_norm for term in fiscal_terms)


def fiscal_item(record, text, municipio, context_chars):
    match = first_match(
        text,
        [
            r"(DEMONSTRATIVO\s+(?:DA|DO|DAS|DOS)\s+.{25,180}?)(?:\n|RGF|RREO)",
            r"(RELAT.{0,3}RIO\s+(?:DE\s+GEST.{0,3}O\s+FISCAL|RESUMIDO\s+DA\s+EXECU.{0,3}O\s+OR.{0,3}AMENT.{0,3}RIA).{20,160}?)(?:\n|RGF|RREO)",
        ],
    )
    if not match:
        return None
    answer = f"O documento apresenta {one_line(match.group(1), 240)}."
    return item(record, "Considerando o trecho, qual demonstrativo fiscal aparece no documento?", answer, "relatorio_fiscal", context_chars)


def valid_normative_answer(answer, record, text):
    if len(answer) < 80:
        return False
    if is_fiscal_doc(record, text):
        return False
    tipo = record.get("tipo_ato_normalizado") or record.get("tipo_ato", "")
    if tipo not in {"Decreto", "Lei", "Resolucao", "Portaria"}:
        return False
    answer_norm = norm(answer)
    blocked_terms = [
        "receita corrente",
        "limites de endividamento",
        "contratacao de empresa",
        "contrato administrativo",
        "demonstrativo",
        "relatorio de gestao fiscal",
        "dados do servidor requerente",
        "despesas corren",
        "encargos despesas",
    ]
    if any(term in answer_norm for term in blocked_terms):
        return False
    return answer_norm.count("dispoe sobre") <= 1


def rubric_from(answer):
    tokens = []
    for token in re.findall(r"[A-Za-z0-9./-]{4,}", norm(answer)):
        if token and token not in {"para", "com", "como", "pelo", "pela", "esse", "essa", "contratacao"}:
            tokens.append(token)
    out = []
    for token in tokens:
        if token not in out:
            out.append(token)
        if len(out) == 6:
            break
    return out


def item(record, question, answer, tema, context_chars):
    context = context_for(record, answer, context_chars)
    if not valid_answer(answer, context):
        return None
    return {
        "tarefa": "QA aberta contextual",
        "formato": "generate_until",
        "tema": tema,
        "dificuldade": "media",
        "pergunta": question,
        "alternativas": [],
        "gabarito": answer,
        "resposta_referencia": answer,
        "answer_aliases": [],
        "rubric_must_include": rubric_from(answer),
        "metricas_sugeridas": ["token_f1", "bleu_unigrama", "rubric_recall", "avaliacao_manual_groundedness"],
        "criterio_correcao": "Resposta aberta aceita parafrase, mas deve preservar os elementos essenciais e estar sustentada pelo trecho.",
        "contexto": context,
        "evidencias": [answer],
        "origem": {
            "territorio": record.get("territorio", ""),
            "municipio": record.get("municipio", ""),
            "data": record.get("data", ""),
            "tipo_ato": record.get("tipo_ato_normalizado") or record.get("tipo_ato", ""),
            "id_publicacao": record.get("id_publicacao", ""),
            "nome_arquivo": record.get("nome_arquivo", ""),
        },
        "curadoria_contexto": {
            "metodo": "geracao_programatica_v2_conservadora_com_evidencia_no_contexto",
            "status": "mantido",
            "benchmark_split": "test",
        },
    }


def licitation_modality(text):
    text_norm = norm(text)
    if "pregao eletronico" in text_norm:
        return "pregao eletronico"
    if "pregao presencial" in text_norm:
        return "pregao presencial"
    if "dispensa de licitacao" in text_norm:
        return "dispensa de licitacao"
    if "inexigibilidade de licitacao" in text_norm:
        return "inexigibilidade de licitacao"
    if "chamamento publico" in text_norm:
        return "chamamento publico"
    if "tomada de precos" in text_norm:
        return "tomada de precos"
    if "pregao" in text_norm:
        return "pregao"
    return ""


def make_candidate(record, context_chars):
    text = record.get("texto", "")
    text_norm = norm(text)
    tipo = record.get("tipo_ato_normalizado") or record.get("tipo_ato", "")
    municipio = record.get("municipio", "o municipio")
    data = record.get("data", "")

    if is_fiscal_doc(record, text):
        fiscal = fiscal_item(record, text, municipio, context_chars)
        if fiscal:
            return fiscal

    objeto = field_after(text, ["OBJETO"])
    contratado = field_after(text, ["CONTRATADA", "CONTRATADO"])
    valor = field_after(text, ["VALOR"])

    if "rescisao" in text_norm or "distrato" in text_norm:
        contrato = field_after(text, ["REFERENTE AO"])
        parte = field_after(text, ["DISTRATADA", "CONTRATADA", "CONTRATADO"])
        if objeto and (contrato or parte):
            answer = one_line(
                " ".join(
                    part
                    for part in [
                        "O documento registra rescisao ou distrato de contrato.",
                        f"Referencia: {contrato}." if contrato else "",
                        f"Parte relacionada: {parte}." if parte else "",
                        f"Objeto: {objeto}." if objeto else "",
                    ]
                ),
                520,
            )
            return item(record, "Considerando o trecho, o que o termo de rescisao ou distrato informa?", answer, "rescisao", context_chars)

    modalidade = licitation_modality(text)
    if tipo in {"Licitacao", "Edital"} or modalidade:
        if objeto and modalidade:
            answer = one_line(f"O documento divulga {modalidade}. Objeto: {objeto}.", 520)
            return item(record, "Considerando o trecho, que procedimento licitatorio foi divulgado?", answer, "licitacao", context_chars)

    if objeto and (contratado or valor):
        answer = one_line(
            " ".join(
                part
                for part in [
                    f"O objeto foi {objeto}.",
                    f"Contratado ou contratada: {contratado}." if contratado else "",
                    f"Valor informado: {valor}." if valor else "",
                ]
            ),
            520,
        )
        return item(record, "Considerando o trecho, que contratacao foi descrita no documento?", answer, "contratacao", context_chars)

    if tipo == "Portaria" or "portaria" in record.get("nome_arquivo", "").lower():
        match = first_match(
            text,
            [
                r"(Art\.?\s*1[^\w]?\s*[-:.]?\s*(?:Nomear|Exonerar|Designar|Conceder|Instituir|Revogar).{25,280}?)(?=\n\s*Art\.?\s*2|\n\s*REGISTRE-SE|\n\s*Publique|\Z)",
            ],
        )
        if match:
            answer = one_line(match.group(1), 420)
            if norm(answer).count("art 1") > 1:
                return None
            return item(record, "Considerando o trecho, o que ocorreu na portaria?", answer, "portaria", context_chars)

    match = first_match(
        text,
        [
            r"((?:Disp.{0,3}e|Institui|Estabelece|Autoriza|Declara|Altera|Regulamenta|Cria|Denomina|Concede|Abre|Fixa|Aprova)\b.{35,300}?[.;])",
            r"(Art\.?\s*1[^\w]?\s*[-:.]?\s*(?:Fica|Nomeia|Exonera|Designa|Autoriza|Institui|Aprova|Declara).{35,300}?[.;])",
        ],
    )
    if match and valid_normative_answer(match.group(1), record, text):
        answer = one_line(match.group(1), 420)
        return item(record, "Considerando o trecho, o que o ato normativo estabelece?", answer, "ato_normativo", context_chars)

    return None


def select_items(records, size, context_chars, seed):
    rng = random.Random(seed)
    rng.shuffle(records)
    selected = []
    seen_docs = set()
    seen_answers = set()
    tema_counts = Counter()
    limits = {"contratacao": 9, "licitacao": 6, "portaria": 6, "ato_normativo": 4, "rescisao": 3, "relatorio_fiscal": 2}

    for record in records:
        candidate = make_candidate(record, context_chars)
        if not candidate:
            continue
        pub_id = candidate["origem"]["id_publicacao"]
        answer_key = norm(candidate["resposta_referencia"])[:140]
        tema = candidate["tema"]
        if pub_id in seen_docs or answer_key in seen_answers:
            continue
        if tema_counts[tema] >= limits.get(tema, size):
            continue
        selected.append(candidate)
        seen_docs.add(pub_id)
        seen_answers.add(answer_key)
        tema_counts[tema] += 1
        if len(selected) == size:
            return selected

    for record in records:
        candidate = make_candidate(record, context_chars)
        if not candidate:
            continue
        pub_id = candidate["origem"]["id_publicacao"]
        answer_key = norm(candidate["resposta_referencia"])[:140]
        if pub_id in seen_docs or answer_key in seen_answers:
            continue
        selected.append(candidate)
        seen_docs.add(pub_id)
        seen_answers.add(answer_key)
        if len(selected) == size:
            return selected
    return selected


def main():
    args = parse_args()
    test_ids = load_ids(args.data_dir / "splits" / "test_ids.txt")
    train_ids = load_ids(args.data_dir / "splits" / "train_ids.txt")
    valid_ids = load_ids(args.data_dir / "splits" / "valid_ids.txt")
    old_benchmark_ids = load_ids(args.data_dir / "splits" / "benchmark_ids.txt")
    allowed = test_ids - train_ids - valid_ids - old_benchmark_ids
    records = load_records(args.extracoes_jsonl, allowed)
    selected = select_items(records, args.size, args.context_chars, args.seed)
    if len(selected) < args.size:
        raise RuntimeError(f"Gerados {len(selected)} itens, esperado {args.size}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected, start=1):
            row["id"] = f"gold25_v2_q{idx:02d}"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_ids = {row["origem"]["id_publicacao"] for row in selected}
    manifest = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "descricao": "Benchmark contextual aberto v2 com 25 questoes, criado fora dos splits de treino/validacao/benchmark original.",
        "output": str(args.output.resolve()),
        "fonte": str(args.extracoes_jsonl.resolve()),
        "base_ids": "test_ids menos train_ids, valid_ids e benchmark_ids originais",
        "size": len(selected),
        "context_chars": args.context_chars,
        "temas": dict(Counter(row["tema"] for row in selected)),
        "tipos": dict(Counter(row["origem"]["tipo_ato"] for row in selected)),
        "sem_vazamento": {
            "intersect_train": len(selected_ids & train_ids),
            "intersect_valid": len(selected_ids & valid_ids),
            "intersect_old_benchmark": len(selected_ids & old_benchmark_ids),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
