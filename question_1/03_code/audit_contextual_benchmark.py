import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "llm_pretraining" / "data_dompi_2025" / "benchmark_gold_contextual_225_v1.jsonl"

STOPWORDS = {
    "para",
    "com",
    "como",
    "pelo",
    "pela",
    "esse",
    "essa",
    "este",
    "esta",
    "documento",
    "objeto",
    "contratacao",
    "contratado",
    "contratada",
    "informa",
    "considerando",
    "trecho",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audita benchmark contextual DOMPI por ancoragem no contexto.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--approved-output", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--audit-md", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-rubric-coverage", type=float, default=0.80)
    parser.add_argument("--min-answer-token-coverage", type=float, default=0.55)
    parser.add_argument("--min-context-chars", type=int, default=500)
    parser.add_argument("--max-answer-chars", type=int, default=520)
    return parser.parse_args()


def norm(text):
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def content_tokens(text):
    out = []
    for token in re.findall(r"[a-z0-9]{4,}", norm(text)):
        if token not in STOPWORDS and token not in out:
            out.append(token)
    return out


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def coverage(tokens, context_norm):
    if not tokens:
        return 0, 0, 0.0
    hits = sum(1 for token in tokens if token in context_norm)
    return hits, len(tokens), hits / len(tokens)


def audit_row(row, seen_docs, seen_questions):
    context = row.get("contexto", "")
    answer = row.get("resposta_referencia") or row.get("gabarito", "")
    context_norm = norm(context)
    answer_norm = norm(answer)
    rubric = [norm(token) for token in row.get("rubric_must_include", []) if norm(token)]
    rubric = [token for token in rubric if token and token not in STOPWORDS]
    answer_tokens = content_tokens(answer)

    rubric_hits, rubric_total, rubric_cov = coverage(rubric, context_norm)
    answer_hits, answer_total, answer_cov = coverage(answer_tokens, context_norm)
    exact_answer_in_context = bool(answer_norm and answer_norm in context_norm)

    pub_id = row.get("origem", {}).get("id_publicacao", "")
    question_key = norm(row.get("pergunta", "")) + "|" + norm(answer)[:120]

    reasons = []
    if len(context) < 500:
        reasons.append("contexto_curto")
    if len(answer) < 20:
        reasons.append("resposta_curta")
    if len(answer) > 520:
        reasons.append("resposta_longa")
    if rubric_total and rubric_cov < 0.80:
        reasons.append("rubrica_pouco_ancorada")
    if answer_total and answer_cov < 0.55:
        reasons.append("gabarito_pouco_ancorado")
    if pub_id in seen_docs:
        reasons.append("documento_duplicado")
    if question_key in seen_questions:
        reasons.append("pergunta_gabarito_duplicado")

    approved = not reasons
    return {
        "id": row.get("id", ""),
        "tema": row.get("tema", ""),
        "tipo_ato": row.get("origem", {}).get("tipo_ato", ""),
        "municipio": row.get("origem", {}).get("municipio", ""),
        "id_publicacao": pub_id,
        "pergunta": row.get("pergunta", ""),
        "resposta_preview": re.sub(r"\s+", " ", answer)[:180],
        "context_chars": len(context),
        "answer_chars": len(answer),
        "rubric_hits": rubric_hits,
        "rubric_total": rubric_total,
        "rubric_coverage": round(rubric_cov, 4),
        "answer_token_hits": answer_hits,
        "answer_token_total": answer_total,
        "answer_token_coverage": round(answer_cov, 4),
        "exact_answer_in_context": exact_answer_in_context,
        "status": "aprovado" if approved else "reprovado",
        "motivos": ";".join(reasons),
        "_approved": approved,
        "_pub_id": pub_id,
        "_question_key": question_key,
    }


def main():
    args = parse_args()
    input_path = args.input
    approved_output = args.approved_output or input_path.with_name(input_path.stem + "_auditado.jsonl")
    audit_csv = args.audit_csv or input_path.with_name(input_path.stem + "_auditoria.csv")
    audit_md = args.audit_md or input_path.with_name(input_path.stem + "_auditoria.md")
    manifest_path = args.manifest or input_path.with_name(input_path.stem + "_auditoria.manifest.json")

    rows = load_jsonl(input_path)
    seen_docs = set()
    seen_questions = set()
    audit_rows = []
    approved_rows = []

    for row in rows:
        audit = audit_row(row, seen_docs, seen_questions)
        audit_rows.append(audit)
        if audit["_approved"]:
            clean_row = dict(row)
            clean_row["id"] = f"auditado_q{len(approved_rows) + 1:03d}"
            clean_row.setdefault("curadoria_contexto", {})
            clean_row["curadoria_contexto"].update(
                {
                    "status": "aprovado_por_auditoria_programatica",
                    "rubric_coverage": audit["rubric_coverage"],
                    "answer_token_coverage": audit["answer_token_coverage"],
                    "criterios": "rubrica>=0.80, tokens_do_gabarito>=0.55, sem duplicata e contexto suficiente",
                }
            )
            approved_rows.append(clean_row)
            seen_docs.add(audit["_pub_id"])
            seen_questions.add(audit["_question_key"])

    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in audit_rows]
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0].keys()))
        writer.writeheader()
        writer.writerows(public_rows)

    write_jsonl(approved_output, approved_rows)

    summary = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path.resolve()),
        "approved_output": str(approved_output.resolve()),
        "audit_csv": str(audit_csv.resolve()),
        "total_itens": len(rows),
        "aprovados": len(approved_rows),
        "reprovados": len(rows) - len(approved_rows),
        "criterios": {
            "min_rubric_coverage": args.min_rubric_coverage,
            "min_answer_token_coverage": args.min_answer_token_coverage,
            "min_context_chars": args.min_context_chars,
            "max_answer_chars": args.max_answer_chars,
        },
        "temas_aprovados": dict(Counter(row.get("tema", "") for row in approved_rows)),
        "tipos_aprovados": dict(Counter(row.get("origem", {}).get("tipo_ato", "") for row in approved_rows)),
        "motivos_reprovacao": dict(
            Counter(reason for row in audit_rows for reason in row.get("motivos", "").split(";") if reason)
        ),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Auditoria programatica do benchmark contextual",
        "",
        f"Arquivo auditado: `{input_path}`",
        f"Total de itens: {len(rows)}",
        f"Aprovados: {len(approved_rows)}",
        f"Reprovados: {len(rows) - len(approved_rows)}",
        "",
        "## Criterios",
        "",
        "- rubrica minima ancorada no contexto: 80%",
        "- tokens relevantes do gabarito presentes no contexto: 55%",
        "- contexto minimo: 500 caracteres",
        "- resposta entre 20 e 520 caracteres",
        "- sem documento duplicado",
        "- sem par pergunta/gabarito duplicado",
        "",
        "## Temas aprovados",
        "",
    ]
    for tema, count in summary["temas_aprovados"].items():
        lines.append(f"- {tema}: {count}")
    lines.extend(["", "## Motivos de reprovacao", ""])
    for motivo, count in summary["motivos_reprovacao"].items():
        lines.append(f"- {motivo}: {count}")
    lines.extend(["", "## Exemplos aprovados", ""])
    for row in approved_rows[:10]:
        lines.append(f"- `{row['id']}` ({row.get('tema', '')}): {row.get('pergunta', '')}")
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
