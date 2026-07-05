import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ID = "gutoportelaa/DOMPI-2025"
SNAPSHOT_DIR = BASE_DIR / "data" / "DOMPI-2025"
EXTRACOES_JSONL = BASE_DIR / "extracoes_dompi_2025.jsonl"
DATA_DIR = BASE_DIR / "llm_pretraining" / "data_dompi_2025"
RELATORIO = BASE_DIR / "dompi_2025_report.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baixa o DOMPI-2025 e prepara corpus/benchmark para a primeira questao de pre-treino."
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--extracoes-jsonl", type=Path, default=EXTRACOES_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extracoes", action="store_true")
    parser.add_argument("--max-docs", type=int, default=0, help="0 usa todos os documentos validos.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--min-text-chars", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--benchmark-size", type=int, default=100)
    parser.add_argument("--context-chars", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def remover_acentos(texto):
    return unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode("ascii")


def slug(texto, max_len=80):
    texto = remover_acentos(texto)
    texto = re.sub(r"[^A-Za-z0-9_.-]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_.-")
    return (texto or "sem_valor")[:max_len]


def texto_curto(valor):
    if valor is None:
        return ""
    texto = corrigir_mojibake(str(valor).strip())
    texto = re.sub(r"\s+", " ", texto)
    return texto


def contar_sinais_mojibake(texto):
    sinais = ("\u00c3", "\u00c2", "\u00e2\u0080", "\ufffd")
    return sum(texto.count(sinal) for sinal in sinais)


def corrigir_mojibake(texto):
    sinais_antes = contar_sinais_mojibake(texto)
    if sinais_antes == 0:
        return texto
    reparado = texto.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    if contar_sinais_mojibake(reparado) < sinais_antes:
        return reparado
    return texto


def limpar_texto(texto):
    texto = corrigir_mojibake(str(texto or ""))
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def territorio_legivel(valor):
    texto = texto_curto(valor).replace("_", " ")
    return " ".join(parte.capitalize() for parte in texto.split())


def normalizar_tipo_ato(valor):
    normalizado = remover_acentos(valor).lower()
    normalizado = re.sub(r"[^a-z0-9]+", " ", normalizado)

    if "rreo" in normalizado:
        return "LRF_RREO"
    if "rgf" in normalizado or normalizado.strip() == "lrf":
        return "LRF_RGF"
    if "dispensa" in normalizado:
        return "Dispensa"
    if "inexig" in normalizado:
        return "Inexigibilidade"
    if "licit" in normalizado or "pregao" in normalizado or "concorrencia" in normalizado:
        return "Licitacao"
    if "portaria" in normalizado:
        return "Portaria"
    if "decreto" in normalizado:
        return "Decreto"
    if "lei" in normalizado:
        return "Lei"
    if "contrato" in normalizado:
        return "Contrato"
    if "edital" in normalizado:
        return "Edital"
    if "ata" in normalizado:
        return "Ata"
    if "convenio" in normalizado:
        return "Convenio"
    if "termo" in normalizado:
        return "Termo"
    if "resolucao" in normalizado:
        return "Resolucao"
    return "Publicacao_oficial"


def baixar_dataset(args):
    parquet_dir = args.snapshot_dir / "data" / "raw"
    if parquet_dir.exists() and list(parquet_dir.glob("*.parquet")) and not args.force_download:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Instale huggingface_hub: python -m pip install huggingface_hub") from exc

    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.snapshot_dir),
        allow_patterns=["README.md", ".gitattributes", "data/raw/*.parquet"],
    )


def arquivos_parquet(snapshot_dir):
    arquivos = sorted((snapshot_dir / "data" / "raw").glob("*.parquet"))
    if arquivos:
        return arquivos
    return sorted(snapshot_dir.rglob("*.parquet"))


def valor_int(valor, padrao=0):
    try:
        if valor is None:
            return padrao
        return int(valor)
    except (TypeError, ValueError):
        return padrao


def nome_arquivo_dompi(row):
    tipo = normalizar_tipo_ato(row.get("tipo_ato"))
    territorio = slug(row.get("territorio"), 45)
    municipio = slug(row.get("municipio"), 60)
    data = slug(row.get("data_publicacao"), 20)
    identificador = slug(row.get("id_publicacao"), 24)
    return f"DOMPI_{territorio}_{municipio}_{tipo}_{data}_{identificador}.txt"


def registro_dompi(row, repo_id, parquet_name, texto):
    tipo_original = texto_curto(row.get("tipo_ato"))
    return {
        "id_publicacao": texto_curto(row.get("id_publicacao")),
        "territorio": territorio_legivel(row.get("territorio")),
        "territorio_original": texto_curto(row.get("territorio")),
        "municipio": texto_curto(row.get("municipio")),
        "tipo_ato": tipo_original,
        "tipo_ato_normalizado": normalizar_tipo_ato(tipo_original),
        "data": texto_curto(row.get("data_publicacao")),
        "ano": valor_int(row.get("ano"), 2025),
        "numero": "",
        "extrator": texto_curto(row.get("extrator")),
        "paginas": valor_int(row.get("paginas")),
        "n_chars": valor_int(row.get("n_chars"), len(texto)),
        "caracteres_texto": len(texto),
        "nome_arquivo": nome_arquivo_dompi(row),
        "caminho_pdf": "",
        "caminho_markdown": "",
        "texto": texto,
        "erro_extracao": "",
        "fonte_dataset": repo_id,
        "arquivo_parquet": parquet_name,
    }


def converter_parquets(args):
    if args.extracoes_jsonl.exists() and not args.force_extracoes:
        return json.loads(RELATORIO.read_text(encoding="utf-8")) if RELATORIO.exists() else {}

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Instale pyarrow: python -m pip install pyarrow") from exc

    parquets = arquivos_parquet(args.snapshot_dir)
    if not parquets:
        raise FileNotFoundError(f"Nenhum parquet encontrado em {args.snapshot_dir}")

    args.extracoes_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_lidos = 0
    total_validos = 0
    total_descartados = 0
    total_chars = 0
    por_territorio = Counter()
    por_tipo = Counter()

    colunas = [
        "id_publicacao",
        "territorio",
        "municipio",
        "tipo_ato",
        "data_publicacao",
        "ano",
        "extrator",
        "texto",
        "n_chars",
        "paginas",
    ]

    with args.extracoes_jsonl.open("w", encoding="utf-8") as saida:
        for parquet_path in parquets:
            parquet = pq.ParquetFile(parquet_path)
            colunas_existentes = [col for col in colunas if col in parquet.schema.names]
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=colunas_existentes):
                for row in batch.to_pylist():
                    total_lidos += 1
                    texto = limpar_texto(row.get("texto", ""))
                    if len(texto) < args.min_text_chars:
                        total_descartados += 1
                        continue

                    registro = registro_dompi(row, args.repo_id, parquet_path.name, texto)
                    saida.write(json.dumps(registro, ensure_ascii=False) + "\n")

                    total_validos += 1
                    total_chars += len(texto)
                    por_territorio[registro["territorio"]] += 1
                    por_tipo[registro["tipo_ato_normalizado"]] += 1

                    if args.max_docs and total_validos >= args.max_docs:
                        break
                if args.max_docs and total_validos >= args.max_docs:
                    break
            if args.max_docs and total_validos >= args.max_docs:
                break

    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "snapshot_dir": str(args.snapshot_dir.resolve()),
        "extracoes_jsonl": str(args.extracoes_jsonl.resolve()),
        "arquivos_parquet": [str(path.resolve()) for path in parquets],
        "max_docs": args.max_docs,
        "min_text_chars": args.min_text_chars,
        "linhas_lidas": total_lidos,
        "documentos_validos": total_validos,
        "documentos_descartados": total_descartados,
        "total_caracteres": total_chars,
        "por_territorio": dict(sorted(por_territorio.items())),
        "por_tipo_ato_normalizado": dict(sorted(por_tipo.items())),
    }
    RELATORIO.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return relatorio


def preparar_corpus_benchmark(args):
    comando = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_benchmark_dataset.py"),
        "--extracoes-jsonl",
        str(args.extracoes_jsonl),
        "--output-dir",
        str(args.output_dir),
        "--min-chars",
        str(args.min_chars),
        "--seed",
        str(args.seed),
        "--benchmark-size",
        str(args.benchmark_size),
        "--context-chars",
        str(args.context_chars),
    ]
    if args.max_docs:
        comando.extend(["--max-docs", str(args.max_docs)])
    subprocess.run(comando, check=True, cwd=str(BASE_DIR))


def preparar_benchmark_gold(args):
    gold_script = SCRIPT_DIR / "generate_gold_benchmark_dompi_2025.py"
    if not gold_script.exists():
        print(f"Skipping optional Gold benchmark generator; file not included: {gold_script}")
        return

    comando = [
        sys.executable,
        str(gold_script),
        "--extracoes-jsonl",
        str(args.extracoes_jsonl),
        "--output-dir",
        str(args.output_dir),
        "--min-chars",
        str(args.min_chars),
        "--seed",
        str(args.seed),
        "--benchmark-size",
        str(args.benchmark_size),
        "--context-chars",
        str(args.context_chars),
    ]
    if args.max_docs:
        comando.extend(["--max-docs", str(args.max_docs)])
    subprocess.run(comando, check=True, cwd=str(BASE_DIR))


def main():
    args = parse_args()
    baixar_dataset(args)
    relatorio = converter_parquets(args)
    preparar_corpus_benchmark(args)
    preparar_benchmark_gold(args)

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "dompi": relatorio,
                "pretreino": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
