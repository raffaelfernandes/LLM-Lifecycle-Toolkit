import argparse
import hashlib
import json
import random
import re
from pathlib import Path


PROJETO = "entre_rios_vale_sambito"
ANO = 2025
BASE_DIR = Path(__file__).resolve().parents[1]
ARQUIVO_EXTRACOES = BASE_DIR / f"extracoes_{PROJETO}_{ANO}.jsonl"
PASTA_SAIDA = BASE_DIR / "llm_pretraining" / "data"


TIPOS_ATO = [
    ("LRF_RREO", "Relatorio Resumido da Execucao Orcamentaria"),
    ("LRF_RGF", "Relatorio de Gestao Fiscal"),
    ("Licitacao", "Licitacao"),
    ("Dispensa", "Dispensa de licitacao"),
    ("Inexigibilidade", "Inexigibilidade de licitacao"),
    ("Portaria", "Portaria"),
    ("Decreto", "Decreto"),
    ("Lei", "Lei"),
    ("Contrato", "Contrato"),
    ("Edital", "Edital"),
    ("Ata", "Ata"),
    ("Convenio", "Convenio"),
    ("Termo", "Termo"),
    ("Resolucao", "Resolucao"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara corpus e benchmark para pre-treinamento continuado com diarios de prefeituras."
    )
    parser.add_argument("--extracoes-jsonl", type=Path, default=ARQUIVO_EXTRACOES)
    parser.add_argument("--output-dir", type=Path, default=PASTA_SAIDA)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-docs", type=int, default=0, help="0 usa todos os documentos validos.")
    parser.add_argument("--benchmark-size", type=int, default=25)
    parser.add_argument("--context-chars", type=int, default=1600)
    return parser.parse_args()


def ler_jsonl(caminho):
    registros = []
    with caminho.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            if linha.strip():
                registros.append(json.loads(linha))
    return registros


def limpar_texto(texto):
    texto = corrigir_mojibake(texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def contar_sinais_mojibake(texto):
    return texto.count("Ã") + texto.count("Â") + texto.count("â€¢") + texto.count("â€“")


def corrigir_mojibake(texto):
    sinais_antes = contar_sinais_mojibake(texto)
    if sinais_antes == 0:
        return texto
    reparado = texto.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    if contar_sinais_mojibake(reparado) < sinais_antes:
        return reparado
    return texto


def contar_sinais_mojibake(texto):
    sinais = ("\u00c3", "\u00c2", "\u00e2\u0080", "\ufffd")
    return sum(texto.count(sinal) for sinal in sinais)


def tipo_ato(nome_arquivo):
    nome = nome_arquivo.replace("-", "_")
    for padrao, rotulo in TIPOS_ATO:
        if padrao.lower() in nome.lower():
            return rotulo
    return "Publicacao oficial"


def identificador_publicacao(registro):
    return str(registro.get("numero") or registro.get("id_publicacao") or "")


def familia_ato(tipo):
    if tipo in {"Relatorio Resumido da Execucao Orcamentaria", "Relatorio de Gestao Fiscal"}:
        return "Relatorio fiscal"
    if tipo in {"Licitacao", "Dispensa de licitacao", "Inexigibilidade de licitacao"}:
        return "Licitacao e contratacao publica"
    if tipo in {"Lei", "Decreto", "Portaria", "Resolucao"}:
        return "Ato normativo ou administrativo"
    if tipo in {"Contrato", "Convenio", "Termo"}:
        return "Contrato, convenio ou termo"
    if tipo in {"Edital", "Ata"}:
        return "Edital ou ata"
    return "Outra publicacao oficial"


def montar_alternativas(resposta, universo, deslocamento=0):
    alternativas = [resposta]
    for item in universo:
        if item != resposta and item not in alternativas:
            alternativas.append(item)
        if len(alternativas) == 4:
            break
    if alternativas:
        deslocamento = deslocamento % len(alternativas)
        alternativas = alternativas[deslocamento:] + alternativas[:deslocamento]
    letras = ["A", "B", "C", "D"]
    return [{"letra": letra, "texto": texto} for letra, texto in zip(letras, alternativas)]


def item_benchmark(doc, indice, template):
    registro = doc["registro"]
    tipo = tipo_ato(registro.get("nome_arquivo", ""))
    familia = familia_ato(tipo)
    contexto = montar_documento(registro, doc["texto"][: template["context_chars"]])

    if template["formato"] == "multiple_choice":
        if template["campo"] == "tipo":
            resposta = tipo
            universo = [rotulo for _, rotulo in TIPOS_ATO if rotulo != resposta]
        else:
            resposta = familia
            universo = [
                "Licitacao e contratacao publica",
                "Ato normativo ou administrativo",
                "Relatorio fiscal",
                "Contrato, convenio ou termo",
                "Edital ou ata",
                "Outra publicacao oficial",
            ]
        alternativas = montar_alternativas(resposta, universo, deslocamento=indice)
        correta = next(item for item in alternativas if item["texto"] == resposta)
        resposta_referencia = f"{correta['letra']}) {correta['texto']}"
        metrica = ["accuracy", "accuracy_normalizada"]
        criterio = "Acerta se escolher a letra correspondente a alternativa correta."
    else:
        alternativas = []
        correta = {}
        resposta = template["responder"](registro, tipo, familia)
        resposta_referencia = resposta
        metrica = ["exact_match_normalizado", "bleu_unigrama"]
        criterio = "Acerta se a resposta curta contiver o gabarito normalizado; BLEU unigram complementa a comparacao textual."

    return {
        "id": f"q{indice:02d}",
        "tarefa": template["tarefa"],
        "formato": template["formato"],
        "tema": template["tema"],
        "dificuldade": template["dificuldade"],
        "pergunta": template["pergunta"],
        "alternativas": alternativas,
        "gabarito": correta.get("letra", resposta_referencia),
        "resposta_referencia": resposta_referencia,
        "metricas_sugeridas": metrica,
        "criterio_correcao": criterio,
        "contexto": contexto,
        "origem": {
            "territorio": registro.get("territorio", ""),
            "municipio": registro.get("municipio", ""),
            "numero": registro.get("numero", ""),
            "data": registro.get("data", ""),
            "nome_arquivo": registro.get("nome_arquivo", ""),
            "caminho_markdown": registro.get("caminho_markdown", ""),
        },
    }


def chave_documento(registro):
    bruto = "|".join(
        [
            str(registro.get("id_publicacao", "")),
            str(registro.get("territorio", "")),
            str(registro.get("municipio", "")),
            str(registro.get("numero", "")),
            str(registro.get("data", "")),
            str(registro.get("nome_arquivo", "")),
        ]
    )
    return hashlib.sha1(bruto.encode("utf-8")).hexdigest()


def montar_documento(registro, texto):
    rotulo_identificador = "Edicao"
    if registro.get("id_publicacao") and not registro.get("numero"):
        rotulo_identificador = "Identificador"

    return "\n".join(
        [
            "### Diario de prefeitura",
            f"Territorio: {registro.get('territorio', '')}",
            f"Municipio: {registro.get('municipio', '')}",
            f"{rotulo_identificador}: {identificador_publicacao(registro)}",
            f"Data: {registro.get('data', '')}",
            f"Tipo estimado: {tipo_ato(registro.get('nome_arquivo', ''))}",
            f"Arquivo: {registro.get('nome_arquivo', '')}",
            "",
            "Texto extraido:",
            texto,
            "",
            "### FIM_DO_DOCUMENTO",
            "",
        ]
    )


def carregar_documentos(extracoes_jsonl, min_chars, max_docs):
    documentos = []
    for registro in ler_jsonl(extracoes_jsonl):
        if registro.get("erro_extracao"):
            continue
        caminho_markdown_valor = registro.get("caminho_markdown", "")
        caminho_markdown = Path(caminho_markdown_valor) if caminho_markdown_valor else None
        if caminho_markdown and caminho_markdown.is_file():
            texto = caminho_markdown.read_text(encoding="utf-8", errors="ignore")
        else:
            texto = registro.get("texto", "")
        texto = limpar_texto(texto)
        if len(texto) < min_chars:
            continue
        item = {
            "id": chave_documento(registro),
            "registro": registro,
            "texto": texto,
            "documento_treino": montar_documento(registro, texto),
        }
        documentos.append(item)
        if max_docs and len(documentos) >= max_docs:
            break
    return documentos


def dividir_documentos(documentos, seed):
    rng = random.Random(seed)
    documentos = documentos[:]
    rng.shuffle(documentos)
    n = len(documentos)
    n_train = int(n * 0.9)
    n_valid = max(1, int(n * 0.05))
    return {
        "train": documentos[:n_train],
        "valid": documentos[n_train : n_train + n_valid],
        "test": documentos[n_train + n_valid :],
    }


def salvar_corpus(split, output_dir):
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    arquivos = {}
    for nome, documentos in split.items():
        caminho = corpus_dir / f"{nome}.txt"
        with caminho.open("w", encoding="utf-8") as arquivo:
            for doc in documentos:
                arquivo.write(doc["documento_treino"])
        arquivos[nome] = str(caminho)
    return arquivos


def selecionar_documentos_benchmark(documentos, tamanho):
    selecionados = []
    usados_municipios = set()
    for doc in sorted(documentos, key=lambda d: (d["registro"].get("municipio", ""), d["registro"].get("data", ""))):
        municipio = doc["registro"].get("municipio", "")
        if municipio in usados_municipios:
            continue
        selecionados.append(doc)
        usados_municipios.add(municipio)
        if len(selecionados) == tamanho:
            return selecionados

    for doc in documentos:
        if doc not in selecionados:
            selecionados.append(doc)
        if len(selecionados) == tamanho:
            return selecionados
    return selecionados


def montar_benchmark(documentos, tamanho, context_chars):
    templates = [
        {
            "tarefa": "QA factual institucional",
            "formato": "generate_until",
            "tema": "municipio",
            "dificuldade": "facil",
            "pergunta": "Qual municipio publicou o documento apresentado no contexto?",
            "responder": lambda r, _tipo, _familia: r.get("municipio", ""),
            "context_chars": context_chars,
        },
        {
            "tarefa": "QA factual institucional",
            "formato": "generate_until",
            "tema": "territorio",
            "dificuldade": "facil",
            "pergunta": "A qual territorio pertence o municipio do documento?",
            "responder": lambda r, _tipo, _familia: r.get("territorio", ""),
            "context_chars": context_chars,
        },
        {
            "tarefa": "QA factual institucional",
            "formato": "generate_until",
            "tema": "data",
            "dificuldade": "facil",
            "pergunta": "Qual e a data de publicacao indicada para este documento?",
            "responder": lambda r, _tipo, _familia: r.get("data", ""),
            "context_chars": context_chars,
        },
        {
            "tarefa": "QA factual institucional",
            "formato": "generate_until",
            "tema": "edicao",
            "dificuldade": "facil",
            "pergunta": "Qual e o numero da edicao ou identificador da publicacao indicado no documento?",
            "responder": lambda r, _tipo, _familia: identificador_publicacao(r),
            "context_chars": context_chars,
        },
        {
            "tarefa": "Classificacao de publicacao",
            "formato": "multiple_choice",
            "tema": "tipo de ato",
            "campo": "tipo",
            "dificuldade": "media",
            "pergunta": "Qual alternativa melhor identifica o tipo de ato administrativo deste documento?",
            "context_chars": context_chars,
        },
        {
            "tarefa": "Roteamento por categoria administrativa",
            "formato": "multiple_choice",
            "tema": "categoria administrativa",
            "campo": "familia",
            "dificuldade": "media",
            "pergunta": "Para fins de triagem administrativa, em qual categoria este documento se enquadra?",
            "context_chars": context_chars,
        },
    ]
    benchmark = []
    docs = selecionar_documentos_benchmark(documentos, tamanho)
    for i, doc in enumerate(docs, start=1):
        benchmark.append(item_benchmark(doc, i, templates[(i - 1) % len(templates)]))
    return benchmark


def salvar_benchmark(benchmark, output_dir):
    caminho = output_dir / "municipal_gazettes_benchmark.jsonl"
    with caminho.open("w", encoding="utf-8") as arquivo:
        for item in benchmark:
            arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
    return caminho


def salvar_manifest(args, documentos, split, arquivos_corpus, benchmark_path):
    manifest = {
        "projeto": "diariosPefeituras",
        "fonte": str(args.extracoes_jsonl),
        "geracao": {
            "min_chars": args.min_chars,
            "seed": args.seed,
            "max_docs": args.max_docs,
            "benchmark_size": args.benchmark_size,
            "context_chars": args.context_chars,
        },
        "documentos_validos": len(documentos),
        "splits": {nome: len(valor) for nome, valor in split.items()},
        "arquivos_corpus": arquivos_corpus,
        "benchmark": str(benchmark_path),
    }
    caminho = args.output_dir / "manifest.json"
    caminho.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    documentos = carregar_documentos(args.extracoes_jsonl, args.min_chars, args.max_docs)
    if len(documentos) < args.benchmark_size:
        raise RuntimeError(
            f"Foram encontrados apenas {len(documentos)} documentos validos; "
            f"o benchmark pede {args.benchmark_size}."
        )

    split = dividir_documentos(documentos, args.seed)
    arquivos_corpus = salvar_corpus(split, args.output_dir)
    benchmark = montar_benchmark(split["test"] or split["valid"], args.benchmark_size, args.context_chars)
    benchmark_path = salvar_benchmark(benchmark, args.output_dir)
    manifest_path = salvar_manifest(args, documentos, split, arquivos_corpus, benchmark_path)

    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
