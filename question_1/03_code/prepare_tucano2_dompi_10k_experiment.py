import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
EXTRACOES_JSONL = BASE_DIR / "extracoes_dompi_2025.jsonl"
OUTPUT_DIR = BASE_DIR / "llm_pretraining" / "data_dompi_2025_tucano2_10k"


STOP_LABELS = (
    "CONTRATANTE",
    "CONTRATADA",
    "CONTRATADO",
    "VALOR",
    "VALOR GLOBAL",
    "VIGENCIA",
    "PRAZO",
    "FUNDAMENTO",
    "FONTE",
    "DOTA",
    "DATA",
    "ASSINATURA",
    "RECURSOS",
    "MODALIDADE",
)


MANUAL_GOLD_SPECS = [
    {
        "id_publicacao": "f1a48737cf5d5f5cb99f3d12f87ce14f",
        "tema": "portaria",
        "pergunta": "O que ocorreu nas portarias da Camara Municipal de Alto Longa publicadas nesse trecho?",
        "resposta": "A Portaria n. 038/2025 exonerou Antonio Quirino da Silva Filho do cargo comissionado CC-II de Diretor Administrativo, e a Portaria n. 039/2025 nomeou Jose Adelano Moura de Sousa para o mesmo cargo.",
        "anchor": "Antonio Quirino da Silva Filho",
        "rubric": ["Portaria n. 038/2025", "Antonio Quirino", "exonerou", "Diretor Administrativo", "Portaria n. 039/2025", "Jose Adelano"],
    },
    {
        "id_publicacao": "a390b61eeb96ad69e0f05b033980e3c9",
        "tema": "contratacao",
        "pergunta": "O que o termo de rescisao de Alto Longa informa sobre o contrato ambiental?",
        "resposta": "O Municipio de Alto Longa rescindiu unilateralmente o Contrato Administrativo n. 014/2025/CPC/PMALPI, ligado a Inexigibilidade n. 007/2025, firmado com Welder de Sousa Melo Sociedade Individual de Advocacia para consultoria e assessoria juridica ambiental.",
        "anchor": "WELDER DE SOUSA MELO",
        "rubric": ["rescindiu unilateralmente", "Contrato Administrativo n. 014/2025", "Inexigibilidade n. 007/2025", "Welder de Sousa Melo", "consultoria", "ambiental"],
    },
    {
        "id_publicacao": "e3f24547051c47794d99ec5b0971e83d",
        "tema": "contratacao",
        "pergunta": "Qual foi o fato principal registrado no extrato de rescisao de Alto Longa sobre aulas de ingles?",
        "resposta": "O documento registrou a rescisao amigavel do Contrato Administrativo n. 060/2025, da Inexigibilidade n. 012/2025, com Juliene Vitorio Ribeiro (LV Ingles Pra Todos), referente a servicos de ensino de lingua inglesa, coordenacao pedagogica e material didatico.",
        "anchor": "JULIENE VITORIO RIBEIRO",
        "rubric": ["rescisao amigavel", "Contrato Administrativo n. 060/2025", "Inexigibilidade n. 012/2025", "Juliene Vitorio Ribeiro", "ensino de lingua inglesa"],
    },
    {
        "id_publicacao": "88df2405b275e3dc1a009a4a40a2c8e6",
        "tema": "contratacao",
        "pergunta": "Que contratacao foi descrita para a Prefeitura de Alegrete do Piaui e qual valor mensal foi informado?",
        "resposta": "A Prefeitura de Alegrete do Piaui contratou Jose Leonel Lopes de Carvalho (LL Assessoria e Consultoria) para assessoria no ambito da educacao municipal e locacao de software de gestao escolar, pelo valor mensal de R$ 5.075,00.",
        "anchor": "JOSE LEONEL LOPES DE CARVALHO",
        "rubric": ["Alegrete", "Jose Leonel Lopes de Carvalho", "assessoria", "educacao", "software de gestao escolar", "R$ 5.075,00"],
    },
    {
        "id_publicacao": "e7889903210db5c0b31a3689514352fe",
        "tema": "objeto_contratacao",
        "pergunta": "Explique qual era o objeto da contratacao mencionada no documento de Anisio de Abreu.",
        "resposta": "O objeto era a contratacao de pessoa juridica para aquisicao de agua mineral e botijao de gas para atender as demandas das secretarias do Municipio de Anisio de Abreu.",
        "anchor": "aquisio de agua mineral",
        "rubric": ["pessoa juridica", "agua mineral", "botijao de gas", "secretarias", "Anisio de Abreu"],
    },
    {
        "id_publicacao": "58c67af000789e4866fb03f835b2a7a6",
        "tema": "objeto_contratacao",
        "pergunta": "Qual servico tecnico era buscado pela contratacao descrita para Arraial?",
        "resposta": "A contratacao buscava uma empresa para elaborar projeto executivo de pavimentacao de vias publicas no perimetro urbano do Municipio de Arraial.",
        "anchor": "pavimentao de vias",
        "rubric": ["empresa", "projeto executivo", "pavimentacao", "vias publicas", "Arraial"],
    },
    {
        "id_publicacao": "83f50b3ea44e76c4d937c4708b18e220",
        "tema": "objeto_contratacao",
        "pergunta": "O que seria prestado no contrato de comunicacao institucional citado no documento?",
        "resposta": "O contrato previa servicos de comunicacao institucional, incluindo criacao de conteudo, producao de pecas publicitarias e veiculacao digital de conteudo informativo sobre as acoes da AMVI e dos municipios consorciados.",
        "anchor": "comunicao institucional",
        "rubric": ["comunicacao institucional", "criacao de conteudo", "pecas publicitarias", "veiculacao digital", "AMVI", "municipios consorciados"],
    },
    {
        "id_publicacao": "15ab5cba299a99373a65a6c679653043",
        "tema": "ato_administrativo",
        "pergunta": "O que a Portaria n. 03/2025 decidiu sobre o dia 13 de maio de 2025?",
        "resposta": "A Portaria n. 03/2025 manteve e confirmou o feriado municipal de 13 de maio de 2025, em homenagem a padroeira do municipio, Nossa Senhora de Fatima.",
        "anchor": "feriado municipal do dia 13 de maio de 2025",
        "rubric": ["Portaria n. 03/2025", "feriado municipal", "13 de maio de 2025", "Nossa Senhora de Fatima"],
    },
    {
        "id_publicacao": "fc0fefbf80a784382ee36564782e0097",
        "tema": "ato_normativo",
        "pergunta": "Em termos praticos, o que a lei mencionada faz em relacao ao exercicio financeiro de 2025?",
        "resposta": "A lei fixa a despesa do municipio para o exercicio financeiro de 2025.",
        "anchor": "Fixa a Despesa",
        "rubric": ["fixa", "despesa", "municipio", "exercicio financeiro de 2025"],
    },
    {
        "id_publicacao": "f3561e03f7e16795a420c0120b6b8569",
        "tema": "objeto_contratacao",
        "pergunta": "Qual apresentacao artistica foi contratada para o aniversario da cidade?",
        "resposta": "O contrato tinha por objeto a contratacao da artista Taty Girl para show durante o evento de aniversario de 31 anos da cidade, previsto para 12 de dezembro de 2025.",
        "anchor": "taty girl",
        "rubric": ["Taty Girl", "show", "aniversario", "31 anos", "12 de dezembro de 2025"],
    },
    {
        "id_publicacao": "ff20cadddf1c49bbada3c6cc4e917a16",
        "tema": "ato_normativo",
        "pergunta": "Que regime ou estatuto foi instituido pelo ato normativo citado no trecho?",
        "resposta": "O ato instituiu o Regime Juridico Unico e o Estatuto dos Servidores Publicos do Municipio de Cristino Castro, no Piaui.",
        "anchor": "Regime Juridico Unico",
        "rubric": ["Regime Juridico Unico", "Estatuto dos Servidores Publicos", "Cristino Castro"],
    },
    {
        "id_publicacao": "7bd116c74b45f441d62d025893b4508d",
        "tema": "ato_normativo",
        "pergunta": "O que foi convocado pelo decreto e qual era o tema informado?",
        "resposta": "O decreto convocou a 9a Conferencia Municipal de Saude de Assuncao do Piaui, marcada para 28 de julho de 2025, com o tema 'Cuidar bem das pessoas: compromisso com o Sistema Unico de Saude (SUS)'.",
        "anchor": "9",
        "rubric": ["9a Conferencia Municipal de Saude", "28 de julho de 2025", "Cuidar bem das pessoas", "SUS"],
    },
    {
        "id_publicacao": "96d5a9233ce87bcd4c0fb86b39ed8400",
        "tema": "objeto_contratacao",
        "pergunta": "Qual era o objeto da prestacao de servicos tecnicos profissionais descrita?",
        "resposta": "O objeto era a prestacao de servicos tecnicos profissionais de assessoria e consultoria juridica e administrativa na area do direito publico, incluindo administrativo, constitucional e tributario.",
        "anchor": "assessoria e consultoria",
        "rubric": ["assessoria", "consultoria juridica", "administrativa", "direito publico", "constitucional", "tributario"],
    },
    {
        "id_publicacao": "896185dddb1353f9d188adba3d7bd79b",
        "tema": "ato_normativo",
        "pergunta": "O que a lei institui na area ambiental?",
        "resposta": "A lei institui a Politica Municipal de Meio Ambiente e cria o Conselho Municipal de Meio Ambiente.",
        "anchor": "Politica Municipal de Meic Ambiente",
        "rubric": ["Politica Municipal de Meio Ambiente", "Conselho Municipal de Meio Ambiente"],
    },
    {
        "id_publicacao": "02d155de3554015188c94760436e7e63",
        "tema": "objeto_contratacao",
        "pergunta": "Que tipo de projeto tecnico de engenharia seria elaborado segundo o documento?",
        "resposta": "O documento tratava da contratacao de empresa especializada para elaborar projeto tecnico de engenharia para execucao de obras e servicos de engenharia em estradas vicinais no Municipio de Assuncao do Piaui.",
        "anchor": "Estradas Vicinais",
        "rubric": ["projeto tecnico de engenharia", "obras", "servicos de engenharia", "estradas vicinais", "Assuncao do Piaui"],
    },
    {
        "id_publicacao": "5f1c7ff1828513fd7473bbb766f1be92",
        "tema": "ato_normativo",
        "pergunta": "O que o Decreto n. 22/2025 estabeleceu sobre luto oficial?",
        "resposta": "O Decreto n. 22/2025 decretou luto oficial nos dias 11, 12 e 13 de setembro de 2025 em homenagem postuma a Sra. Maria Alves, sem prejuizo ao funcionamento dos orgaos publicos.",
        "anchor": "Luto Oficial",
        "rubric": ["Decreto n. 22/2025", "luto oficial", "11, 12 e 13 de setembro de 2025", "Maria Alves", "orgaos publicos"],
    },
    {
        "id_publicacao": "1c4fff11a30c812bf976d911b2733d6d",
        "tema": "objeto_contratacao",
        "pergunta": "Qual obra era objeto da contratacao de empresa construtora?",
        "resposta": "A contratacao tinha por objeto a execucao dos servicos de reforma da fachada da UBS sede no Municipio de Assuncao do Piaui.",
        "anchor": "reforma da fachada da UBS",
        "rubric": ["empresa construtora", "reforma", "fachada", "UBS sede", "Assuncao do Piaui"],
    },
    {
        "id_publicacao": "df69a41c762ba8cefc3cf41f663fe962",
        "tema": "objeto_contratacao",
        "pergunta": "Que servico seria prestado ao Municipio de Assuncao do Piaui?",
        "resposta": "O documento descreve a contratacao de empresa para prestacao de servicos de transporte escolar do Municipio de Assuncao do Piaui.",
        "anchor": "transporte escolar",
        "rubric": ["empresa", "prestacao de servicos", "transporte escolar", "Assuncao do Piaui"],
    },
    {
        "id_publicacao": "b9e772545b3fa87c0b59378506a1cb76",
        "tema": "objeto_contratacao",
        "pergunta": "Que servicos especializados a Camara Municipal de Boqueirao do Piaui pretendia contratar?",
        "resposta": "A Camara pretendia contratar empresa para servicos especializados em contabilidade publica, incluindo elaboracao de prestacao de contas mensal e demais obrigacoes acessorias junto aos orgaos fiscalizadores.",
        "anchor": "CONTABILIDADE PUBLICA",
        "rubric": ["contabilidade publica", "prestacao de contas mensal", "obrigacoes acessorias", "orgaos fiscalizadores", "Camara Municipal de Boqueirao"],
    },
    {
        "id_publicacao": "0dcb5910dda4228d058a19abd1cc3f9f",
        "tema": "objeto_contratacao",
        "pergunta": "Qual era a finalidade da implantacao de sistemas automatizados citada no documento?",
        "resposta": "A finalidade era implantar sistemas automatizados para suprir as necessidades da Camara Municipal de Juazeiro do Piaui.",
        "anchor": "sistemas automatizados",
        "rubric": ["sistemas automatizados", "suprir necessidades", "Camara Municipal", "Juazeiro do Piaui"],
    },
    {
        "id_publicacao": "3bdc514297775c4622008dd941601de2",
        "tema": "objeto_contratacao",
        "pergunta": "Quais atividades de engenharia eram objeto da prestacao de servicos?",
        "resposta": "O objeto envolvia elaboracao de projetos arquitetonicos, hidraulicos, eletricos e estruturais de edificacoes, alem de monitoramento do sistema FNDE-SIMEC e outras atribuicoes como engenheiro fiscal e projetista do municipio.",
        "anchor": "PROJETOS ARQUITET",
        "rubric": ["projetos arquitetonicos", "hidraulicos", "eletricos", "estruturais", "FNDE", "SIMEC", "engenheiro fiscal"],
    },
    {
        "id_publicacao": "4fbc010911027a38baac0651bf3b0aa0",
        "tema": "ato_normativo",
        "pergunta": "Qual conselho municipal foi instituido pelo decreto citado?",
        "resposta": "O decreto instituiu o Conselho Municipal dos Direitos Humanos de Sao Miguel do Tapuio, identificado como CMDH-SMT.",
        "anchor": "Conselho Municipal dos Direitos Humanos",
        "rubric": ["Conselho Municipal dos Direitos Humanos", "Sao Miguel do Tapuio", "CMDH-SMT"],
    },
    {
        "id_publicacao": "dcc1b937a3f7077e5387fd6020aa2b86",
        "tema": "relatorio_fiscal",
        "pergunta": "Qual demonstrativo fiscal foi apresentado no documento?",
        "resposta": "O documento apresentou o Demonstrativo da Disponibilidade de Caixa e dos Restos a Pagar dos orcamentos fiscais e da seguridade social, referente a janeiro de 2025 a agosto de 2025.",
        "anchor": "DISPONIBILIDADE DE CAIXA",
        "rubric": ["Demonstrativo da Disponibilidade de Caixa", "Restos a Pagar", "orcamentos fiscais", "seguridade social", "janeiro de 2025", "agosto de 2025"],
    },
    {
        "id_publicacao": "95c00a2610fb3bf3dc9094fa3b2abd11",
        "tema": "relatorio_fiscal",
        "pergunta": "Qual demonstrativo de pessoal foi identificado no trecho?",
        "resposta": "O trecho identifica o Demonstrativo da Despesa com Pessoal do Poder Executivo, dos orcamentos fiscais e da seguridade social, referente ao periodo de maio de 2024 a abril de 2025.",
        "anchor": "DESPESA COM PESSOAL",
        "rubric": ["Demonstrativo da Despesa com Pessoal", "Poder Executivo", "orcamentos fiscais", "seguridade social", "maio de 2024", "abril de 2025"],
    },
    {
        "id_publicacao": "c059d21bd38ec66ec9d3c3208fe99705",
        "tema": "objeto_contratacao",
        "pergunta": "Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?",
        "resposta": "O documento registra a contratacao da banda ou atracao artistica Galicia Cruz para realizar show em 26 de janeiro de 2026 durante as comemoracoes do aniversario da cidade de Boqueirao do Piaui.",
        "anchor": "GALICIA CRUZ",
        "rubric": ["Galicia Cruz", "show", "26 de janeiro de 2026", "aniversario", "Boqueirao do Piaui"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara experimento Tucano2 DOMPI-2025 com amostra 10k e benchmark Gold aberto sem vazamento."
    )
    parser.add_argument("--extracoes-jsonl", type=Path, default=EXTRACOES_JSONL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--train-docs", type=int, default=10000)
    parser.add_argument("--valid-docs", type=int, default=1000)
    parser.add_argument("--test-docs", type=int, default=1000)
    parser.add_argument("--benchmark-size", type=int, default=25)
    parser.add_argument("--context-chars", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def remover_acentos(texto):
    return unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode("ascii")


def normalizar(texto):
    texto = remover_acentos(texto).lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def contar_sinais_mojibake(texto):
    sinais = ("\u00c3", "\u00c2", "\u00e2\u0080", "\ufffd")
    return sum(str(texto or "").count(sinal) for sinal in sinais)


def corrigir_mojibake(texto):
    texto = str(texto or "")
    sinais_antes = contar_sinais_mojibake(texto)
    if sinais_antes == 0:
        return texto
    reparado = texto.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    if contar_sinais_mojibake(reparado) < sinais_antes:
        return reparado
    return texto


def limpar_texto(texto):
    texto = corrigir_mojibake(texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def compactar(texto):
    return re.sub(r"\s+", " ", limpar_texto(texto)).strip()


def chave_doc(registro):
    valor = str(registro.get("id_publicacao") or "").strip()
    if valor:
        return valor
    bruto = "|".join(
        str(registro.get(campo, ""))
        for campo in ("territorio", "municipio", "tipo_ato_normalizado", "data", "nome_arquivo")
    )
    return hashlib.sha1(bruto.encode("utf-8")).hexdigest()


def carregar_unicos(caminho, min_chars):
    por_id = {}
    linhas_validas = 0
    with caminho.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            if not linha.strip():
                continue
            registro = json.loads(linha)
            if registro.get("erro_extracao"):
                continue
            texto = limpar_texto(registro.get("texto", ""))
            if len(texto) < min_chars:
                continue
            linhas_validas += 1
            registro = {**registro, "texto": texto}
            doc_id = chave_doc(registro)
            atual = por_id.get(doc_id)
            if atual is None or len(texto) > len(atual["texto"]):
                por_id[doc_id] = {
                    "id": doc_id,
                    "registro": registro,
                    "texto": texto,
                    "n_chars": len(texto),
                }
    return list(por_id.values()), linhas_validas


def numero_ato(texto, tipo):
    padroes = [
        rf"\b{re.escape(tipo)}\s*(?:N|No|N[oº°.]?)\s*([0-9][0-9./-]*)",
        r"\b(?:CONTRATO|PROCESSO|DISPENSA|INEXIGIBILIDADE|PREGAO|EDITAL)\s*(?:ADMINISTRATIVO|ELETRONICO)?\s*(?:N|No|N[oº°.]?)\s*([0-9][0-9./-]*)",
    ]
    for padrao in padroes:
        match = re.search(padrao, texto, flags=re.I)
        if match:
            return match.group(1).strip(" .,/;-")
    return ""


def limpar_resposta(texto):
    texto = compactar(texto)
    texto = re.sub(r"&[a-zA-Z]+;", " ", texto)
    texto = re.sub(r"\bId\s*:[A-Za-z0-9]+", " ", texto, flags=re.I)
    texto = texto.replace("##", " ")
    texto = re.sub(r"^(objeto|resolve|art\.?\s*1[oº°]?|ementa)\s*[:.\-]*\s*", "", texto, flags=re.I)
    for label in STOP_LABELS:
        texto = re.sub(rf"\s+\b{label}\b\s*:?.*$", "", texto, flags=re.I)
    texto = re.sub(r"\s+\b(?:fim|inicio|abertura|recebimento)\s+de\s+.{0,80}$", "", texto, flags=re.I)
    texto = texto.strip(" :-;,.")
    texto = re.sub(r"\s+", " ", texto)
    if texto and texto[-1] not in ".!?":
        texto += "."
    return texto


def resposta_valida(texto):
    limpo = normalizar(texto)
    tokens = limpo.split()
    if not (7 <= len(tokens) <= 75):
        return False
    if len(texto) < 45 or len(texto) > 520:
        return False
    ruins = (
        "powered by",
        " id:",
        " ##",
        " cpf",
        " cnpj",
        "fim de cadastramento",
        "inicio da disputa",
        "abertura das propostas",
        "recebimento de propostas",
        "cor branca",
        "avaliacao",
        "observacao",
        "representantes do corpo docente",
        "prefeito municipal",
        "prefeita municipal",
        "no uso de suas atribu",
    )
    if any(item in texto.lower() for item in ruins):
        return False
    if re.search(r"\b(de|do|da|para|com|por|e)\.$", limpo):
        return False
    digitos = sum(ch.isdigit() for ch in texto)
    if digitos / max(1, len(texto)) > 0.22:
        return False
    letras = sum(ch.isalpha() for ch in texto)
    return letras / max(1, len(texto)) >= 0.45


def primeiro_match(texto, padroes):
    for padrao in padroes:
        match = re.search(padrao, texto, flags=re.I | re.S)
        if match:
            return match
    return None


def extrair_objeto(texto):
    match = primeiro_match(
        texto,
        [
            r"\bOBJETO\s*[:\-]\s*(.{45,500}?)(?=\s+(?:CONTRATANTE|CONTRATADA|CONTRATADO|VALOR|FONTE|VIG[ÊE]NCIA|PRAZO|FUNDAMENTO|DATA|ASSINATURA)\s*:?)",
            r"\bObjeto\s*[:\-]\s*(.{45,500}?)(?=\s+(?:Contratante|Contratada|Contratado|Valor|Fonte|Vig[êe]ncia|Prazo|Fundamento|Data|Assinatura)\s*:?)",
            r"\bOBJETO\s*[:\-]\s*(.{45,260}?[.;])",
            r"\bObjeto\s*[:\-]\s*(.{45,260}?[.;])",
            r"\b(?:objeto\s+e|objeto\s+é)\s+a?\s*(contrata[çc][aã]o.{45,420}?)(?:\.|\s+para\s+atender|\s+conforme)",
            r"\b(contrata[çc][aã]o\s+de\s+.{45,300}?)(?:\.|\s+para\s+atender|\s+conforme|\s+CONTRATANTE|\s+VALOR)",
        ],
    )
    if not match:
        return "", -1
    resposta = limpar_resposta(match.group(1))
    if resposta_valida(resposta):
        return resposta, match.start(1)
    return "", -1


def extrair_valor_com_objeto(texto):
    objeto, pos_objeto = extrair_objeto(texto)
    valor_match = re.search(r"\bVALOR(?:\s+GLOBAL)?\s*[:\-]\s*(R\$\s*[0-9][0-9.,]*(?:\s*\([^)]+\))?)", texto, flags=re.I)
    if not objeto or not valor_match:
        return "", -1
    resposta = f"O documento informa valor de {compactar(valor_match.group(1))} para {objeto[0].lower() + objeto[1:]}"
    if resposta_valida(resposta):
        return resposta, min(pos_objeto, valor_match.start(1))
    return "", -1


def extrair_artigo_ou_resolve(texto):
    trecho_match = primeiro_match(
        texto,
        [
            r"\bRESOLVE\s*[:.\-]?\s*(.{45,760}?)(?=\s+Art\.?\s*2[oº°]?|\s+REGISTRE|\s+PUBLIQUE|\s+CUMPRA|$)",
            r"\bArt\.?\s*1[oº°]?\s*[-.:]?\s*(.{45,760}?)(?=\s+Art\.?\s*2[oº°]?|\s+REGISTRE|\s+PUBLIQUE|\s+CUMPRA|$)",
        ],
    )
    if trecho_match:
        trecho = limpar_resposta(trecho_match.group(1))
        match_acao = re.search(
            r"\b(Nomear|Exonerar|Designar)\s+(?:o|a|ao|aos|sr\.?|sra\.?|senhor|senhora)?\s*([^,.;]{5,100}?)(?:,|\s+inscrit|\s+portador|\s+CPF|\s+RG).{0,220}?\b(?:para|do|da)\s+(?:exercer\s+)?(?:o\s+)?(?:cargo|funcao|função)\s+(?:em\s+comissao\s+)?(?:de\s+)?([^.;]{5,150})",
            trecho,
            flags=re.I | re.S,
        )
        if match_acao:
            acao = normalizar(match_acao.group(1))
            pessoa = limpar_resposta(match_acao.group(2)).strip(".")
            cargo = limpar_resposta(match_acao.group(3)).strip(".")
            cargo = re.sub(r"\s+(lotad[oa]|na|no)\s+.*$", "", cargo, flags=re.I).strip(" .")
            if acao == "nomear":
                resposta = f"Nomeou {pessoa} para o cargo de {cargo}."
            elif acao == "exonerar":
                resposta = f"Exonerou {pessoa} do cargo de {cargo}."
            else:
                resposta = f"Designou {pessoa} para o cargo ou funcao de {cargo}."
            if resposta_valida(resposta):
                return resposta, trecho_match.start(1)

    match = primeiro_match(
        texto,
        [
            r"\bRESOLVE\s*[:.\-]?\s*(?:Art\.?\s*1[oº°]?\s*[-.:]?\s*)?(.{45,520}?)(?=\s+Art\.?\s*2[oº°]?|\s+REGISTRE|\s+PUBLIQUE|\s+CUMPRA|$)",
            r"\bArt\.?\s*1[oº°]?\s*[-.:]?\s*(.{45,520}?)(?=\s+Art\.?\s*2[oº°]?|\s+REGISTRE|\s+PUBLIQUE|\s+CUMPRA|$)",
        ],
    )
    if not match:
        return "", -1
    resposta = limpar_resposta(match.group(1))
    if resposta_valida(resposta):
        return resposta, match.start(1)
    return "", -1


def extrair_ementa(texto):
    match = primeiro_match(
        texto,
        [
            r"\bEMENTA\s*[:.\-]?\s*((?:Disp[oõ]e|Institui|Autoriza|Altera|Cria|Estabelece|Declara|Regulamenta|Fixa|Abre|Aprova)\b.{45,420}?[.;])",
            r"\b((?:Disp[oõ]e|Institui|Autoriza|Altera|Cria|Estabelece|Declara|Regulamenta|Fixa|Abre|Aprova)\b.{45,420}?[.;])",
            r"\bArt\.?\s*1[oº°]?\s*[-.:]?\s*(.{45,420}?)(?=\s+Art\.?\s*2[oº°]?|$)",
        ],
    )
    if not match:
        return "", -1
    resposta = limpar_resposta(match.group(1))
    if resposta_valida(resposta):
        return resposta, match.start(1)
    return "", -1


def extrair_fiscal(texto):
    match = primeiro_match(
        texto,
        [
            r"\b(RELATORIO\s+RESUMIDO\s+DA\s+EXECUCAO\s+ORCAMENTARIA.{35,220}?)(?:\n|ORCAMENTOS|RREO|$)",
            r"\b(RELATORIO\s+DE\s+GESTAO\s+FISCAL.{35,220}?)(?:\n|ORCAMENTOS|RGF|$)",
            r"\b(DEMONSTRATIVO\s+(?:DA|DO|DAS|DOS)\s+.{35,220}?)(?:\n|ORCAMENTOS|RGF|RREO|$)",
        ],
    )
    if not match:
        return "", -1
    resposta = limpar_resposta(match.group(1))
    if resposta_valida(resposta):
        return resposta, match.start(1)
    return "", -1


def palavras_rubrica(resposta):
    norm = normalizar(resposta)
    stop = {
        "para",
        "pela",
        "pelo",
        "com",
        "das",
        "dos",
        "uma",
        "que",
        "por",
        "documento",
        "informa",
        "contratacao",
        "empresa",
        "municipio",
    }
    termos = []
    for valor in re.findall(r"R\$\s*[0-9][0-9.,]*", resposta):
        termos.append(valor)
    for valor in re.findall(r"\b[0-9]{2,}[0-9./-]*\b", resposta):
        termos.append(valor)
    for token in norm.split():
        if len(token) >= 5 and token not in stop and token not in termos:
            termos.append(token)
        if len(termos) >= 8:
            break
    return termos[:8]


def construir_contexto(doc, posicao, context_chars):
    registro = doc["registro"]
    texto = compactar(doc["texto"])
    inicio = max(0, posicao - 650)
    fim = min(len(texto), inicio + context_chars)
    trecho = texto[inicio:fim]
    return "\n".join(
        [
            "### Documento DOMPI-2025",
            f"Territorio: {registro.get('territorio', '')}",
            f"Municipio: {registro.get('municipio', '')}",
            f"Data: {registro.get('data', '')}",
            f"Tipo: {registro.get('tipo_ato_normalizado') or registro.get('tipo_ato', '')}",
            f"ID publicacao: {registro.get('id_publicacao', '')}",
            "",
            "Trecho do documento:",
            trecho,
            "",
            "### FIM_DO_TRECHO",
        ]
    )


def candidato_para_doc(doc, context_chars):
    registro = doc["registro"]
    texto = compactar(doc["texto"])
    tipo = registro.get("tipo_ato_normalizado") or registro.get("tipo_ato") or "Publicacao_oficial"
    numero = numero_ato(texto, tipo)

    extratores = []
    if tipo in {"Licitacao", "Contrato", "Edital", "Termo", "Dispensa", "Inexigibilidade"}:
        extratores.append(("valor_contratacao", extrair_valor_com_objeto))
        extratores.append(("objeto_contratacao", extrair_objeto))
    if tipo == "Portaria":
        extratores.append(("portaria", extrair_artigo_ou_resolve))
    if tipo in {"Lei", "Decreto", "Resolucao"}:
        extratores.append(("ato_normativo", extrair_ementa))
    if tipo in {"LRF_RGF", "LRF_RREO"}:
        extratores.append(("relatorio_fiscal", extrair_fiscal))

    for tema, extrator in extratores:
        resposta, posicao = extrator(texto)
        if not resposta:
            continue
        contexto = construir_contexto(doc, posicao, context_chars)
        if normalizar(resposta)[:80] not in normalizar(contexto):
            continue
        if tema == "objeto_contratacao":
            pergunta = "Explique, em uma frase, qual era o objeto do procedimento ou contrato descrito no documento."
        elif tema == "valor_contratacao":
            pergunta = "Que valor foi informado e a que contratacao esse valor se refere?"
        elif tema == "portaria":
            alvo = f" n. {numero}" if numero else ""
            pergunta = f"O que a Portaria{alvo} decidiu ou determinou no documento?"
        elif tema == "ato_normativo":
            alvo = f" n. {numero}" if numero else ""
            pergunta = f"Em termos praticos, o que o ato normativo{alvo} estabelece?"
        else:
            pergunta = "Qual demonstrativo ou relatorio fiscal e apresentado no documento?"

        return {
            "tarefa": "QA aberta contextual",
            "formato": "generate_until",
            "tema": tema,
            "dificuldade": "media",
            "pergunta": pergunta,
            "alternativas": [],
            "gabarito": resposta,
            "resposta_referencia": resposta,
            "answer_aliases": [],
            "rubric_must_include": palavras_rubrica(resposta),
            "metricas_sugeridas": [
                "token_f1",
                "bleu_unigrama",
                "rubric_recall",
                "avaliacao_manual_groundedness",
            ],
            "criterio_correcao": (
                "Resposta aberta aceita parafrase, mas deve preservar os elementos essenciais da rubrica "
                "e permanecer sustentada pelo trecho do documento."
            ),
            "contexto": contexto,
            "evidencias": [resposta],
            "origem": {
                "territorio": registro.get("territorio", ""),
                "municipio": registro.get("municipio", ""),
                "data": registro.get("data", ""),
                "tipo_ato": tipo,
                "id_publicacao": doc["id"],
                "nome_arquivo": registro.get("nome_arquivo", ""),
                "arquivo_parquet": registro.get("arquivo_parquet", ""),
            },
        }
    return None


def gerar_benchmark(docs, tamanho, context_chars, seed):
    manual = gerar_benchmark_manual(docs, tamanho, context_chars)
    if len(manual) >= tamanho:
        return manual[:tamanho]

    rng = random.Random(seed)
    docs = docs[:]
    rng.shuffle(docs)
    candidatos = []
    vistos = set()
    for doc in docs:
        item = candidato_para_doc(doc, context_chars)
        if not item:
            continue
        chave = (item["tema"], item["origem"]["municipio"], normalizar(item["resposta_referencia"])[:120])
        if chave in vistos:
            continue
        vistos.add(chave)
        candidatos.append(item)

    alvos = {
        "objeto_contratacao": 8,
        "portaria": 6,
        "ato_normativo": 5,
        "valor_contratacao": 4,
        "relatorio_fiscal": 2,
    }
    selecionados = []
    usados_ids = set()
    usados_municipios = set()
    por_tema = defaultdict(list)
    for item in candidatos:
        por_tema[item["tema"]].append(item)

    for tema, limite in alvos.items():
        for item in por_tema.get(tema, []):
            if len([x for x in selecionados if x["tema"] == tema]) >= limite:
                break
            doc_id = item["origem"]["id_publicacao"]
            municipio = normalizar(item["origem"]["municipio"])
            if doc_id in usados_ids:
                continue
            if municipio in usados_municipios and len(usados_municipios) < tamanho:
                continue
            selecionados.append(item)
            usados_ids.add(doc_id)
            usados_municipios.add(municipio)

    for item in candidatos:
        if len(selecionados) >= tamanho:
            break
        doc_id = item["origem"]["id_publicacao"]
        if doc_id in usados_ids:
            continue
        selecionados.append(item)
        usados_ids.add(doc_id)

    if len(selecionados) < tamanho:
        raise RuntimeError(f"Foram gerados apenas {len(selecionados)} itens Gold; esperado: {tamanho}.")
    return selecionados[:tamanho]


def buscar_posicao(texto, anchor, resposta):
    texto_norm = normalizar(texto)
    for alvo in (anchor, resposta):
        alvo_norm = normalizar(alvo)
        if not alvo_norm:
            continue
        indice_norm = texto_norm.find(alvo_norm[: min(60, len(alvo_norm))])
        if indice_norm >= 0:
            proporcao = indice_norm / max(1, len(texto_norm))
            return min(len(texto) - 1, max(0, int(len(texto) * proporcao)))
    return 0


def gerar_benchmark_manual(docs, tamanho, context_chars):
    por_id = {doc["id"]: doc for doc in docs}
    itens = []
    faltantes = []
    for spec in MANUAL_GOLD_SPECS[:tamanho]:
        doc = por_id.get(spec["id_publicacao"])
        if not doc:
            faltantes.append(spec["id_publicacao"])
            continue
        posicao = buscar_posicao(doc["texto"], spec.get("anchor", ""), spec["resposta"])
        contexto = construir_contexto(doc, posicao, context_chars)
        registro = doc["registro"]
        itens.append(
            {
                "tarefa": "QA aberta contextual",
                "formato": "generate_until",
                "tema": spec["tema"],
                "dificuldade": "media",
                "pergunta": spec["pergunta"],
                "alternativas": [],
                "gabarito": spec["resposta"],
                "resposta_referencia": spec["resposta"],
                "answer_aliases": [],
                "rubric_must_include": spec["rubric"],
                "metricas_sugeridas": [
                    "token_f1",
                    "bleu_unigrama",
                    "rubric_recall",
                    "avaliacao_manual_groundedness",
                ],
                "criterio_correcao": (
                    "Resposta aberta aceita parafrase, mas deve preservar os elementos essenciais da rubrica "
                    "e permanecer sustentada pelo trecho do documento."
                ),
                "contexto": contexto,
                "evidencias": [spec["anchor"], spec["resposta"]],
                "origem": {
                    "territorio": registro.get("territorio", ""),
                    "municipio": registro.get("municipio", ""),
                    "data": registro.get("data", ""),
                    "tipo_ato": registro.get("tipo_ato_normalizado") or registro.get("tipo_ato", ""),
                    "id_publicacao": doc["id"],
                    "nome_arquivo": registro.get("nome_arquivo", ""),
                    "arquivo_parquet": registro.get("arquivo_parquet", ""),
                    "curadoria": "manual_gold",
                },
            }
        )
    if faltantes:
        raise RuntimeError(f"IDs manuais nao encontrados no DOMPI: {faltantes}")
    return itens


def amostra_estratificada(docs, tamanho, seed):
    rng = random.Random(seed)
    grupos = defaultdict(list)
    for doc in docs:
        registro = doc["registro"]
        chave = (registro.get("territorio", ""), registro.get("tipo_ato_normalizado") or registro.get("tipo_ato", ""))
        grupos[chave].append(doc)
    for itens in grupos.values():
        rng.shuffle(itens)

    total = sum(len(itens) for itens in grupos.values())
    selecionados = []
    for itens in grupos.values():
        quota = int(tamanho * len(itens) / max(1, total))
        selecionados.extend(itens[:quota])
        del itens[:quota]

    sobras = [doc for itens in grupos.values() for doc in itens]
    rng.shuffle(sobras)
    selecionados.extend(sobras[: max(0, tamanho - len(selecionados))])
    rng.shuffle(selecionados)
    return selecionados[:tamanho]


def montar_documento_treino(doc):
    registro = doc["registro"]
    return "\n".join(
        [
            "### Diario de prefeitura",
            f"Territorio: {registro.get('territorio', '')}",
            f"Municipio: {registro.get('municipio', '')}",
            f"Data: {registro.get('data', '')}",
            f"Tipo: {registro.get('tipo_ato_normalizado') or registro.get('tipo_ato', '')}",
            f"ID publicacao: {doc['id']}",
            "",
            "Texto extraido:",
            doc["texto"],
            "",
            "### FIM_DO_DOCUMENTO",
            "",
        ]
    )


def salvar_corpus(output_dir, split):
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    for nome, docs in split.items():
        caminho = corpus_dir / f"{nome}.txt"
        with caminho.open("w", encoding="utf-8") as arquivo:
            for doc in docs:
                arquivo.write(montar_documento_treino(doc))
        caminhos[nome] = str(caminho)
    return caminhos


def salvar_jsonl(caminho, itens):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for indice, item in enumerate(itens, start=1):
            arquivo.write(json.dumps({**item, "id": f"gold_open_q{indice:02d}"}, ensure_ascii=False) + "\n")


def salvar_ids(caminho, docs_ou_ids):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for item in docs_ou_ids:
            arquivo.write((item if isinstance(item, str) else item["id"]) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs, linhas_validas = carregar_unicos(args.extracoes_jsonl, args.min_chars)
    benchmark = gerar_benchmark(docs, args.benchmark_size, args.context_chars, args.seed)
    benchmark_ids = {item["origem"]["id_publicacao"] for item in benchmark}

    pool = [doc for doc in docs if doc["id"] not in benchmark_ids]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    train = amostra_estratificada(pool, args.train_docs, args.seed)
    train_ids = {doc["id"] for doc in train}
    restante = [doc for doc in pool if doc["id"] not in train_ids]
    valid = amostra_estratificada(restante, args.valid_docs, args.seed + 1)
    valid_ids = {doc["id"] for doc in valid}
    restante = [doc for doc in restante if doc["id"] not in valid_ids]
    test = amostra_estratificada(restante, args.test_docs, args.seed + 2)

    split = {"train": train, "valid": valid, "test": test}
    caminhos = salvar_corpus(args.output_dir, split)

    benchmark_path = args.output_dir / "municipal_gazettes_benchmark.jsonl"
    gold_path = args.output_dir / "benchmark_gold_open_25_dompi_2025.jsonl"
    salvar_jsonl(benchmark_path, benchmark)
    salvar_jsonl(gold_path, benchmark)

    split_dir = args.output_dir / "splits"
    salvar_ids(split_dir / "train_ids.txt", train)
    salvar_ids(split_dir / "valid_ids.txt", valid)
    salvar_ids(split_dir / "test_ids.txt", test)
    salvar_ids(split_dir / "benchmark_ids.txt", sorted(benchmark_ids))

    manifest = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "fonte": str(args.extracoes_jsonl),
        "linhas_validas_antes_deduplicacao": linhas_validas,
        "documentos_unicos": len(docs),
        "deduplicacao": "Mantido um documento por id_publicacao, escolhendo o texto mais longo quando havia repeticao.",
        "splits": {nome: len(valor) for nome, valor in split.items()},
        "benchmark_size": len(benchmark),
        "benchmark_ids": len(benchmark_ids),
        "sem_vazamento": {
            "benchmark_intersect_train": len(benchmark_ids & train_ids),
            "benchmark_intersect_valid": len(benchmark_ids & valid_ids),
            "benchmark_intersect_test": len(benchmark_ids & {doc["id"] for doc in test}),
        },
        "treino_usa_perguntas": False,
        "observacao_treino": "O pre-treino continuado usa apenas texto bruto dos diarios. Perguntas Gold ficam exclusivamente para avaliacao antes/depois.",
        "amostragem_treino": "Amostra estratificada por territorio e tipo_ato_normalizado, com seed fixa.",
        "arquivos_corpus": caminhos,
        "benchmark": str(benchmark_path),
        "benchmark_gold": str(gold_path),
        "temas_benchmark": dict(Counter(item["tema"] for item in benchmark)),
        "tipos_benchmark": dict(Counter(item["origem"]["tipo_ato"] for item in benchmark)),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
