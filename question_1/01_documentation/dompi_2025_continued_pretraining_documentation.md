# Documentacao do pre-treino continuado com DOMPI-2025

## Escopo da primeira questao

A primeira questao do trabalho pede somente o experimento de pre-treino:

- usar o dataset unificado `diariosPefeituras`;
- fazer pre-treinamento continuado de um LLM;
- avaliar o modelo antes e depois do treinamento;
- criar um benchmark com pelo menos 25 perguntas e respostas de referencia;
- calcular metricas como entropia cruzada, perplexidade e acuracia de previsao de tokens.

SFT, RAG e guardrails nao fazem parte do nucleo do pre-treino continuado. Nesta entrega, SFT aparece apenas como etapa complementar de diagnostico para mostrar que pergunta e resposta documental exige supervisao alem do pre-treino.

## Decisao sobre o dataset

O dataset usado agora e o `gutoportelaa/DOMPI-2025`, disponivel no Hugging Face:

```text
https://huggingface.co/datasets/gutoportelaa/DOMPI-2025
```

Essa escolha substitui o dataset anterior restrito a Entre Rios e Vale do Sambito. O DOMPI-2025 ja esta organizado em arquivos `.parquet`, cada um representando um territorio, e contem o texto extraido das publicacoes.

Arquivos baixados em:

```text
C:\Users\okaza\Documents\IA_Dados\data\DOMPI-2025\data\raw
```

Foram usados os 12 arquivos:

```text
carnaubais.parquet
chapada_vale_do_rio_itaim.parquet
cocais.parquet
entre_rios.parquet
mangabeiras.parquet
planice_litoran.parquet
serra_da_capivara.parquet
tabuleiros_alto_parnaiba.parquet
vale_do_caninde.parquet
vale_do_rio_guaribas.parquet
vale_do_sambito.parquet
vale_dos_rios_piaui_e_itaueiras.parquet
```

Ou seja, o experimento nao usa apenas `entre_rios.parquet`.

## Estrutura original dos dados

As colunas principais usadas do DOMPI-2025 sao:

```text
id_publicacao
territorio
municipio
tipo_ato
data_publicacao
ano
extrator
texto
n_chars
paginas
```

O campo mais importante para o pre-treino e `texto`, pois ele contem o conteudo textual extraido das publicacoes oficiais. Os demais campos entram como metadados para compor o documento de treino e o benchmark.

## Limpeza e preparacao dos dados

Script responsavel:

```text
llm_pretraining\prepare_dompi_2025.py
```

O script executa estas etapas:

1. Baixa o dataset do Hugging Face, se ainda nao estiver local.
2. Le todos os arquivos `.parquet` em lotes, para nao carregar tudo de uma vez de forma desnecessaria.
3. Extrai os campos relevantes.
4. Limpa o texto.
5. Corrige mojibake simples, por exemplo `PiauÃ­` para `Piauí`.
6. Normaliza quebras de linha e espacos repetidos.
7. Descarta textos com menos de 50 caracteres.
8. Gera um arquivo intermediario em JSONL.
9. Chama o preparador de corpus e benchmark.

Arquivo intermediario gerado:

```text
C:\Users\okaza\Documents\IA_Dados\extracoes_dompi_2025.jsonl
```

Esse JSONL mantem o texto inline, porque o DOMPI ja vem com texto extraido. Diferente do dataset antigo, nao foi necessario baixar PDFs nem extrair Markdown de cada PDF.

## Quantidade de dados apos limpeza

Relatorio gerado:

```text
C:\Users\okaza\Documents\IA_Dados\dompi_2025_report.json
```

Resumo da execucao atual:

```text
linhas_lidas: 77337
documentos_validos: 77337
documentos_descartados: 0
total_caracteres: 883775389
```

Distribuicao por territorio:

```text
Carnaubais: 7380
Chapada Vale Do Rio Itaim: 7496
Cocais: 12188
Entre Rios: 1066
Mangabeiras: 10889
Planice Litoran: 2723
Serra Da Capivara: 8747
Tabuleiros Alto Parnaiba: 7115
Vale Do Caninde: 4252
Vale Do Rio Guaribas: 8715
Vale Do Sambito: 600
Vale Dos Rios Piaui E Itaueiras: 6166
```

Depois disso, o preparador de pre-treino aplica um filtro mais rigoroso para o corpus:

```text
min_chars: 500
documentos_validos_para_pretreino: 76649
```

Esse filtro remove publicacoes muito curtas, que normalmente trazem pouco sinal para modelagem de linguagem.

## Formato do documento de treino

Cada publicacao e convertida para um bloco textual padronizado:

```text
### Diario de prefeitura
Territorio: ...
Municipio: ...
Identificador: ...
Data: ...
Tipo estimado: ...
Arquivo: ...

Texto extraido:
...

### FIM_DO_DOCUMENTO
```

Essa decisao deixa o corpus mais auditavel: o modelo recebe tanto o texto da publicacao quanto metadados basicos que tambem sao usados no benchmark.

## Explicacao do split

Script responsavel:

```text
llm_pretraining\prepare_benchmark_dataset.py
```

O split divide os documentos em tres partes:

```text
train: 90%
valid: 5%
test: restante
```

Antes da divisao, os documentos sao embaralhados com seed fixa:

```text
seed: 42
```

Isso torna a divisao reprodutivel. Se o script for executado novamente com a mesma entrada e a mesma seed, os mesmos documentos tendem a cair nos mesmos splits.

Resultado atual:

```text
total usado no corpus: 76649 documentos
train: 68984 documentos
valid: 3832 documentos
test: 3833 documentos
```

Papel de cada split:

```text
train
Usado para atualizar os pesos/adaptadores durante o pre-treino.

valid
Usado para acompanhar a qualidade durante o treino, sem ser usado para atualizar pesos.

test
Usado para avaliacao final antes e depois do treinamento.
```

A separacao evita medir o modelo apenas nos textos que ele viu durante treinamento.

Arquivos gerados:

```text
llm_pretraining\data_dompi_2025\corpus\train.txt
llm_pretraining\data_dompi_2025\corpus\valid.txt
llm_pretraining\data_dompi_2025\corpus\test.txt
```

## Benchmark de 25 perguntas

O benchmark e gerado a partir do split `test`, ou do `valid` apenas se o `test` estiver vazio.

Arquivo:

```text
llm_pretraining\data_dompi_2025\municipal_gazettes_benchmark.jsonl
```

Foram geradas 25 perguntas. O script tenta diversificar os municipios: primeiro seleciona documentos de municipios diferentes; se faltarem documentos, completa com os demais.

Tipos de pergunta:

```text
QA factual institucional
- municipio da publicacao
- territorio da publicacao
- data da publicacao
- identificador da publicacao

Classificacao
- tipo de ato administrativo

Roteamento
- categoria administrativa do documento
```

Cada item do benchmark contem:

```text
id
tarefa
formato
tema
pergunta
alternativas, quando houver
gabarito
resposta_referencia
metricas_sugeridas
criterio_correcao
contexto
origem
```

Esse benchmark serve para coletar respostas do modelo antes e depois do pre-treino.

## Modelo escolhido

O modelo final da entrega e:

```text
Polygl0t/Tucano2-qwen-0.5B-Base
```

Motivos:

- e pequeno o suficiente para execucao local com GPU de 6 GB usando LoRA+;
- e um modelo causal LM, adequado para pre-treinamento continuado;
- e voltado ao portugues, o que combina melhor com diarios oficiais municipais;
- deriva da familia Qwen, mas foi continuado para portugues.

## Avaliacao antes e depois do treino

O experimento final avalia o mesmo modelo antes e depois do pre-treino continuado, usando validacao, teste e benchmark.

Arquivos principais da entrega:

```text
05_results\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\run_config.json
05_results\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\metrics.jsonl
05_results\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\summary_before_after.json
06_model_adapters\continued_pretraining_final
```

Configuracao final:

```text
modelo: Polygl0t/Tucano2-qwen-0.5B-Base
estrategia: LoRA+
epochs: 3
planned_steps: 15876
global_step final: 15873
block_size: 512
stride: 512
grad_accum_steps: 16
learning_rate: 0.0001
loraplus_lr_ratio: 16
```

Resultado principal no teste:

```text
cross_entropy: 2.5139 -> 1.2848
perplexity: 12.3529 -> 3.6138
token_accuracy: 0.5422 -> 0.7254
```

Interpretacao: o pre-treino continuado foi bem-sucedido como adaptacao de linguagem ao dominio DOMPI-2025. A queda de perplexidade e o aumento da acuracia de token sao as evidencias centrais da primeira questao.

## Metricas calculadas

### Entropia cruzada

A entropia cruzada mede o erro medio por token na previsao do proximo token. No script, ela e calculada a partir da loss do modelo causal LM:

```text
cross_entropy = total_loss / total_tokens
```

Quanto menor, melhor.

### Perplexidade

A perplexidade e derivada da entropia cruzada:

```text
perplexity = exp(cross_entropy)
```

Ela pode ser interpretada como o grau de incerteza medio do modelo ao prever o proximo token. Quanto menor, melhor.

### Acuracia de previsao de tokens

A acuracia de token compara o token mais provavel previsto pelo modelo com o token real seguinte:

```text
token_accuracy = tokens_corretos / total_tokens
```

Quanto maior, melhor.

### Benchmark de perguntas e respostas

No benchmark, o modelo gera respostas curtas. A avaliacao registra:

```text
accuracy_or_exact_match
bleu_unigrama
por_formato
```

Para perguntas abertas, a correcao usa comparacao normalizada: remove acentos, coloca em minusculas e ignora pontuacao. Para multipla escolha, aceita a letra correta ou o texto da alternativa correta.

## Pre-treino continuado

Script de orquestracao final:

```text
llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1
```

Comando para treino completo:

```powershell
powershell -ExecutionPolicy Bypass -File .\llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1
```

Comando para teste rapido:

```powershell
powershell -ExecutionPolicy Bypass -File .\llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1 -MaxSteps 10 -EvalEvery 5 -SaveEvery 10
```

O script usa:

```text
data_dir: llm_pretraining\data_dompi_2025_tucano2_10k
output_dir: llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k
model_id: Polygl0t/Tucano2-qwen-0.5B-Base
epochs: 3
block_size: 512
grad_accum_steps: 16
LoRA+: ativo
```

O script de treino `train_continued_pretraining_gpu.py` faz automaticamente:

1. carrega o modelo;
2. avalia antes do treino em `valid`, `test` e benchmark;
3. salva as respostas antes em `benchmark_outputs\before.jsonl`;
4. treina o modelo;
5. avalia depois do treino em `valid`, `test` e benchmark;
6. salva as respostas depois em `benchmark_outputs\after.jsonl`;
7. salva o resumo antes/depois.

## Saidas do treino

Pasta principal:

```text
llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k
```

Arquivos esperados:

```text
run_config.json
metrics.jsonl
summary_before_after.json
benchmark_outputs\before.jsonl
benchmark_outputs\after.jsonl
final\
checkpoint-*
```

O arquivo mais importante para o relatorio final e:

```text
summary_before_after.json
```

Ele contem:

```text
metricas antes
metricas depois
delta de cross_entropy
delta de perplexity
delta de token_accuracy
```

## Como interpretar o resultado

O pre-treino foi bem-sucedido se, depois do treinamento:

```text
cross_entropy diminuir
perplexity diminuir
token_accuracy aumentar
```

O benchmark de perguntas e respostas pode ou nao melhorar muito, porque pre-treino continuado ensina distribuicao de linguagem e dominio, mas nao necessariamente ensina o modelo a seguir instrucoes. Por isso, a metrica principal da primeira questao deve ser linguagem:

```text
cross_entropy
perplexity
token_accuracy
```

As respostas do benchmark entram como evidencia qualitativa e quantitativa complementar.

## Comandos de reproducao

Preparar tudo sem treinar:

```powershell
powershell -ExecutionPolicy Bypass -File .\llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1 -PrepareOnly
```

Avaliar modelo base antes do treino:

```powershell
.\.venv-llm\Scripts\python.exe .\llm_pretraining\evaluate_base_model_dompi_2025.py --model-id Polygl0t/Tucano2-qwen-0.5B-Base --data-dir .\llm_pretraining\data_dompi_2025_tucano2_10k
```

Treinar teste rapido:

```powershell
powershell -ExecutionPolicy Bypass -File .\llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1 -MaxSteps 10 -EvalEvery 5 -SaveEvery 10
```

Treinar completo:

```powershell
powershell -ExecutionPolicy Bypass -File .\llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1
```

Continuar treino interrompido:

```powershell
powershell -ExecutionPolicy Bypass -File .\llm_pretraining\run_tucano2_dompi_10k_loraplus.ps1 -Resume
```

## Estado atual

Ja foram concluidos:

```text
download do DOMPI-2025
conversao para extracoes_dompi_2025.jsonl
limpeza basica e correcao de mojibake
geracao do corpus train/valid/test
geracao do benchmark com 25 perguntas
criacao do script de avaliacao antes do treino
criacao do script de pre-treino continuado
avaliacao baseline do modelo antes do treino
coleta das 25 respostas do benchmark antes do treino
pre-treino continuado
avaliacao depois do treino
comparacao final das metricas
geracao do adaptador final
SFT contextual complementar para diagnostico de QA
```
