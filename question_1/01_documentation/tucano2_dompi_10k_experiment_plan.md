# Planejamento do experimento Tucano2 + DOMPI-2025

Este documento descreve o plano tecnico da Questao 1 do trabalho final: fazer pre-treinamento continuado de um LLM com diarios de prefeituras e avaliar o modelo antes e depois do treinamento.

## 1. Objetivo

O objetivo principal e verificar se um modelo pequeno, treinado continuamente no dominio dos diarios municipais, melhora sua adaptacao linguistica ao DOMPI-2025 e se essa adaptacao aparece nas respostas a perguntas abertas sobre documentos oficiais.

O requisito formal da Questao 1 e:

- usar o dataset unificado `diariosPrefeituras`;
- escolher um LLM;
- fazer pre-treinamento continuado;
- avaliar o modelo antes e depois;
- criar benchmark com pelo menos 25 perguntas e respostas de referencia;
- reportar perplexidade, entropia cruzada e acuracia de previsao de tokens.

## 2. Decisoes principais

### 2.1 Modelo escolhido

Modelo: `Polygl0t/Tucano2-qwen-0.5B-Base`.

Motivo:

- e pequeno o suficiente para a RTX 4050 Laptop de 6 GB;
- tem cerca de 490M parametros, proximo ao Qwen2.5-0.5B;
- foi continuado em portugues, o que e mais adequado para diarios oficiais brasileiros;
- e modelo base, nao instruct, portanto combina melhor com pre-treinamento continuado.

Alternativas consideradas:

- `Qwen/Qwen2.5-0.5B`: ja testado antes, mas menos especializado em portugues;
- `Qwen/Qwen3-0.6B-Base`: mais novo, mas nao tao especifico para portugues quanto Tucano2;
- modelos 1.5B ou maiores: melhores em capacidade geral, mas pouco praticos para 6 GB de VRAM em treino local longo.

Referencias:

- Tucano2: https://huggingface.co/Polygl0t/Tucano2-qwen-0.5B-Base
- Qwen2.5: https://arxiv.org/abs/2412.15115
- Qwen3: https://arxiv.org/abs/2505.09388

### 2.2 Metodo de treino

Metodo: pre-treinamento continuado com LoRA+.

Motivo:

- treinar todos os pesos seria pesado para 6 GB de VRAM;
- LoRA reduz a quantidade de parametros treinaveis;
- LoRA+ usa taxas de aprendizado diferentes para as matrizes LoRA e pode acelerar convergencia sem mudar muito o consumo de VRAM;
- o ambiente local ja tem `peft 0.19.1` e suporta `create_loraplus_optimizer`.

Configuracao executada:

- `lora_r = 16`;
- `lora_alpha = 32`;
- `lora_dropout = 0.05`;
- `learning_rate = 0.0001`;
- `loraplus_lr_ratio = 16`;
- `block_size = 512`;
- `train_batch_size = 1`;
- `gradient_accumulation_steps = 16`;
- `epochs = 3`;
- `eval_every = 500`;
- `save_every = 1000`.

Smoke test ja executado:

- modelo carregou na GPU;
- LoRA+ foi criado corretamente;
- parametros treinaveis: 10.092.544;
- parametros totais: 500.891.648;
- percentual treinavel: 2,01%;
- 1 passo de treino executou com sucesso.

Referencias:

- LoRA: https://arxiv.org/abs/2106.09685
- LoRA+: https://proceedings.mlr.press/v235/hayou24a.html

## 3. Preparacao dos dados

Arquivo bruto usado:

`C:\Users\okaza\Documents\IA_Dados\extracoes_dompi_2025.jsonl`

Foram encontradas:

- 76.649 linhas validas antes de deduplicacao;
- 67.130 documentos unicos por `id_publicacao`.

Decisao: deduplicar por `id_publicacao`.

Por que:

- o mesmo diario aparece repetido em mais de um territorio;
- se duplicatas forem mantidas, o modelo pode memorizar documentos repetidos;
- a avaliacao pode ficar contaminada se uma publicacao aparecer no treino e no benchmark.

Regra usada:

- manter um documento por `id_publicacao`;
- quando havia repeticao, manter o texto mais longo.

Script:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\prepare_tucano2_dompi_10k_experiment.py`

## 4. Split do experimento

O benchmark foi separado primeiro e depois removido de todos os splits de linguagem.

Split final:

- treino: 10.000 documentos;
- validacao: 1.000 documentos;
- teste: 1.000 documentos;
- benchmark Gold: 25 documentos/perguntas;
- intersecao benchmark x treino: 0;
- intersecao benchmark x validacao: 0;
- intersecao benchmark x teste: 0.

Por que o benchmark tambem fica fora do teste de linguagem:

- o teste de linguagem mede perda por token em texto bruto;
- o benchmark mede respostas a perguntas;
- manter os dois separados evita duvida sobre vazamento e deixa a comparacao mais limpa.

Por que usar amostra de 10.000 documentos:

- o DOMPI completo e grande demais para uma execucao local rapida;
- 10.000 documentos ainda cobrem varios territorios e tipos de ato;
- permite rodar 3 epocas em tempo viavel na RTX 4050;
- a amostra foi estratificada por territorio e tipo de ato para reduzir vies de pegar apenas os primeiros arquivos.

Arquivos do split:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\splits\train_ids.txt`

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\splits\valid_ids.txt`

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\splits\test_ids.txt`

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\splits\benchmark_ids.txt`

Manifesto:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\manifest.json`

## 5. Benchmark de perguntas abertas

O benchmark Gold tem 25 perguntas abertas, curadas manualmente a partir de documentos do DOMPI-2025.

Formato de cada item:

- pergunta aberta;
- resposta de referencia;
- rubrica com elementos essenciais;
- trecho de contexto;
- `id_publicacao` de origem;
- metadados: municipio, data, tipo de ato, arquivo.

Por que perguntas abertas:

- o professor pediu perguntas menos triviais;
- perguntas como "qual municipio publicou?" ou "qual a data?" medem extracao simples;
- perguntas abertas avaliam se o modelo consegue sintetizar o que ocorreu no documento.

Exemplos de tipos de pergunta:

- "O que ocorreu nas portarias da Camara Municipal de Alto Longa publicadas nesse trecho?"
- "Que contratacao foi descrita para a Prefeitura de Alegrete do Piaui e qual valor mensal foi informado?"
- "Qual apresentacao artistica foi contratada para o aniversario da cidade?"
- "O que o Decreto n. 22/2025 estabeleceu sobre luto oficial?"
- "Qual demonstrativo fiscal foi apresentado no documento?"

Arquivo do benchmark:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\benchmark_gold_open_25_dompi_2025.jsonl`

Arquivo usado pelo avaliador:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\municipal_gazettes_benchmark.jsonl`

## 6. Bronze, Silver e Gold

O experimento usa a ideia de camadas de qualidade:

### Bronze

Dados brutos extraidos do DOMPI-2025:

`extracoes_dompi_2025.jsonl`

Uso:

- preservar texto original;
- preservar metadados;
- permitir auditoria.

### Silver

Candidatos automaticos de perguntas e respostas.

Uso:

- apoiar busca de bons exemplos;
- detectar padroes de objeto, portaria, lei, decreto e relatorio fiscal;
- nao usar diretamente como benchmark final, porque o OCR mistura publicacoes em algumas paginas.

### Gold

25 perguntas revisadas manualmente.

Uso:

- avaliacao antes/depois;
- resposta de referencia auditavel;
- rubrica de correcao;
- nenhum item Gold entra no treino.

## 7. Avaliacao antes do treino

A avaliacao antes do treino ja foi executada em 20/06/2026 as 16:08:32.

Modelo avaliado:

`Polygl0t/Tucano2-qwen-0.5B-Base`

Metricas de linguagem:

Validacao:

- cross-entropy: 2.6515;
- perplexity: 14.1748;
- token accuracy: 0.5341.

Teste:

- cross-entropy: 2.5139;
- perplexity: 12.3529;
- token accuracy: 0.5422.

Metricas do benchmark aberto com 25 perguntas:

- accuracy por rubrica/exato: 0.04;
- BLEU unigram: 0.2063;
- token-F1: 0.2466;
- rubric recall: 0.2286.

Arquivos do baseline:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\baseline_before_training\resultado_baseline_before.json`

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\baseline_before_training\benchmark_before.jsonl`

O arquivo `benchmark_before.jsonl` contem as respostas concretas do modelo antes do treino.

## 8. Treinamento executado

Comando principal:

```powershell
.\.venv-llm\Scripts\python.exe -u .\llm_pretraining\train_continued_pretraining_gpu.py `
  --model-id Polygl0t/Tucano2-qwen-0.5B-Base `
  --data-dir .\llm_pretraining\data_dompi_2025_tucano2_10k `
  --output-dir .\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k `
  --epochs 3 `
  --block-size 512 `
  --stride 512 `
  --train-batch-size 1 `
  --eval-batch-size 1 `
  --grad-accum-steps 16 `
  --learning-rate 0.0001 `
  --warmup-steps 100 `
  --eval-every 500 `
  --save-every 1000 `
  --eval-max-batches 96 `
  --benchmark-max-items 25 `
  --benchmark-max-new-tokens 128 `
  --use-loraplus `
  --loraplus-lr-ratio 16 `
  --resume-from latest
```

O treino salva:

- `metrics.jsonl`: metricas durante o treino;
- `checkpoint-1000`, `checkpoint-2000`, etc.;
- `benchmark_outputs/before.jsonl`;
- `benchmark_outputs/after.jsonl`;
- `summary_before_after.json`;
- `final/`: adaptador final LoRA.

Diretorio do run:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k`

Logs ativos:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\logs`

## 9. Avaliacao depois do treino e resultado final

O treino completo terminou em 21/06/2026, com avaliacao final registrada em:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\summary_before_after.json`

Modelo final:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\final`

Resultado de modelagem de linguagem:

| Split | Cross-entropy antes | Cross-entropy depois | Perplexity antes | Perplexity depois | Token accuracy antes | Token accuracy depois |
|---|---:|---:|---:|---:|---:|---:|
| Validacao | 2.6515 | 1.3438 | 14.1748 | 3.8337 | 0.5341 | 0.7200 |
| Teste | 2.5139 | 1.2848 | 12.3529 | 3.6138 | 0.5422 | 0.7254 |

Interpretacao:

- o pre-treinamento continuado foi bem-sucedido como adaptacao de linguagem ao DOMPI;
- a perplexidade caiu de forma forte em validacao e teste;
- a acuracia de token subiu cerca de 18 pontos percentuais em validacao e teste;
- isso mostra que o modelo ficou melhor em prever texto do dominio dos diarios oficiais.

Resultado bruto do benchmark Gold de 25 perguntas abertas:

| Metrica | Antes | Depois | Leitura |
|---|---:|---:|---|
| Exact match / accuracy | 0.0400 | 0.0000 | piorou |
| BLEU unigrama | 0.2063 | 0.0767 | piorou |
| Token-F1 | 0.2466 | 0.1184 | piorou |
| Rubric recall | 0.2286 | 0.0973 | piorou |

Interpretacao inicial do benchmark:

- o benchmark bruto de QA aberta nao melhorou com pre-treinamento continuado;
- isso nao invalida o experimento, porque pre-treinamento continuado nao e o mesmo que SFT/instruction tuning;
- a conclusao correta e separar duas coisas: o modelo aprendeu melhor a distribuicao textual do DOMPI, mas nao aprendeu sozinho a responder perguntas abertas com fidelidade;
- para melhorar QA aberta, a etapa posterior recomendada e SFT com pares pergunta-resposta de treino ou RAG sobre os documentos, sem usar o benchmark Gold no treino.
- apos auditoria dos contextos, esse benchmark deve ser tratado como preliminar, porque parte dos itens Gold cobra elementos que nao aparecem claramente no contexto fornecido ao modelo.

## 10. Como a comparacao de respostas foi preservada na entrega

A tabela comparativa preserva, para cada uma das 25 perguntas:

- ID da pergunta;
- municipio/data/tipo;
- pergunta;
- resposta de referencia;
- resposta antes do treino;
- resposta depois do treino;
- token-F1 antes/depois;
- rubric recall antes/depois;
- comentario manual curto.

Arquivos gerados:

Antes:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\benchmark_outputs\before.jsonl`

Depois:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\benchmark_outputs\after.jsonl`

Resumo final:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\summary_before_after.json`

Tabela apresentavel com as 25 respostas concretas:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\tucano2_dompi_response_comparison.md`

Auditoria de validade dos contextos Gold:

`C:\Users\okaza\Documents\IA_Dados\llm_pretraining\benchmark_gold_contexts_audit.md`

Leitura global das 25 perguntas antes da auditoria:

- em perguntas abertas, as respostas ficaram menos alinhadas ao gabarito depois do pre-treinamento;
- exemplos fortes dessa queda aparecem em itens como `gold_open_q13`, `gold_open_q19` e `gold_open_q21`, nos quais o Token-F1 antes era relativamente alto e caiu bastante depois;
- alguns itens ficaram parecidos ou levemente melhores, como `gold_open_q04`, `gold_open_q11`, `gold_open_q17`, `gold_open_q20` e `gold_open_q24`, mas a media geral caiu.

Depois da auditoria, a tabela deve ser usada com cautela. Ela mostra evidencia empirica das respostas antes/depois, mas nao deve ser tratada como benchmark conclusivo enquanto os itens frageis nao forem revisados.

## 11. Riscos e mitigacoes

Risco: benchmark contaminado pelo treino.

Mitigacao:

- os 25 `id_publicacao` do benchmark sao removidos antes da amostragem de treino, validacao e teste;
- o manifesto registra intersecao zero.

Risco: OCR mistura atos de municipios diferentes.

Mitigacao:

- perguntas Gold foram curadas manualmente;
- cada item tem trecho de evidencia;
- itens automaticos ruins nao foram aceitos como Gold.

Risco: pre-treino nao melhora respostas abertas.

Mitigacao:

- separar conclusao linguistica de conclusao QA;
- explicar que SFT/RAG seriam etapas posteriores para melhorar resposta a perguntas;
- ainda assim manter comparacao empirica das respostas antes/depois.

Risco: VRAM insuficiente.

Mitigacao:

- usar LoRA+ em vez de full fine-tuning;
- batch 1;
- gradient accumulation 16;
- gradient checkpointing ativo;
- checkpoints para retomar se interromper.

## 12. Estado atual

Concluido:

- DOMPI-2025 carregado localmente;
- deduplicacao por `id_publicacao`;
- split 10k/1k/1k + benchmark 25 sem vazamento;
- benchmark Gold aberto curado manualmente;
- baseline antes do treino executado;
- respostas concretas antes do treino salvas;
- smoke test de treino com Tucano2 + LoRA+ executado com sucesso;
- treino completo Tucano2 + LoRA+ com 3 epocas;
- avaliacao depois do treino;
- benchmark Gold depois do treino;
- auditoria dos contextos do benchmark Gold;
- benchmark contextual corrigido e compacto criado;
- reavaliacao antes/depois no benchmark contextual corrigido;
- comparacao numerica antes/depois;
- tabela lado a lado das 25 respostas concretas antes/depois;
- modelo final salvo.

Fechado para esta entrega:

- os exemplos qualitativos validados foram mantidos nas comparacoes e discutidos no relatorio final;
- SFT e RAG foram tratados como analise complementar e continuidade, sem substituir o resultado central de pre-treinamento continuado da Questao 1.

## 13. O que isso significa para a atividade

Para a Questao 1, o resultado consolidado fica apresentado assim:

1. O objetivo de pre-treinamento continuado foi cumprido: as metricas de linguagem melhoraram fortemente em validacao e teste.
2. O benchmark aberto original foi aplicado antes e depois, sem usar os documentos Gold no treino, mas a auditoria mostrou que parte dos itens Gold precisava de revisao.
3. O benchmark foi corrigido com contexto reconstruido da publicacao original e filtragem por ancoragem minima da rubrica; 12 itens foram mantidos e 13 removidos.
4. A reavaliacao corrigida continuou sem ganho em QA aberta: Token-F1 caiu de 0.2676 para 0.1313 e rubric recall caiu de 0.1778 para 0.1083.
5. A explicacao tecnica e que pre-treinamento causal melhora previsao de texto do dominio, mas nao treina comportamento instrucional nem recuperacao fiel de fatos.
6. A proxima etapa natural, se o objetivo for responder perguntas com qualidade, e treinar SFT/RAG separado, mantendo o benchmark Gold reservado somente para avaliacao.

Portanto, a entrega nao deve afirmar que o modelo ficou melhor em perguntas abertas. A entrega deve afirmar que o modelo ficou melhor em linguagem do dominio DOMPI e que a avaliacao de QA ainda precisa de curadoria mais rigorosa para ser conclusiva.

## 14. Revisao metodologica das perguntas abertas

A critica sobre perguntas abertas sem contexto e valida em termos metodologicos: um modelo pequeno nao deve ser avaliado como se tivesse que adivinhar fatos especificos de um diario oficial apenas pela pergunta. Para uma avaliacao justa de QA em documentos, o item precisa ser tratado como leitura contextual/open-book ou precisa usar recuperacao de contexto.

Neste experimento, o benchmark Gold nao foi aplicado sem contexto. O avaliador montou cada prompt no seguinte padrao:

```text
Responda de forma curta usando apenas o contexto.

Contexto:
<trecho do documento DOMPI>

Pergunta: <pergunta aberta>
Resposta:
```

Foi verificado com o tokenizer do modelo final que os 25 prompts do benchmark ficaram entre 315 e 931 tokens, abaixo do limite de 1024 tokens usado na avaliacao. Portanto, nenhum item Gold foi truncado na avaliacao antes/depois.

O que precisa ficar claro no relatorio:

- a tabela de comparacao mostra pergunta, gabarito e respostas, mas omite o contexto completo para ficar legivel;
- os arquivos `before.jsonl` e `after.jsonl` preservam o campo `contexto` usado no prompt;
- o benchmark avaliou leitura contextual, nao adivinhacao sem documento;
- a auditoria mostrou que alguns itens Gold nao estavam bem ancorados no contexto, como `gold_open_q01`;
- mesmo assim, as perguntas devem sempre apontar para um trecho/documento especifico e pedir uma resposta ancorada no contexto.

Padrao correto para os proximos itens Gold:

1. Incluir sempre o contexto no prompt de avaliacao.
2. Formular perguntas abertas ancoradas no trecho, por exemplo: "Segundo o trecho, o que ocorreu na Portaria n. X?".
3. Evitar perguntas vagas quando o contexto mistura atos diferentes; nesses casos, citar municipio, data, tipo de ato ou numero do ato.
4. Exigir resposta curta e baseada apenas no contexto.
5. Guardar uma resposta de referencia e uma rubrica com elementos obrigatorios.
6. Incluir item "nao consta no trecho" somente quando a tarefa permitir resposta abstentiva.
7. Nao usar os documentos Gold no treino, mantendo o benchmark reservado para avaliacao.

Base academica:

- SQuAD define QA como compreensao de leitura: perguntas sobre artigos, com respostas extraidas do trecho correspondente.
- Natural Questions apresenta ao anotador uma pergunta e uma pagina da Wikipedia, marcando resposta longa, resposta curta ou ausencia de resposta.
- OpenBookQA segue a ideia de prova com livro aberto: o objetivo nao e memorizar, mas aplicar fatos disponiveis.
- DPR e RAG mostram que, quando o contexto nao e dado diretamente, QA aberta precisa de recuperacao de passagens; confiar so na memoria parametrica do modelo e limitado para tarefas intensivas em conhecimento.

Referencias para citar:

- Rajpurkar et al. (2016), SQuAD: `https://aclanthology.org/D16-1264/`
- Kwiatkowski et al. (2019), Natural Questions: `https://aclanthology.org/Q19-1026/`
- Mihaylov et al. (2018), OpenBookQA: `https://arxiv.org/abs/1809.02789`
- Karpukhin et al. (2020), Dense Passage Retrieval: `https://aclanthology.org/2020.emnlp-main.550/`
- Lewis et al. (2020), RAG: `https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html`

Conclusao metodologica: o formato de avaliacao estava correto por usar contexto e nao ter truncamento, mas a curadoria de parte dos itens Gold falhou porque alguns gabaritos cobravam informacoes ausentes ou pouco claras no trecho fornecido. Por isso, a avaliacao de QA foi refeita com benchmark contextual corrigido, descrito na secao seguinte.

## 15. Correcao do benchmark e nova reavaliacao

Procedimento aplicado:

1. Para cada item Gold, recuperar o texto original da publicacao em `extracoes_dompi_2025.jsonl` usando `id_publicacao`.
2. Selecionar uma janela compacta do texto original que maximize a presenca dos elementos obrigatorios da rubrica.
3. Manter somente itens com pelo menos 80% dos elementos da rubrica presentes no contexto corrigido.
4. Remover itens que continuaram sem ancoragem suficiente.
5. Reavaliar o modelo base e o modelo final LoRA no mesmo benchmark corrigido.

Resultado da curadoria:

- itens mantidos: 12;
- itens removidos: 13;
- maior prompt corrigido: 1725 tokens;
- limite usado na reavaliacao: 2048 tokens;
- geracao: 64 novos tokens por resposta.

Arquivos gerados:

- benchmark corrigido: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025\benchmark_gold_contextual_corrigido_compacto.jsonl`
- auditoria do benchmark corrigido: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\audit_benchmark_gold_contexts_compact.md`
- respostas antes/depois corrigidas: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\tucano2_dompi_response_comparison_corrected.md`
- resumo numerico: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\contextual_corrected_compact_summary_before_after.json`

Nova comparacao de QA contextual corrigida:

| Metrica | Antes | Depois | Delta |
|---|---:|---:|---:|
| Exact match / accuracy | 0.0000 | 0.0000 | 0.0000 |
| BLEU unigrama | 0.2247 | 0.0966 | -0.1280 |
| Token-F1 | 0.2676 | 0.1313 | -0.1362 |
| Rubric recall | 0.1778 | 0.1083 | -0.0694 |

Conclusao apos correcao:

- a critica sobre o benchmark original era procedente;
- a reavaliacao corrigida removeu itens injustos e recuperou contexto correto para itens como `gold_open_q01`;
- mesmo assim, o modelo final nao melhorou em QA aberta contextual;
- a conclusao fica mais forte: o pre-treinamento continuado melhorou metricas de linguagem, mas nao melhorou a capacidade de responder perguntas abertas ancoradas em contexto;
- para melhorar a tarefa principal de perguntas e respostas, o caminho tecnico indicado e uma etapa separada de SFT ou RAG.

## 16. Como o modelo foi treinado e como deve ser usado

O modelo nao treinou usando o benchmark.

Verificacao dos splits:

- treino: 10.000 documentos;
- validacao: 1.000 documentos;
- teste: 1.000 documentos;
- benchmark Gold: 25 documentos/perguntas;
- intersecao benchmark x treino: 0;
- intersecao benchmark x validacao: 0;
- intersecao benchmark x teste: 0.

O proprio manifesto registra `treino_usa_perguntas: false`.

O que entrou no treino:

- apenas texto bruto dos diarios oficiais do DOMPI;
- arquivo de treino: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\corpus\train.txt`;
- aproximadamente 126 milhoes de caracteres e 19,7 milhoes de tokens por separacao simples em espacos;
- objetivo de treino: prever o proximo token, isto e, causal language modeling;
- nao houve treino supervisionado de pergunta-resposta nessa etapa.

Configuracao do treino:

- modelo base: `Polygl0t/Tucano2-qwen-0.5B-Base`;
- metodo: LoRA com LoRA+;
- parametros treinaveis: 10.092.544, cerca de 2,01% do modelo;
- epocas: 3;
- `block_size`: 512;
- `gradient_accumulation`: 16;
- `learning_rate`: 0.0001.

Consequencia pratica:

- o modelo ficou melhor em continuar/prever texto do dominio dos diarios oficiais;
- isso nao significa que ele virou um banco de dados consultavel;
- fatos especificos de uma publicacao, como nome de contratado, valor, data ou atracao artistica, devem ser respondidos com contexto recuperado;
- sem documento/contexto, o modelo tende a completar a pergunta com algo plausivel, podendo alucinar.

Teste fechado sem contexto:

Pergunta:

```text
Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?
```

Resposta do modelo final sem contexto:

```text
A musica escolhida e a "Samba da Vida", que conta com uma composicao popular, sendo um dos mais famosos.
```

Com instrucao para admitir incerteza:

```text
A artista escolhida e a cantora LUCAS DA SILVA RODRIGUES, com uma versao musical
```

Ambas estao incorretas. No benchmark contextual corrigido, quando o trecho do documento foi fornecido, a resposta depois do treino foi:

```text
Galicia Cruz
```

Isso mostra a diferenca entre:

- pergunta fechada/sem contexto: depende de memoria parametrica e e insegura;
- pergunta contextual/RAG: o sistema busca o documento e passa o trecho ao modelo.

Uso real recomendado:

1. Usuario pergunta em linguagem natural.
2. O sistema busca no acervo DOMPI os documentos mais relevantes usando municipio, data, tipo de ato, nomes e termos da pergunta.
3. O sistema passa os trechos recuperados para o modelo.
4. O modelo responde apenas com base nesses trechos.
5. Se nao houver evidencia suficiente, o modelo deve responder que precisa de mais informacao ou que nao consta nos documentos recuperados.

Exemplo de uso real:

Usuario:

```text
Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?
```

Sistema internamente recupera:

- municipio: Boqueirao do Piaui;
- tipo: Licitacao/contratacao artistica;
- data/publicacao relacionada;
- trecho do DOMPI com a contratacao.

Modelo recebe:

```text
Contexto:
<trecho recuperado do DOMPI>

Pergunta:
Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?

Resposta:
```

Resposta esperada:

```text
Foi contratada a atracao artistica Galicia Cruz para show nas comemoracoes do aniversario de Boqueirao do Piaui.
```

O que fazer para melhorar a capacidade de responder perguntas:

1. Implementar RAG sobre todos os documentos DOMPI, com busca por embeddings e metadados.
2. Criar pares supervisionados de treino no formato `contexto + pergunta -> resposta`, usando apenas documentos de treino, nao o benchmark Gold.
3. Fazer SFT/LoRA de instrucao para ensinar o modelo a responder curto, com evidencia e sem inventar quando o contexto nao contem a resposta.
4. Adicionar exemplos negativos: perguntas cuja resposta nao consta no trecho, para o modelo aprender a dizer "nao consta no contexto".
5. Manter o benchmark Gold corrigido separado apenas para avaliacao.
6. Avaliar em dois modos separados:
   - closed-book: sem contexto, para medir memoria parametrica;
   - open-book/RAG: com contexto recuperado, que e o modo correto para documentos oficiais.

## 17. Teste de perguntas genericas sem contexto

Tambem foi testado se o modelo final ficou melhor em perguntas conceituais/genericas, sem documento anexado.

Perguntas testadas:

- "O que e uma portaria municipal?"
- "O que e um diario oficial municipal?"
- "O que e uma licitacao municipal?"
- "Qual a diferenca entre decreto e portaria municipal?"
- "O que e um portal municipal?"

Resultado observado:

- o modelo base respondeu de forma mais limpa e conceitual;
- o modelo final pre-treinado em DOMPI comecou algumas respostas corretamente, mas frequentemente continuou com trechos no estilo de diario oficial, edital, observacao de prova ou publicacao;
- isso indica adaptacao de estilo/domino, mas nao melhora de capacidade instrucional generica.

Exemplo:

Pergunta:

```text
O que e uma portaria municipal?
```

Resposta do modelo base:

```text
Uma Portaria Municipal e um documento oficial emitido pelo Poder Executivo local para regulamentar normas internas relacionadas a aspectos administrativos municipais.
```

Resposta do modelo final:

```text
Sera um documento oficial emitido pelo poder publico Municipal com informacoes sobre o cargo ou funcao publica exercida por quem quer que seja..
Obs.: A resposta devera ser escrita no minimo duas linhas apos a pergunta acima citada.
Ano Xxill - Teresina...
```

Interpretacao:

- o pre-treinamento continuado em texto bruto nao ensina o modelo a seguir melhor instrucoes;
- ele pode ate reduzir a qualidade de resposta generica porque aproxima a distribuicao de saida do estilo dos documentos DOMPI;
- para perguntas conceituais, um modelo instruct ou SFT posterior tende a ser melhor;
- para perguntas factuais sobre documentos, o caminho correto continua sendo RAG ou contexto explicito;
- para o produto final, o ideal e combinar RAG + SFT de instrucao, nao apenas pre-treinamento continuado.

## 18. Implementacao inicial de RAG e preparo de SFT

Foi implementada uma primeira versao funcional de RAG lexical usando SQLite FTS5.

Arquivos criados:

- indexador: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\criar_rag_sqlite_dompi.py`
- script de pergunta RAG: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\perguntar_rag_dompi.py`
- indice completo: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\rag_dompi_2025.sqlite`
- gerador SFT contextual: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\preparar_sft_dompi_contextual.py`
- dataset SFT: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\sft_contextual`

Resultado da indexacao RAG:

- documentos indexados: 67.411;
- chunks indexados: 576.062;
- tempo aproximado de indexacao completa: 5,5 minutos;
- nao houve treino nessa etapa, apenas criacao de indice de busca.

Exemplo de pergunta real:

```text
Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?
```

Sem RAG, o modelo inventou respostas.

Com RAG, o sistema recuperou documentos de Boqueirao do Piaui, incluindo a publicacao `c059d21bd38ec66ec9d3c3208fe99705`, e a resposta gerada comecou corretamente com:

```text
Galicia Cruz
```

Observacao: o modelo ainda tende a continuar com residuos de diario oficial; por isso o script RAG aplica pos-processamento para retornar somente a primeira resposta util. O SFT deve melhorar esse comportamento.

Dataset SFT contextual preparado:

- documentos de treino usados: 8.000;
- exemplos de treino: 35.888;
- documentos de validacao usados: 500;
- exemplos de validacao: 2.257;
- benchmark Gold excluido;
- tarefas geradas: metadata, extracao de contratado, extracao de valor, extracao de objeto, extracao de assunto e abstencao.

Estimativa de tempo:

- RAG/indexacao: minutos, nao horas;
- pergunta RAG com modelo carregado: segundos a dezenas de segundos;
- pergunta RAG carregando o modelo do zero: pode levar alguns minutos no notebook;
- SFT curto, 100 a 200 steps: dezenas de minutos;
- SFT medio, 500 steps: cerca de 1 a 4 horas;
- SFT maior, 1000+ steps: algumas horas, dependendo do `max_length` e da temperatura da GPU;
- SFT completo em todo o dataset por epocas pode demorar bastante, mas ainda tende a ser menor e mais direcionado que o pre-treinamento de 26 horas.

Comandos principais:

Criar ou recriar o indice RAG:

```powershell
.\.venv-llm\Scripts\python.exe .\llm_pretraining\criar_rag_sqlite_dompi.py `
  --output-db .\llm_pretraining\rag_dompi_2025.sqlite `
  --chunk-chars 1800 `
  --overlap-chars 300
```

Perguntar usando RAG:

```powershell
.\.venv-llm\Scripts\python.exe .\llm_pretraining\perguntar_rag_dompi.py `
  --db .\llm_pretraining\rag_dompi_2025.sqlite `
  --question "Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?" `
  --top-k 3 `
  --context-chars 5000 `
  --max-new-tokens 80
```

Gerar dataset SFT contextual:

```powershell
.\.venv-llm\Scripts\python.exe .\llm_pretraining\preparar_sft_dompi_contextual.py `
  --max-train-docs 8000 `
  --max-valid-docs 500 `
  --context-chars 2200 `
  --max-examples-per-doc 5
```

Treinar SFT contextual a partir do adaptador de pre-treinamento:

```powershell
.\.venv-llm\Scripts\python.exe -u .\llm_pretraining\treinar_sft_instrucoes_gpu.py `
  --model-id Polygl0t/Tucano2-qwen-0.5B-Base `
  --sft-dir .\llm_pretraining\data_dompi_2025_tucano2_10k\sft_contextual `
  --benchmark-path .\llm_pretraining\data_dompi_2025\benchmark_gold_contextual_corrigido_compacto.jsonl `
  --pretrain-adapter .\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\final `
  --output-dir .\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200 `
  --max-length 1024 `
  --benchmark-block-size 2048 `
  --max-steps 200 `
  --grad-accum-steps 8 `
  --eval-every 50 `
  --save-every 100 `
  --learning-rate 0.0001 `
  --benchmark-max-items 12
```

Papel de cada etapa:

- RAG faz o modelo encontrar o documento certo;
- SFT ensina o modelo a usar o contexto, responder curto e nao inventar;
- pre-treinamento continuado ajuda no dominio textual, mas nao substitui RAG nem SFT.

## 19. Resultado do SFT contextual curto e teste RAG + SFT

Foi executado um SFT contextual curto de 200 passos usando o adaptador final do pre-treinamento DOMPI como ponto de partida.

Artefatos:

- run SFT: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200`
- modelo/adaptador final SFT: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200\final`
- checkpoint intermediario: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200\checkpoint-100`
- checkpoint final: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200\checkpoint-200`
- respostas antes do SFT: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200\benchmark_outputs\before_sft.jsonl`
- respostas depois do SFT: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200\benchmark_outputs\after_sft.jsonl`
- teste RAG + SFT: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_200\rag_sft_outputs\testes_rag_sft.jsonl`

Configuracao executada:

- base: `Polygl0t/Tucano2-qwen-0.5B-Base`
- adaptador inicial: `tucano2_qwen_0p5b_loraplus_dompi_10k\final`
- `max_length`: 1024 para caber na VRAM;
- `benchmark_block_size`: 2048 para nao truncar o benchmark contextual corrigido;
- `grad_accum_steps`: 8;
- `max_steps`: 200;
- `learning_rate`: 0.0001;
- GPU usada: RTX 4050 Laptop 6 GB, com uso aproximado entre 4,5 GB e 5,0 GB de VRAM.

Evolucao da perda SFT em validacao:

| fase | passo | cross-entropy | perplexity |
|---|---:|---:|---:|
| antes do SFT | 0 | 2.0980 | 8.1497 |
| durante SFT | 50 | 0.4489 | 1.5666 |
| durante SFT | 100 | 0.3484 | 1.4168 |
| durante SFT | 150 | 0.3077 | 1.3603 |
| depois do SFT | 200 | 0.3035 | 1.3546 |

Interpretacao da perda: o modelo aprendeu rapidamente o formato de resposta curta com contexto nos exemplos SFT. Isso nao significa, sozinho, que ele passou a responder bem perguntas abertas complexas; por isso o benchmark Gold continuou sendo necessario.

Benchmark contextual corrigido antes/depois do SFT:

| metrica | antes do SFT | depois do SFT | leitura |
|---|---:|---:|---|
| exact/accuracy normalizada | 0.0000 | 0.0833 | passou de 0 para 1 acerto em 12 |
| BLEU unigrama | 0.0921 | 0.0913 | praticamente estavel |
| token-F1 | 0.1283 | 0.1901 | melhorou sobreposicao lexical |
| rubric recall | 0.1083 | 0.1069 | praticamente estavel |

Exemplos concretos:

| item | antes | depois | leitura |
|---|---|---|---|
| `gold_open_q04` | resposta longa e com valores errados | `R$ 5.075,00...` | melhorou no valor, mas ainda incompleto |
| `gold_open_q11` | resposta generica sobre licenca | `Regime Juridico Unico...` | melhorou bastante lexicalmente |
| `gold_open_q25` | `Galicia Cruz` | `GALICIA PRODUCOES... GALICIA CRUZ` | manteve parte correta e trouxe empresa |
| `gold_open_q01` | resposta generica | `Portaria` | piorou por ficar curto demais |

Conclusao do SFT curto:

- o SFT melhorou o comportamento de resposta curta e algumas extracoes;
- ainda nao e suficiente para o objetivo final de QA aberta robusta;
- o dataset SFT usado tem muitas tarefas atomicas, como metadata, valor, contratado e objeto;
- o benchmark Gold cobra respostas compostas, com varios campos no mesmo texto;
- portanto, o proximo SFT deve incluir mais exemplos compostos do tipo "o que ocorreu", "que contratacao foi feita e por qual valor", "quem foi nomeado/exonerado", sempre com contexto e resposta sintetica.

Teste RAG + SFT:

Foi testado o fluxo real: pergunta do usuario -> recuperacao lexical SQLite FTS5 -> contexto -> modelo SFT.

Resultados observados:

| pergunta | resposta RAG + SFT | observacao |
|---|---|---|
| `Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?` | `MAURINHO DO ACOREON` | resposta sustentada por um documento recuperado, mas a pergunta e ambigua; outro documento de Boqueirao aparece no top 2 com `Galicia Cruz` |
| `O que ocorreu nas portarias da Camara Municipal de Alto Longa?` | vazia | pergunta sem data/trecho recupera varias portarias diferentes; nao ha como saber qual "portarias" o usuario quer |
| `Que contratacao foi descrita para a Prefeitura de Alegrete do Piaui e qual valor mensal foi informado?` | `Prazo de vigencia: 12 meses` | o trecho com `R$ 5.075,00` foi recuperado, mas o modelo escolheu a linha errada |
| `Qual foi o fato principal registrado no extrato de rescisao de Alto Longa sobre aulas de ingles?` | resposta incorreta sobre fiscal/suplente | recuperou documento relevante no top 1, mas a geracao ainda falhou na sintese |

Interpretacao metodologica:

- o RAG melhora o problema de "memoria", porque traz documentos reais para o prompt;
- mas pergunta aberta factual continua precisando de desambiguacao: municipio, data, tipo de ato, numero do contrato, nome da pessoa/empresa ou trecho;
- perguntas do tipo "nesse trecho" so fazem sentido quando o trecho e fornecido explicitamente ou quando o sistema ja selecionou um documento;
- se houver varios documentos plausiveis, o sistema deve mostrar as fontes recuperadas ou pedir refinamento;
- o proximo gargalo nao e VRAM, e sim qualidade de recuperacao/reranqueamento e qualidade dos exemplos SFT compostos.

Proximas decisoes tecnicas:

1. Melhorar o RAG com reranqueamento por entidade, municipio, data, tipo de ato, numero de contrato e proximidade dos termos.
2. Gerar SFT v2 com perguntas abertas compostas e respostas de 1 a 3 frases, sem usar o benchmark Gold no treino.
3. Manter o benchmark Gold corrigido apenas como avaliacao antes/depois.
4. Reportar no trabalho que pre-treinamento reduziu perplexidade de linguagem, enquanto RAG/SFT sao as etapas necessarias para QA.

## 20. SFT v2 balanceado

Apos o SFT curto inicial, foi criada uma segunda versao de dataset SFT, porque a primeira estava muito dominada por perguntas atomicas de metadata e extracao simples. Esse comportamento explicou respostas excessivamente curtas, como `Portaria`, `Contrato` ou apenas um valor.

Mudancas no gerador:

- arquivo alterado: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\preparar_sft_dompi_contextual.py`
- foram adicionadas tarefas compostas:
  - `sintese_contratacao`;
  - `sintese_rescisao`;
  - `sintese_ato`;
- as respostas SFT passaram a aceitar ate 420 caracteres, para permitir respostas com objeto, contratado e valor.

Dataset v2 bruto:

- caminho: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\sft_contextual_v2`
- treino: 41.709 exemplos;
- validacao: 2.643 exemplos;
- benchmark Gold continuou excluido.

Dataset v2 balanceado:

- caminho: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025_tucano2_10k\sft_contextual_v2_balanced`
- criterio: manter tarefas de extracao/sintese e limitar metadata/abstencao;
- treino: 14.952 exemplos;
- validacao: 1.160 exemplos;
- tarefas de treino:
  - `extracao_objeto`: 2.741;
  - `extracao_contratado`: 2.125;
  - `sintese_contratacao`: 1.908;
  - `extracao_assunto`: 1.169;
  - `sintese_ato`: 1.078;
  - `extracao_valor`: 765;
  - `sintese_rescisao`: 166;
  - `metadata`: 2.500;
  - `abstencao`: 2.500.

Comando executado:

```powershell
.\.venv-llm\Scripts\python.exe -u .\llm_pretraining\treinar_sft_instrucoes_gpu.py `
  --model-id Polygl0t/Tucano2-qwen-0.5B-Base `
  --sft-dir .\llm_pretraining\data_dompi_2025_tucano2_10k\sft_contextual_v2_balanced `
  --benchmark-path .\llm_pretraining\data_dompi_2025\benchmark_gold_contextual_corrigido_compacto.jsonl `
  --pretrain-adapter .\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k\final `
  --output-dir .\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250 `
  --max-length 1024 `
  --benchmark-block-size 2048 `
  --max-steps 250 `
  --grad-accum-steps 8 `
  --eval-every 50 `
  --save-every 125 `
  --learning-rate 0.00008 `
  --benchmark-max-items 12
```

Artefatos:

- run: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250`
- modelo final: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250\final`
- checkpoint 125: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250\checkpoint-125`
- checkpoint 250: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250\checkpoint-250`
- benchmark antes: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250\benchmark_outputs\before_sft.jsonl`
- benchmark depois: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250\benchmark_outputs\after_sft.jsonl`
- teste RAG + v2: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250\rag_sft_outputs\testes_rag_sft_v2.jsonl`

Evolucao da perda SFT v2:

| fase | passo | cross-entropy | perplexity |
|---|---:|---:|---:|
| antes do SFT v2 | 0 | 1.3705 | 3.9373 |
| durante SFT v2 | 50 | 0.9103 | 2.4851 |
| durante SFT v2 | 100 | 0.7971 | 2.2191 |
| durante SFT v2 | 150 | 0.7529 | 2.1232 |
| durante SFT v2 | 200 | 0.7407 | 2.0973 |
| depois do SFT v2 | 250 | 0.7381 | 2.0920 |

Benchmark contextual corrigido no SFT v2:

| metrica | antes | depois | leitura |
|---|---:|---:|---|
| exact/accuracy normalizada | 0.0000 | 0.0833 | 1 acerto em 12 |
| BLEU unigrama | 0.0921 | 0.0726 | caiu |
| token-F1 | 0.1283 | 0.1780 | melhorou, mas menos que v1 |
| rubric recall | 0.1083 | 0.1194 | melhorou levemente |

Comparacao com SFT v1:

- v1 teve token-F1 maior: 0.1901;
- v2 teve rubric recall maior: 0.1194;
- v2 produziu algumas respostas qualitativamente melhores, como `Taty Girl` e `Galicia Cruz`;
- v2 ainda errou itens de portaria e rescisao por nao extrair bem nomeacao/exoneracao e detalhes contratuais completos.

Exemplos depois do SFT v2:

| item | resposta v2 | leitura |
|---|---|---|
| `gold_open_q04` | `R$ 5.075,00...` | acertou o valor, mas faltou contratado/objeto |
| `gold_open_q10` | `Taty Girl` | resposta curta e correta no elemento principal |
| `gold_open_q25` | `Galicia Cruz` | resposta curta e correta no elemento principal |
| `gold_open_q01` | `Dispoe sobre nomeacao de servidor publico` | generica; nao capturou exoneracao/nomeacao especificas |
| `gold_open_q03` | `Dispoe sobre nomeagao... Conselho Tutelar` | incorreta; mostra que rescisao ainda precisa de exemplos melhores |

Teste RAG + SFT v2:

| pergunta | resposta | diagnostico |
|---|---|---|
| `Qual atracao artistica foi contratada para o aniversario de Boqueirao do Piaui?` | `Washington Brasileira` | pergunta ambigua; RAG trouxe varios documentos plausiveis de Boqueirao, e o modelo escolheu uma entidade do trecho errado |
| `O que ocorreu nas portarias da Camara Municipal de Alto Longa?` | `Dispoe sobre A NOMEACAO` | pergunta sem data/trecho; ha muitas portarias de Alto Longa |
| `Que contratacao foi descrita para a Prefeitura de Alegrete do Piaui e qual valor mensal foi informado?` | `Contratacao temporaria dos servicos como Medic` | recuperou o documento com `R$ 5.075,00`, mas geracao escolheu uma linha errada |
| `Qual foi o fato principal registrado no extrato de rescisao de Alto Longa sobre aulas de ingles?` | `Dispoe sobre a nomeagao de fiscal...` | recuperou documento relevante, mas sintese de rescisao falhou |

Conclusao apos v2:

- treinar por mais steps no mesmo formato nao e a melhor proxima acao;
- o SFT precisa de exemplos supervisionados mais proximos do benchmark, especialmente:
  - portarias com nomeacao/exoneracao;
  - rescisao/distrato com contrato, contratado, objeto e tipo de rescisao;
  - contratacao artistica com artista, empresa, municipio, evento, data e valor;
- o RAG precisa de reranqueamento e/ou pergunta de desambiguacao antes de gerar resposta;
- para uma pergunta leiga ambigua, o sistema correto deve responder com fontes candidatas ou pedir refinamento, nao fingir certeza.

## 21. Benchmark contextual v2 com 25 perguntas e avaliacao final do SFT

Apos a revisao metodologica das perguntas abertas, foi criada uma nova versao do benchmark com 25 itens contextuais. A mudanca principal foi tornar explicito que a pergunta depende do trecho fornecido ao modelo. Assim, em vez de perguntas como `O que ocorreu nas portarias...` sem apontar o documento, o padrao passou a ser:

`Considerando o trecho, ...`

Isso segue o formato correto para QA sobre documentos oficiais: o modelo nao deve adivinhar pela memoria parametrica; ele deve responder com base no contexto recuperado ou fornecido.

Artefatos criados:

- gerador do benchmark: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\generate_contextual_benchmark_25_v2.py`
- benchmark final com 25 itens: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025\benchmark_gold_contextual_25_v2.jsonl`
- manifesto do benchmark: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025\benchmark_gold_contextual_25_v2.manifest.json`
- script de avaliacao: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\evaluate_benchmark_25_v2.py`
- respostas antes/depois: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\benchmark_25_v2`
- tabela completa em CSV: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\benchmark_25_v2\benchmark_25_v2_comparison.csv`

Distribuicao do benchmark contextual v2:

| tema | quantidade |
|---|---:|
| licitacao | 6 |
| relatorio_fiscal | 2 |
| ato_normativo | 4 |
| contratacao | 9 |
| portaria | 3 |
| rescisao | 1 |

Controle de vazamento:

| verificacao | resultado |
|---|---:|
| intersecao com treino | 0 |
| intersecao com validacao | 0 |
| intersecao com benchmark Gold antigo | 0 |

Observacao: este benchmark v2 foi criado a partir de documentos reservados fora do treino. Ele nao foi usado no pre-treinamento continuado nem no SFT. O benchmark antigo de 25 perguntas fica documentado como versao preliminar auditada; a versao contextual v2 e a versao recomendada para apresentar a comparacao final da Questao 1.

Comando executado:

```powershell
.\.venv-llm\Scripts\python.exe .\llm_pretraining\evaluate_benchmark_25_v2.py
```

Comparacao final no benchmark contextual v2 de 25 perguntas:

| modelo avaliado | itens | acertos automaticos | BLEU unigrama | token-F1 | rubric recall |
|---|---:|---:|---:|---:|---:|
| apos pre-treino continuado | 25 | 0/25 | 0.0452 | 0.0797 | 0.0467 |
| apos SFT contextual v2 | 25 | 7/25 | 0.0479 | 0.1659 | 0.3067 |

Resultado por tema depois do SFT v2:

| tema | itens | token-F1 medio | rubric recall medio | acertos |
|---|---:|---:|---:|---:|
| ato_normativo | 4 | 0.465 | 0.750 | 3 |
| contratacao | 9 | 0.187 | 0.519 | 4 |
| licitacao | 6 | 0.053 | 0.000 | 0 |
| portaria | 3 | 0.063 | 0.000 | 0 |
| relatorio_fiscal | 2 | 0.000 | 0.000 | 0 |
| rescisao | 1 | 0.093 | 0.000 | 0 |

Exemplos concretos:

| item | gabarito resumido | resposta antes | resposta depois | leitura |
|---|---|---|---|---|
| `gold25_v2_q09` | nomeacao dos membros da Comissao do Processo Seletivo Simplificado | repetiu a pergunta | `Dispoe sobre a nomeacao dos membros da Comissao do Processo Seletivo Simplificado da Secretaria` | melhora clara; F1 0.049 -> 0.609; rubric 0 -> 1 |
| `gold25_v2_q21` | contratacao de empresa para fornecimento de material grafico impresso | `Sim` | `Contratacao de empresa especializada para o fornecimento de material grafico impresso...` | melhora parcial; F1 0 -> 0.308; rubric 0 -> 0.667 |
| `gold25_v2_q01` | dispensa de licitacao para entidade/fundacao de concurso publico | resposta generica sobre CPL | resposta incorreta sobre direitos das mulheres | erro persistente; mostra que licitacao ainda precisa de RAG/reranqueamento e SFT mais direcionado |

Interpretacao para a Questao 1:

- o pre-treinamento continuado melhorou fortemente as metricas de linguagem do dominio DOMPI, mas nao bastou para QA aberta;
- o SFT contextual v2 melhorou a capacidade de responder com base no contexto, principalmente em atos normativos e contratacoes;
- o resultado ainda nao e suficiente para um sistema final de perguntas abertas, porque licitacoes, portarias, relatorios fiscais e rescisoes continuam fracos;
- para responder perguntas reais de usuario leigo, o caminho recomendado e RAG com reranqueamento + SFT de instrucao, mantendo este benchmark de 25 itens apenas para avaliacao.

## 22. Benchmark especifico de 25 perguntas com resposta curta

Depois da avaliacao contextual v2, foi criado um segundo benchmark complementar, com perguntas menos abertas. A motivacao foi reduzir ambiguidade do enunciado e medir uma capacidade mais objetiva: localizar e copiar campos especificos do documento.

Base metodologica:

- SQuAD usa leitura contextual em que a resposta e um trecho do proprio paragrafo; isso inspira respostas curtas e extraiveis.
- SQuAD 2.0 mostra a importancia de controlar se a resposta esta ou nao sustentada pelo contexto, para evitar chute.
- Natural Questions separa resposta longa e resposta curta, o que ajuda a documentar evidencias e gabarito objetivo.
- HotpotQA reforca o uso de fatos de suporte para explicar por que uma resposta foi aceita.
- BEIR mostra a importancia de benchmarks heterogeneos e de avaliacao separada para recuperacao/QA, especialmente quando o sistema usa RAG.

Referencias:

- SQuAD: https://arxiv.org/abs/1606.05250
- SQuAD 2.0: https://aclanthology.org/P18-2124/
- Natural Questions: https://aclanthology.org/Q19-1026/
- HotpotQA: https://arxiv.org/abs/1809.09600
- BEIR: https://arxiv.org/abs/2104.08663

Decisoes para este benchmark:

- perguntas com mais contexto no enunciado: municipio, data, tipo de documento, objeto resumido ou campo alvo;
- resposta curta, preferencialmente um span do trecho;
- evidencias armazenadas no proprio item;
- exclusao de documentos usados em treino, validacao, benchmark Gold antigo e benchmark contextual v2;
- manutencao de ruido OCR quando ele aparece no proprio documento, porque o modelo tambem recebera esse texto no contexto.

Artefatos:

- gerador: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\generate_specific_benchmark_25_v1.py`
- benchmark: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025\benchmark_gold_specific_25_v1.jsonl`
- manifesto: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\data_dompi_2025\benchmark_gold_specific_25_v1.manifest.json`
- respostas e metricas: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\specific_benchmark_25_v1`
- tabela completa CSV: `C:\Users\okaza\Documents\IA_Dados\llm_pretraining\runs\specific_benchmark_25_v1\specific_benchmark_25_v1_comparison.csv`

Distribuicao:

| tema | quantidade |
|---|---:|
| campo_objeto | 7 |
| campo_contratado | 5 |
| campo_valor | 5 |
| modalidade_licitacao | 4 |
| portaria_artigo_1 | 3 |
| demonstrativo_fiscal | 1 |

Controle de vazamento:

| verificacao | resultado |
|---|---:|
| intersecao com treino | 0 |
| intersecao com validacao | 0 |
| intersecao com benchmark Gold antigo | 0 |
| intersecao com benchmark contextual v2 | 0 |

Exemplos de perguntas:

| tipo | exemplo |
|---|---|
| objeto | `No documento DOMPI de Paulistana, data 26/05/2025, qual texto aparece no campo OBJETO?` |
| contratado | `No trecho em que o OBJETO menciona 'prestacao de servicos...', qual parte aparece como CONTRATADA/CONTRATADO?` |
| valor | `No trecho sobre 'contratacao de empresa especializada em engenharia...', qual valor monetario aparece no campo VALOR ou no extrato?` |
| modalidade | `No documento de Piracuruca, cujo objeto menciona 'prestacao de servicos advocaticios...', qual procedimento licitatorio e citado?` |
| portaria | `Na portaria de Piracuruca, data 15/01/2025, qual acao aparece no Art. 1?` |

Como usar na atividade:

- o benchmark contextual v2 avalia resposta aberta curta com contexto;
- o benchmark especifico v1 avalia extracao objetiva de campos;
- os dois devem ser apresentados como avaliacoes complementares;
- o especifico v1 e mais justo para medir se o modelo consegue ler o trecho e extrair a informacao correta sem depender de adivinhacao.

Comando de avaliacao executado:

```powershell
.\.venv-llm\Scripts\python.exe .\llm_pretraining\evaluate_benchmark_25_v2.py `
  --benchmark .\llm_pretraining\data_dompi_2025\benchmark_gold_specific_25_v1.jsonl `
  --output-dir .\llm_pretraining\runs\specific_benchmark_25_v1 `
  --max-new-tokens 80 `
  --block-size 2048 `
  --max-items 25
```

Resultado antes/depois:

| modelo avaliado | itens | acertos automaticos | BLEU unigrama | token-F1 | rubric recall |
|---|---:|---:|---:|---:|---:|
| apos pre-treino continuado | 25 | 5/25 | 0.1112 | 0.1558 | 0.1858 |
| apos SFT contextual v2 | 25 | 6/25 | 0.2027 | 0.2490 | 0.3075 |

Resultado por tema depois do SFT v2:

| tema | itens | token-F1 medio | rubric recall medio | acertos |
|---|---:|---:|---:|---:|
| campo_contratado | 5 | 0.554 | 0.630 | 3 |
| campo_objeto | 7 | 0.245 | 0.257 | 2 |
| campo_valor | 5 | 0.211 | 0.000 | 0 |
| modalidade_licitacao | 4 | 0.082 | 0.250 | 0 |
| portaria_artigo_1 | 3 | 0.120 | 0.067 | 0 |
| demonstrativo_fiscal | 1 | 0.000 | 0.000 | 0 |

Leitura:

- o benchmark especifico confirma que perguntas com mais pistas e resposta curta sao mais justas para avaliar leitura de documento;
- o SFT v2 melhorou de forma consistente em F1 e rubric recall, mas ainda nao resolveu campos de valor, modalidade e portaria;
- `campo_contratado` foi o melhor caso, sugerindo que o modelo aprendeu melhor extracao nominal do que extracao de valor ou classificacao de modalidade;
- para a proxima versao do SFT, os exemplos supervisionados devem privilegiar perguntas especificas por campo, no mesmo formato deste benchmark, mas sem usar estes 25 itens no treino.
