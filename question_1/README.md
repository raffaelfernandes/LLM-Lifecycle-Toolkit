# Questão 1 - Entrega Final: Pré-treino Continuado de LLM

Esta pasta contém a entrega final da primeira questão do trabalho de Desenvolvimento de IA / Tópicos em IA. O foco é o experimento de pré-treino continuado de um modelo de linguagem usando publicações oficiais do dataset DOMPI-2025.

## Objetivo

O que está sendo entregue:

1. Preparar o dataset textual.
2. Dividir os dados em treino, validação e teste.
3. Criar benchmark com perguntas e respostas de referência.
4. Avaliar o modelo antes do treinamento.
5. Fazer pré-treino continuado com LoRA/LoRA+.
6. Avaliar o modelo depois do treinamento.
7. Adaptador final LoRA/PEFT do pré-treino continuado.

## Modelo e dados

- Modelo base: `Polygl0t/Tucano2-qwen-0.5B-Base`.
- Dataset: `gutoportelaa/DOMPI-2025`.
- GPU usada no experimento: NVIDIA GeForce RTX 4050 Laptop GPU.
- Treinamento: 3 épocas, 15.873 passos concluídos, bloco de 512 tokens.
- Parâmetros treináveis: 10.092.544, cerca de 2,01% do modelo.

## Resultado principal

No conjunto de teste, o pré-treino continuado reduziu a perplexidade e aumentou a acurácia de previsão de tokens:

| Métrica | Antes | Depois | Variação |
| --- | ---: | ---: | ---: |
| Entropia cruzada, teste | 2,5139 | 1,2848 | -1,2291 |
| Perplexidade, teste | 12,3529 | 3,6138 | -8,7391 |
| Acurácia de token, teste | 0,5422 | 0,7254 | +0,1832 |

O benchmark aberto de 25 perguntas é mantido como evidência complementar. A documentação explica a diferença entre métricas de linguagem, que melhoraram, e respostas abertas exatas, que exigem avaliação mais cuidadosa.

## Pastas

- `final_delivery.md`: manifesto dos artefatos entregues.
- `01_documentation/`: documentação técnica, planejamento, auditorias e relatório JSON do preparo dos dados.
- `02_report/`: relatório final definitivo em PDF.
- `03_code/`: scripts de preparação, treino e avaliação usados no experimento.
- `04_benchmark_data/`: manifestos, benchmarks e arquivos de auditoria; não inclui corpus bruto completo.
- `05_results/`: métricas, comparações e saídas pequenas do pré-treino continuado, com apêndice SFT citado no relatório.
- `06_model_adapters/`: adaptador final do pré-treino continuado e adaptador SFT complementar.

## O que não foi copiado

- Ambiente virtual `.venv`.
- `__pycache__`.
- `token_cache`.
- Corpus bruto completo.
- Checkpoints intermediários `checkpoint-*`.
- Dataset DOMPI-2025 bruto em `.parquet`.
- Materiais antigos de referência, modelos de relatório e checklists de preparação.

Esses itens são reproduzíveis ou pesados demais para uma entrega limpa. Os manifestos, scripts e resultados preservam a rastreabilidade do experimento.

## Ordem de leitura da entrega

Comece por:

1. `final_delivery.md`.
2. `02_report/llm_final_report.pdf`.
3. `01_documentation/dompi_2025_continued_pretraining_documentation.md`.
4. `05_results/runs/tucano2_qwen_0p5b_loraplus_dompi_10k/summary_before_after.json`.
5. `04_benchmark_data/data_dompi_2025/benchmark_gold_contextual_25_v2.jsonl`.
