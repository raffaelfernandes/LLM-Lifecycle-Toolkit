# Manifesto da Entrega Final

Esta pasta entrega a Questão 1 do trabalho: pré-treino continuado de um LLM com diários oficiais municipais do DOMPI-2025. O relatório definitivo está em `02_report/llm_final_report.pdf`.

## Artefatos entregues

- Relatório final definitivo em PDF: `02_report/llm_final_report.pdf`.
- Documentação técnica do experimento: `01_documentation/dompi_2025_continued_pretraining_documentation.md`.
- Scripts usados para preparar dados, treinar e avaliar: `03_code/`.
- Benchmarks e manifestos auditáveis: `04_benchmark_data/`.
- Métricas de antes/depois do pré-treino: `05_results/runs/tucano2_qwen_0p5b_loraplus_dompi_10k/`.
- Adaptador final LoRA/PEFT do pré-treino: `06_model_adapters/continued_pretraining_final/`.
- Artefatos SFT complementares citados no relatório: `05_results/runs/tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250/` e `06_model_adapters/contextual_sft_final/`.

## Resultado principal entregue

O modelo base `Polygl0t/Tucano2-qwen-0.5B-Base` foi adaptado ao domínio DOMPI-2025 com LoRA+.

| Métrica de teste | Antes | Depois |
| --- | ---: | ---: |
| Entropia cruzada | 2,5139 | 1,2848 |
| Perplexidade | 12,3529 | 3,6138 |
| Acurácia de token | 0,5422 | 0,7254 |

## Escopo

Esta entrega cobre o pré-treino continuado, avaliação antes/depois, benchmark de perguntas e artefatos de rastreabilidade. O SFT aparece como segundo estágio histórico/complementar no relatório final, porque foi usado para analisar a limitação do pré-treino em pergunta e resposta. RAG e guardrails aparecem apenas como discussão de continuidade.

## Curadoria aplicada

Foram removidos checklists de preparação, modelos antigos, relatórios de referência, ambiente virtual, caches, checkpoints intermediários e corpus bruto completo. A pasta mantém o relatório final, os resultados centrais do pré-treino e o apêndice SFT necessário para sustentar o histórico descrito no PDF.
