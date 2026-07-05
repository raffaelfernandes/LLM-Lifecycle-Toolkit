# 05 - Resultados

Esta seção reúne as evidências numéricas do experimento.

## Resultado central de pré-treino

Arquivo principal:

- `runs/tucano2_qwen_0p5b_loraplus_dompi_10k/summary_before_after.json`

Resumo no teste:

- Entropia cruzada: 2,5139 antes, 1,2848 depois.
- Perplexidade: 12,3529 antes, 3,6138 depois.
- Acurácia de token: 0,5422 antes, 0,7254 depois.

## Outros arquivos entregues

- `metrics.jsonl`: histórico de métricas do treinamento.
- `run_config.json`: configuração do experimento.
- `runs/benchmark_25_v2/summary_benchmark_25_v2.json`: avaliação em benchmark contextual de 25 itens, incluindo comparação com SFT quando disponível.
- `runs/specific_benchmark_25_v1/`: avaliação em benchmark específico.
- `runs/benchmark_25_combined_audited_v1/`: benchmark combinado auditado.
- `runs/tucano2_qwen_0p5b_loraplus_dompi_10k_sft_contextual_v2_balanced_250/`: métricas do SFT contextual citado no relatório final.

O pré-treino continuado é o resultado central da Questão 1; o SFT é mantido como evidência complementar do histórico descrito no relatório.
