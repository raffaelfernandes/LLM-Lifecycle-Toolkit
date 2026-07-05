# 04_benchmark_data — Dados e Benchmark

Dataset de treino e benchmark de avaliação da Questão 4.

| Arquivo | Descrição |
|---|---|
| `dataset_cot_limpo.json` | 1.077 pares CoT válidos (`instruction, input, reasoning, answer`), usados no treino de destilação. |
| `benchmark_q4.json` | 100 perguntas de Ciência da Computação com resposta de referência e categoria, para avaliação. |
| `excluir_do_treino.json` | Lista de instruções do benchmark removidas do treino, para evitar vazamento de dados. |

Todas as 100 perguntas do benchmark têm interseção conceitual nula com o conjunto
de treino (exclusão exata + remoção de quase-duplicatas com Jaccard > 0,7).
