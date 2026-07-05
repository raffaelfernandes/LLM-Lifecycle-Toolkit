# 05_results — Resultados

Resultados quantitativos e qualitativos da avaliação.

| Arquivo | Descrição |
|---|---|
| `q4_metrics.json` | Métricas das três variantes (teacher, student base, student destilado): PPL, ROUGE-L, cobertura, % português. |
| `q4_qualitative.json` | Respostas comparativas do aluno base e destilado para perguntas-âncora. |
| `q4_curva_treino.png` | Curva de perda de treino e validação (evidencia o overfitting mitigado por early stopping). |
| `split_teste.json` | Conjunto de teste usado no cálculo de perplexidade (mesmo para todas as variantes). |

**Resultado central:** houve transferência de conhecimento — perplexidade do aluno
caiu 30,7% (6,52 → 4,52) e ROUGE-L subiu de 0,175 para 0,226.
