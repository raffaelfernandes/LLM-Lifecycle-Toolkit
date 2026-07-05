# Entrega Final — Questão 4: Destilação de Conhecimento

**Disciplina:** Tópicos em Inteligência Artificial (2026.1) — DC/UFPI
**Professor:** Prof. Dr. Raimundo Moura
**Grupo:** Émery Freitas Moriconi, Eryck Kawã Pereira Torres, Felipe Lages de Lima,
Raffael Ferreira Fernandes

---

## 1. O que a questão pedia

Investigar quais LLMs são usados em destilação; definir teacher e student; gerar um
dataset sintético; destilar o professor para o aluno; criar um benchmark de 100
perguntas; avaliar professor e aluno, antes e depois; e analisar se houve
transferência de conhecimento.

## 2. O que foi entregue

| Requisito | Como foi atendido |
|---|---|
| Investigar LLMs de destilação | Seção 2 do relatório (teachers, students, casos reais) |
| Definir teacher e student | Qwen2.5-7B-Instruct → Qwen2.5-0.5B-Instruct (mesma família) |
| Dataset sintético | 1.077 pares CoT gerados pelo professor via Ollama |
| Destilação | SFT + LoRA (Step-by-Step / CoT) no Colab |
| Benchmark de 100 perguntas | `benchmark_q4.json`, estratificado, sem vazamento |
| Avaliação antes/depois | 3 variantes: teacher, student base, student destilado |
| Análise de transferência | Veredito positivo: PPL −30,7%, ROUGE-L +0,051 |

## 3. Resultado

**Houve transferência de conhecimento.** O aluno destilado superou o aluno base em
perplexidade (6,52 → 4,52, redução de 30,7%) e em ROUGE-L (0,175 → 0,226). A análise
qualitativa mostrou que a destilação transferiu o comportamento de raciocínio e o
estilo do professor, mas a transferência de conhecimento factual foi parcial —
resultado esperado e coerente com a literatura, dado o porte de 0.5B do aluno.

## 4. Decisões técnicas principais

| Decisão | Justificativa |
|---|---|
| Teacher/student da mesma família (Qwen2.5) | Mesmo tokenizer; sem estender vocabulário |
| Destilação CoT (não logits) | Restrição de memória + transferir raciocínio, não só decisão |
| LoRA em vez de SFT completo | Viável na T4; reaproveita experiência da Q3 |
| Pré-filtro de chunks na geração | 92% de aproveitamento; descarta código/inglês/listas |
| Limpeza de contexto no reasoning | Recuperou 330 pares que seriam descartados |
| Anti-leakage por quase-duplicata (Jaccard) | Garante avaliação de generalização real |
| `load_best_model_at_end` | Mitiga o overfitting observado no treino |

## 5. Limitações e trabalhos futuros

As principais limitações são o porte reduzido do aluno (0.5B), o dataset restrito
(1.077 pares) e o overfitting observado no treino. Como continuidade: ampliar o
dataset, usar alunos intermediários (1.5B–3B) e aplicar filtragem por confiança do
professor para reduzir a propagação de erros factuais.

## 6. Artefatos

| Pasta | Conteúdo |
|---|---|
| `01_documentation/` | Planejamento e guia de execução no Colab |
| `02_report/` | Relatório técnico (PDF, formato SBC) e fonte LaTeX |
| `03_code/` | 4 scripts Python + 2 notebooks (treino e avaliação) |
| `04_benchmark_data/` | Benchmark de 100 perguntas, dataset de treino, lista anti-leakage |
| `05_results/` | Métricas, respostas qualitativas, curva de treino, split de teste |
| `06_model_adapters/` | Adapter LoRA destilado |

O relatório técnico completo está em `02_report/relatorio_q4_sbc.pdf`.
