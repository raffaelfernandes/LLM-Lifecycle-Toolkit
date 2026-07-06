# LLM Lifecycle Toolkit

Repositório consolidado das entregas do trabalho final de Tópicos em IA / Desenvolvimento de IA. O projeto cobre as seis questões do enunciado, com experimentos de pré-treino, pós-treino, destilação, RAG e guardrails.

## Visão geral das questões

| Questão | Tema | Pasta principal | Resumo |
| --- | --- | --- | --- |
| 1 | Pré-treino continuado | `question_1/` | Continuação de treinamento de um LLM com o corpus DOMPI-2025, com benchmark, métricas antes/depois, documentação, código e adaptadores finais. |
| 2 | Pós-treino com SFT | `questions_02-03/` | Geração de dataset sintético, fine-tuning supervisionado e avaliação do modelo base vs. modelo ajustado. |
| 3 | Pós-treino com LoRA/QLoRA | `questions_02-03/` | Repetição do experimento anterior com adaptadores eficientes, comparando qualidade, tempo e uso de VRAM. |
| 4 | Destilação de conhecimento | `question_4/` | Pipeline de destilação CoT/teacher-student, com benchmark, métricas e análise de transferência de conhecimento. |
| 5 | RAG | `questao5/` | Aplicação de RAG self-reflective com Qdrant, benchmark de 20 perguntas e comparação entre cenário sem RAG e com RAG. |
| 6 | Guardrails | `question_6/` | Camadas de proteção de entrada e saída sobre um LLM local via Ollama, com benchmark de validação. |

## Estrutura do repositório

- `question_1/`: documentação, código, benchmark, resultados e adaptadores do pré-treino continuado.
- `questions_02-03/`: relatórios, notebooks, scripts, modelos e métricas das questões 2 e 3.
- `question_4/`: materiais da questão de destilação de conhecimento.
- `questao5/`: notebooks e artefatos do RAG, incluindo base vetorial local e comparações de desempenho.
- `question_6/`: notebook e benchmark da implementação de guardrails.
- `qdrant_storage/`: estado local do banco vetorial usado na questão 5.
- `TrabalhoFinal.pdf`: versão consolidada do trabalho final.

## Destaques técnicos

- Questão 1: continuação de pré-treino com avaliação por perplexidade, entropia cruzada e acurácia de tokens.
- Questões 2 e 3: geração sintética de pares instrução-resposta a partir de `docentesDC`, com SFT, LoRA e QLoRA.
- Questão 4: distilação de conhecimento em formato teacher-student, com benchmark próprio e comparação entre baseline e aluno destilado.
- Questão 5: RAG local com embeddings, Qdrant e um fluxo self-reflective para decidir quando responder com ou sem contexto recuperado.
- Questão 6: guardrails de entrada e saída para bloquear ou sinalizar conteúdo indevido antes de expor a resposta ao usuário.

## Observação

Os arquivos foram organizados para entrega acadêmica e auditoria. Caches, ambientes virtuais, corpus bruto completo e checkpoints intermediários não foram copiados para evitar peso desnecessário no repositório.
