# 03_code — Código

Scripts e notebooks que implementam o pipeline de destilação.

| Arquivo | Descrição |
|---|---|
| `gerar_dataset_cot.py` | Gera o dataset sintético CoT chamando o professor Qwen2.5-7B via Ollama. Inclui pré-filtro de qualidade e checkpoint retomável. |
| `validar_dataset_cot.py` | Aplica filtros de qualidade (idioma, consistência, deduplicação, foco em CC). |
| `limpar_contexto_reasoning.py` | Reescreve raciocínios que referenciam o contexto de origem, recuperando pares que seriam descartados. |
| `montar_benchmark_q4.py` | Extrai as 100 perguntas do benchmark de forma estratificada, com prevenção de vazamento (exclusão exata + quase-duplicatas). |
| `destilacao_cot_COLAB.ipynb` | Notebook de treino: preparação, LoRA, loop de destilação e salvamento do adapter. Executado no Colab (T4). |
| `avaliacao_q4_COLAB.ipynb` | Notebook de avaliação: compara teacher, student base e student destilado; emite o veredito de transferência. |

**Ordem de execução:** `gerar_dataset_cot` → `limpar_contexto_reasoning` →
`validar_dataset_cot` → `montar_benchmark_q4` → `destilacao_cot_COLAB` →
`avaliacao_q4_COLAB`.
