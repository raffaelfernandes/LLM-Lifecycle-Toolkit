# Questão 4 — Destilação de Conhecimento (Step-by-Step / CoT)

Esta pasta contém a entrega final da quarta questão do trabalho final de Tópicos
em Inteligência Artificial (2026.1 — DC/UFPI): a **destilação de conhecimento** de
um modelo professor **Qwen2.5-7B-Instruct** para um modelo aluno
**Qwen2.5-0.5B-Instruct**, uma redução de aproximadamente 14× em parâmetros.

## Visão geral

A técnica adotada foi a **destilação Step-by-Step (Chain-of-Thought)**: o aluno é
treinado, via SFT com adaptadores LoRA, para imitar o *raciocínio* e a *resposta*
gerados pelo professor. O dataset foi gerado sinteticamente pelo próprio professor
a partir do corpus `docentesDC`, e a transferência de conhecimento foi avaliada
sobre um benchmark de 100 perguntas de Ciência da Computação, comparando o aluno
antes e depois da destilação.

## Resultado principal

Houve **transferência de conhecimento**. Sobre o benchmark de 100 perguntas:

| Variante | PPL | ROUGE-L | Cobertura | % PT |
|---|---|---|---|---|
| Teacher 7B (referência, 4-bit) | 5,85 | 0,212 | 0,675 | 100% |
| Student 0.5B (base, antes) | 6,52 | 0,175 | 0,559 | 100% |
| **Student 0.5B (destilado, depois)** | **4,52** | **0,226** | 0,409 | 100% |

A perplexidade do aluno caiu **30,7%** (6,52 → 4,52) e o ROUGE-L subiu de 0,175
para 0,226. A análise mostra que a destilação CoT transferiu com sucesso o
comportamento de raciocínio e o estilo do professor, enquanto a transferência de
conhecimento factual foi parcial, limitada pela capacidade do aluno de 0.5B.

## Estrutura de pastas

```
question_4/
├── 01_documentation/     Planejamento e guia de execução
├── 02_report/            Relatório técnico (PDF + fonte LaTeX)
├── 03_code/              Scripts de geração/validação e notebooks
├── 04_benchmark_data/    Benchmark de 100 perguntas e dataset de treino
├── 05_results/           Métricas, resultados qualitativos e curva de treino
├── 06_model_adapters/    Adapter LoRA destilado (student_cot_adapter)
├── README.md
└── final_delivery.md
```

## Pipeline (ordem de execução)

O experimento seguiu cinco etapas encadeadas:

1. **Geração do dataset** — `03_code/gerar_dataset_cot.py` chama o professor
   Qwen2.5-7B via Ollama e produz pares no formato
   `{instruction, input, reasoning, answer}`. Executado localmente.
2. **Validação e limpeza** — `validar_dataset_cot.py` e
   `limpar_contexto_reasoning.py` filtram ruído e reescrevem raciocínios com
   referências ao contexto de origem. Resultado: **1.077 pares CoT válidos**.
3. **Benchmark** — `montar_benchmark_q4.py` extrai 100 perguntas de forma
   estratificada, com prevenção de vazamento (exclusão exata + quase-duplicatas
   com Jaccard > 0,7). Zero sobreposição conceitual com o treino.
4. **Treino (destilação)** — `destilacao_cot_COLAB.ipynb` treina o adapter LoRA
   sobre o aluno de 0.5B. Executado no Google Colab (GPU T4).
5. **Avaliação** — `avaliacao_q4_COLAB.ipynb` compara base, destilado e professor,
   emitindo o veredito de transferência de conhecimento.

## Como reproduzir

**Etapa local (geração):**
```bash
pip install ollama datasets
ollama pull qwen2.5:7b-instruct-q4_K_M
python 03_code/gerar_dataset_cot.py --n-alvo 1400 --saida dataset_cot_docentesDC.json
python 03_code/limpar_contexto_reasoning.py --entrada dataset_cot_docentesDC.json --saida dataset_cot_reasoning_limpo.json
python 03_code/validar_dataset_cot.py --entrada dataset_cot_reasoning_limpo.json --saida dataset_cot_limpo.json
python 03_code/montar_benchmark_q4.py --fonte sft_dataset_docentesDC.json --treino dataset_cot_limpo.json
```

**Etapa Colab (treino e avaliação):**
Abrir `03_code/destilacao_cot_COLAB.ipynb` no Google Colab (runtime T4),
seguir o guia em `01_documentation/GUIA_COLAB_treino.md`, e depois rodar
`03_code/avaliacao_q4_COLAB.ipynb`.

## Ambiente

- **Geração do dataset:** notebook Asus TUF Gaming F16 — Intel Core i7 (13ª geração),
  NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM), 16 GB RAM. Professor executado via
  Ollama (Qwen2.5-7B-Instruct, quantização Q4_K_M).
- **Treino e avaliação:** Google Colab, GPU Tesla T4 (16 GB). Professor avaliado
  com quantização NF4 (4-bit) para caber na T4.

Todos os experimentos foram executados sem dependência de APIs pagas de terceiros.
