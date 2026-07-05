# Planejamento de Implementação — Questão 4: Destilação de Conhecimento (CoT)

**Disciplina:** Tópicos em IA (2026.1) — DC/UFPI — Prof. Dr. Raimundo Moura
**Grupo:** Émery Freitas Moriconi, Eryck Kawa Pereira Torres, Felipe Lages de Lima, Raffael Ferreira Fernandes
**Técnica:** Step-by-Step / Chain-of-Thought (CoT) Distillation via SFT + LoRA

---

## 0. O que a questão exige (checklist literal do enunciado)

- [ ] Investigar quais LLMs são normalmente usados para destilação.
- [ ] Definir teacher e student.
- [ ] Gerar um **dataset sintético** (produzido pelo próprio teacher).
- [ ] Fazer a destilação teacher → student.
- [ ] Construir um benchmark de **100 perguntas**.
- [ ] Avaliar teacher **e** student, **antes e depois** da destilação.
- [ ] Analisar se houve (ou não) transferência de conhecimento.

> Observação: "antes e depois" se aplica ao **student**. O teacher é fixo (referência/teto). O baseline obrigatório é o student **sem destilação**.

---

## 1. Decisões já fechadas

| Item | Decisão | Justificativa |
|---|---|---|
| Domínio | CC/UFPI (`docentesDC`) | Consistência com Q2/Q3; benchmark comparável |
| Teacher | `Qwen2.5-7B-Instruct` (via Ollama, Q4_K_M) | Já usado na Q2/Q3 para geração sintética |
| Student | `Qwen2.5-0.5B-Instruct` | Mesma família → mesmo tokenizer, sem estender vocab |
| Técnica | CoT distillation (reasoning + answer) | Slide Vogado: "imitar pensamento" > "imitar decisão" |
| Treino | SFT + LoRA (PEFT) | Viável em T4 16GB; reaproveita know-how da Q3 |
| Ambiente | Colab / Kaggle (T4 16GB) | Exige checkpointing (Seção 9) |
| Benchmark | 100 perguntas CC | Exigência do enunciado |

---

## 2. Por que CoT distillation (e não logit distillation)?

Decisão central a defender na avaliação:

- **Logit distillation** exige acesso à distribuição de probabilidades do teacher. Como teacher e student são da **mesma família (Qwen2.5)**, seria *tecnicamente possível*. Porém:
  - Em Colab, manter teacher 7B + student 0.5B + logits top-k em memória é caro.
  - O slide do Vogado (slide 41) reforça que a tendência prática é **response/CoT distillation** com teachers que você controla.
- **CoT distillation** transfere o **raciocínio explícito** (passos intermediários), não apenas a resposta final. É a forma mais rica de response distillation e a mais alinhada à literatura recente (Distilling Step-by-Step; DeepSeek-R1 destilando para modelos menores com 800k amostras de raciocínio).

**Conclusão:** o sinal supervisionado será `prompt → (reasoning + answer)`, e o student é treinado por next-token prediction (SFT) sobre esse alvo, com LoRA.

---

## 3. Modelos de referência (item "investigar LLMs usados")

Tabela para o relatório (fonte: slides do Vogado + papers):

| Papel | Exemplos citados na literatura |
|---|---|
| Teachers (reasoning) | DeepSeek-R1 (671B+MoE), Qwen3-32B, Gemini, GPT-4o |
| Teachers (soft logits, abertos) | Qwen2.5-7B/32B, Llama |
| Students | Qwen2.5-0.5B/1.5B/3B, BERT-tiny, DistilBERT, Llama-1B/3B |
| Casos reais | BioBERT destilado (clínico); NER de espécies; DeepSeek-R1-Distill-Qwen-1.5B/7B/32B |

---

## 4. Pipeline de dados (dataset sintético CoT)

### 4.1 Fonte
- `vickminari/docentesDC` (HuggingFace) — mesma base da Q2/Q3.

### 4.2 Geração CoT pelo teacher
- Chunking dos documentos (≈600 chars, overlap 50%) — igual à Q2/Q3.
- Para cada chunk, pedir ao teacher (Qwen2.5-7B via Ollama) um par no formato:
```json
{
  "instruction": "pergunta independente de contexto sobre CC",
  "input": "",
  "reasoning": "Passo 1... Passo 2... Passo 3...",
  "answer": "resposta final concisa"
}
```
- Prompt do teacher deve **exigir reasoning explícito** ("explique passo a passo antes de responder").

### 4.3 Filtros de qualidade (reaproveitar `validate_dataset.py` da Q3)
- Descartar reasoning < 30 chars ou answer < 20 chars.
- Descartar perguntas dependentes de contexto ("neste trecho...").
- Deduplicação por hash da instrução.
- Remover pares onde reasoning não conduz à answer (checagem simples de consistência).

### 4.4 Volume e split
- **Alvo:** ~1.000 pares CoT válidos (consistente com Q2/Q3).
- Split: treino 85% / val 10% / teste 5%.
- **Importante:** as 100 perguntas do benchmark NÃO podem estar no treino (data leakage).

### 4.5 Template (Qwen2.5 chat template nativo)
```
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{reasoning}

Resposta: {answer}<|im_end|>
```
- Mascarar tokens do prompt com `-100` (só reasoning+answer contam na loss) — igual à Q2/Q3.

---

## 5. Benchmark de 100 perguntas

### 5.1 Construção
- 100 perguntas de CC cobrindo: algoritmos, estruturas de dados, SO, redes, BD, IA.
- Cada item: `{id, pergunta, resposta_referencia, categoria}`.
- Geradas a partir de chunks **não usados no treino**; revisão manual leve.

### 5.2 Métricas (alinhar com o que o grupo já usa)
**Quantitativas:**
- Perplexidade no conjunto de teste (antes/depois) — métrica central, igual Q2/Q3.
- Acurácia de resposta: comparação semântica answer vs referência (ROUGE-L + checagem de presença de termos-chave; opcionalmente LLM-as-judge usando o próprio teacher).
- Taxa de respostas em PT correto (o baseline 0.5B tende a misturar idioma, como visto na Q2).

**Qualitativas:**
- 5 perguntas-âncora respondidas por: teacher, student-base, student-destilado (tabela comparativa, igual Q3).

### 5.3 As 5 variantes a avaliar
1. Teacher (Qwen2.5-7B) — teto de referência
2. Student base (Qwen2.5-0.5B sem treino) — baseline obrigatório
3. Student + SFT comum (sem reasoning) — *opcional, fortalece a análise*
4. **Student + CoT distillation (LoRA)** — variante principal
5. (Opcional) Student destilado + quantização — trade-off de deploy

> Sem o baseline (#2) não dá para afirmar "houve transferência de conhecimento". Ele é o eixo da conclusão.

---

## 6. Hiperparâmetros (ponto de partida)

| Item | Valor | Observação |
|---|---|---|
| Modelo base student | Qwen2.5-0.5B-Instruct | |
| Camadas-alvo LoRA | q_proj, k_proj, v_proj, o_proj, gate/up/down_proj | padrão Qwen |
| Rank (r) | 16 | igual Q3; r=8 se faltar memória |
| Alpha | 32 | escala 2× |
| Dropout LoRA | 0.05 | |
| Épocas | 3 | early stopping na val |
| Batch efetivo | 8–16 | batch × grad. accumulation |
| LR | 2e-4 | padrão LoRA |
| Max seq len | 512 | CoT é mais longo; medir distribuição |
| Precisão | bf16 (T4 → fp16) | |
| Loss | CrossEntropy (next-token) sobre reasoning+answer | |

---

## 7. Avaliação justa (controlar tudo)

- Mesmo conjunto de teste e mesmo benchmark para todas as variantes.
- Mesma seed (42), mesmo template de prompt.
- Geração **determinística** (greedy, `do_sample=False`) para reprodutibilidade.
- `max_new_tokens`, `repetition_penalty` fixos entre variantes.
- Mesmo hardware para medir latência/VRAM.

---

## 8. Critério de sucesso (definir antes de rodar)

> Houve transferência de conhecimento se o **student destilado** reduzir a perplexidade vs. o student base **e** aproximar-se do teacher nas métricas de qualidade, mantendo o porte de 0.5B.

Meta operacional sugerida:
- Redução de PPL ≥ 30% vs student base.
- Acurácia de resposta no benchmark superior à do student base por margem clara.
- Eliminação do problema de idioma errado (como ocorreu na Q2).

---

## 9. Checkpoints, retomada e persistência (Colab/Kaggle)

Crítico porque sessões Colab/Kaggle caem. Estratégia:

- **Geração do dataset:** salvar incrementalmente (cada N pares → arquivo JSON + hash set), permitir retomada (já existe na Q2/Q3).
- **Treino:** HuggingFace `Trainer` com:
  - `output_dir` em pasta sincronizada;
  - `save_strategy="steps"`, `save_steps=50`, `save_total_limit=2`;
  - `load_best_model_at_end=True`, métrica = val_loss;
  - retomada via `trainer.train(resume_from_checkpoint=True)`.
- **Google Drive:** montar e salvar checkpoints importantes lá; treinar em disco local da sessão e **sincronizar periodicamente** (a doc do Colab recomenda minimizar I/O direto no Drive).
- **Adapters LoRA:** salvar só o adapter (`best_adapter/`) — leve (MBs), fácil de versionar.
- **Cache:** salvar dataset tokenizado e outputs do teacher para não regerar.
- **Teste de retomada obrigatório:** matar a sessão de propósito uma vez e confirmar que `resume_from_checkpoint` funciona antes do treino longo.

---

## 10. Entregáveis finais

1. `gerar_dataset_cot.py` — geração sintética CoT via Ollama/Qwen2.5-7B
2. `validar_dataset_cot.py` — filtros de qualidade
3. `dataset_cot_docentesDC.json` — ~1.000 pares (instruction/input/reasoning/answer)
4. `benchmark_q4.json` — 100 perguntas + referências
5. `destilacao_cot.ipynb` — preparação, treino LoRA, curvas
6. `avaliacao_q4.ipynb` — perplexidade, métricas de resposta, comparação qualitativa
7. `student_cot_adapter/` — adapter LoRA destilado
8. `q4_metrics.json` + `q4_qualitative.json` — resultados
9. `relatorio_q4.pdf` — relatório técnico (mesmo padrão da Q2/Q3/Q6)

---

## 11. Ordem de execução (anti-desperdício)

1. Sanity check com **20 pares** e **5 perguntas** — pipeline ponta a ponta.
2. Geração de ~200 pares + treino curto — validar curvas.
3. Geração completa (~1.000) + benchmark 100 + treino final.
4. Avaliação das variantes + relatório.

---

## 12. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Teacher imperfeito (alucina labels) | Filtro de qualidade + consistência reasoning↔answer (slide 42-43) |
| Student 0.5B repetitivo (visto na Q3) | CoT ajuda; controlar repetition_penalty; dataset limpo |
| Data leakage benchmark↔treino | Gerar benchmark de chunks separados; checar interseção |
| Sessão Colab cai | Checkpointing + resume (Seção 9) |
| CoT estoura 512 tokens | Medir distribuição; truncar reasoning longo |
| "Acurácia" subjetiva | Combinar ROUGE-L + termos-chave + LLM-as-judge |

---

## 13. Contribuição de Ciência da Computação (defesa)

A questão não é "treinar um modelo", e sim demonstrar **engenharia de compressão de modelos**: transferir capacidade de um modelo 14× maior (7B → 0.5B) para viabilizar inferência local/barata, quantificando o trade-off qualidade × custo (PPL, latência, VRAM) — exatamente o eixo "performance vs eficiência" do slide do Vogado.
