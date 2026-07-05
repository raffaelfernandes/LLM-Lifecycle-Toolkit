# Guia de Execução — Treino no Google Colab (Etapa 2)

## Antes de começar
1. Acesse https://colab.research.google.com
2. **Runtime → Change runtime type → T4 GPU** (obrigatório!)
3. Faça upload do `destilacao_cot.ipynb` (File → Upload notebook)

---

## Passo 1 — Cole ESTA célula no TOPO do notebook (antes de tudo)

Ela monta o Drive, instala as libs certas e cria a pasta de trabalho.
Substitui a necessidade de descomentar o pip da célula de setup.

```python
# ===== SETUP COLAB (rodar primeiro) =====
# 1. Monta o Google Drive (checkpoints sobrevivem se a sessão cair)
from google.colab import drive
drive.mount('/content/drive')

# 2. Cria pasta de trabalho no Drive
import os
WORK = '/content/drive/MyDrive/Q4_destilacao'
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print('Pasta de trabalho:', os.getcwd())

# 3. Instala dependências (SEM pin de versão — pega o compatível com Qwen2.5)
!pip install -q -U transformers peft datasets accelerate
print('Dependências instaladas')

# 4. Confirma GPU
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA — ative T4!')
```

---

## Passo 2 — Suba os dados PARA A PASTA DO DRIVE

Depois de rodar a célula acima (que cria a pasta), faça upload destes 2 arquivos
para `/content/drive/MyDrive/Q4_destilacao/` (ícone de pasta à esquerda → navegue
até a pasta → upload):

- `dataset_cot_limpo.json`   (seus 1.077 pares de treino)
- `excluir_do_treino.json`   (proteção anti-leakage)

> Dica: subir para o Drive uma vez evita reupload se a sessão cair.
> Alternativa: copie os arquivos manualmente pelo Drive web para essa pasta.

---

## Passo 3 — Ajuste a célula de SETUP original do notebook

A célula 2 (que começa com `# Em Colab/Kaggle, descomente:`) tem um `import torch`
que já foi feito. **Pode rodá-la como está** — não precisa descomentar o pip dela,
porque já instalamos no Passo 1. Se der erro de import, rode-a mesmo assim.

---

## Passo 4 — (Opcional) Aponte checkpoints para o Drive

Na célula de CONFIG (célula 4), o `OUTPUT_DIR` já grava na pasta atual, que agora
é o Drive (por causa do `os.chdir` no Passo 1). Então os checkpoints já vão para
o Drive automaticamente. Nada a mudar.

---

## Passo 5 — Rode as células na ordem

Execute célula por célula (Shift+Enter), de cima para baixo. A sequência é:

1. Setup Colab (Passo 1) ← a que você colou
2. Setup original + seed
3. Config (STUDENT_ID, hiperparâmetros)
4. Carga do dataset
5. **Anti-leakage** (remove as 100 do benchmark do treino)
6. Split 85/10/5
7. Tokenizer + template (confira o % de truncamento que aparece!)
8. Dataset + collator
9. Modelo + LoRA (baixa o Qwen2.5-0.5B — ~1 GB)
10. **Treino** (a demorada — ~30-60 min para 1.077 pares × 3 épocas na T4)
11. Curvas
12. Salvar adapter
13. Teste qualitativo rápido

---

## O que observar durante o treino

- **Célula 7 (tokenizer):** aparece o `% truncados em 512`. Se for alto (>10%),
  me avise — pode valer subir MAX_LEN ou apertar o reasoning.
- **Célula 10 (treino):** acompanhe `train_loss` e `eval_loss` caindo. Se
  `eval_loss` começar a SUBIR enquanto train cai, é overfitting (o early stopping
  trata, mas é bom notar).
- **Se a sessão cair:** reabra, rode o Passo 1 de novo (remonta o Drive), e
  execute até a célula de treino — ela detecta o checkpoint no Drive e retoma.

---

## Se der erro de falta de memória (OOM) — improvável no 0.5B, mas...

Na célula de CONFIG, reduza:
- `BATCH = 1` (e aumente `GRAD_ACCUM = 16` para manter batch efetivo)
- ou `MAX_LEN = 384`

---

## Ao terminar

O treino gera:
- `student_cot_adapter/`  → o adapter LoRA destilado (no Drive)
- `split_teste.json`      → conjunto de teste (usado na avaliação)
- `q4_curva_treino.png`   → curva para o relatório

Baixe esses do Drive. Depois seguimos para a **Etapa 3 — avaliação**
(`avaliacao_q4.ipynb`), que compara base vs. destilado e responde se houve
transferência de conhecimento.
```
