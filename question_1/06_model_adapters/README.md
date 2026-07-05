# 06 - Modelos e adaptadores

Esta seção contém os adaptadores finais, não o modelo base completo.

## Conteúdo

- `continued_pretraining_final/`: adaptador final do pré-treino continuado.
- `contextual_sft_final/`: adaptador SFT complementar citado no relatório final.

## Como interpretar

Os adaptadores são pesos LoRA/PEFT. Eles precisam ser carregados sobre o modelo base:

```text
Polygl0t/Tucano2-qwen-0.5B-Base
```

Não foram incluídos checkpoints intermediários. Os dois diretórios mantidos são apenas os adaptadores finais necessários para reproduzir o histórico descrito no relatório.
