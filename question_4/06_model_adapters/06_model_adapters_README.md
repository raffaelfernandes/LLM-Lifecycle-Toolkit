# 06_model_adapters — Adaptadores do Modelo

Adapter LoRA resultante da destilação.

| Item | Descrição |
|---|---|
| `student_cot_adapter/` | Adapter LoRA destilado, treinado sobre o Qwen2.5-0.5B-Instruct. Corresponde ao melhor checkpoint de validação (early stopping). |

Para usar o adapter, carregue o modelo base `Qwen/Qwen2.5-0.5B-Instruct` e aplique
o adapter com a biblioteca PEFT (`PeftModel.from_pretrained`). O procedimento
completo está no notebook `03_code/avaliacao_q4_COLAB.ipynb`.
