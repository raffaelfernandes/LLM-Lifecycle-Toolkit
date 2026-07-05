# 03 - Codigo

Scripts usados para preparar dados, gerar benchmarks, treinar e avaliar a entrega final da Questao 1.

## Observacao sobre caminhos

Os scripts preservam a estrutura original de execucao usada no experimento, com pasta `llm_pretraining/`. Nesta entrega, os dados brutos, corpus completo, caches e checkpoints intermediarios foram omitidos por curadoria. Portanto, a pasta `03_code/` serve como evidencia do pipeline e base de reproducao; para reexecutar tudo, recrie a estrutura original ou ajuste os argumentos `--data-dir` e `--output-dir`.

## Fluxo final

1. `prepare_dompi_2025.py`: baixa/le o DOMPI-2025 e gera extracoes limpas.
2. `prepare_tucano2_dompi_10k_experiment.py`: cria a divisao final de 10k treino, 1k validacao, 1k teste e benchmark reservado.
3. `train_continued_pretraining_gpu.py`: executa o pre-treino continuado com `Polygl0t/Tucano2-qwen-0.5B-Base`.
4. `generate_contextual_benchmark_25_v2.py` e `generate_specific_benchmark_25_v1.py`: geram benchmarks de avaliacao.
5. `audit_contextual_benchmark.py`: audita perguntas e contextos.
6. `evaluate_base_model_dompi_2025.py` e `evaluate_benchmark_25_v2.py`: avaliam metricas antes/depois e comparacoes com SFT.

## Script de execucao principal

- `setup_gpu_windows.ps1`: preparacao do ambiente em Windows com GPU.
- `run_tucano2_dompi_10k_loraplus.ps1`: execucao do experimento final Tucano2 + DOMPI-2025 + LoRA+.

Os scripts pressupõem que Python, PyTorch/CUDA, Hugging Face Transformers, PEFT e dependencias de leitura de dados estejam instalados.
