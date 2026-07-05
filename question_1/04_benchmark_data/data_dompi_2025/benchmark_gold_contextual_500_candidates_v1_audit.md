# Auditoria programatica do benchmark contextual

Arquivo auditado: `llm_pretraining\data_dompi_2025\benchmark_gold_contextual_500_candidatos_v1.jsonl`
Total de itens: 500
Aprovados: 352
Reprovados: 148

## Criterios

- rubrica minima ancorada no contexto: 80%
- tokens relevantes do gabarito presentes no contexto: 55%
- contexto minimo: 500 caracteres
- resposta entre 20 e 520 caracteres
- sem documento duplicado
- sem par pergunta/gabarito duplicado

## Temas aprovados

- relatorio_fiscal: 91
- contratacao: 99
- ato_normativo: 121
- portaria: 37
- licitacao: 3
- rescisao: 1

## Motivos de reprovacao

- rubrica_pouco_ancorada: 146
- gabarito_pouco_ancorado: 51
- pergunta_gabarito_duplicado: 1

## Exemplos aprovados

- `auditado_q001` (relatorio_fiscal): Considerando o trecho, qual demonstrativo fiscal aparece no documento?
- `auditado_q002` (contratacao): Considerando o trecho, que contratacao foi descrita no documento?
- `auditado_q003` (contratacao): Considerando o trecho, que contratacao foi descrita no documento?
- `auditado_q004` (ato_normativo): Considerando o trecho, o que o ato normativo estabelece?
- `auditado_q005` (ato_normativo): Considerando o trecho, o que o ato normativo estabelece?
- `auditado_q006` (ato_normativo): Considerando o trecho, o que o ato normativo estabelece?
- `auditado_q007` (portaria): Considerando o trecho, o que ocorreu na portaria?
- `auditado_q008` (ato_normativo): Considerando o trecho, o que o ato normativo estabelece?
- `auditado_q009` (relatorio_fiscal): Considerando o trecho, qual demonstrativo fiscal aparece no documento?
- `auditado_q010` (contratacao): Considerando o trecho, que contratacao foi descrita no documento?
