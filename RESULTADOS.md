# Resultados do experimento executado

Rede: Word2Vec + BiLSTM, dimensão 200, semente 123. Treinamento concluído em CPU: 4.94 minutos; 4 épocas executadas; checkpoint selecionado: época 1, pela menor perda de validação. O teste foi avaliado após essa seleção.

| Métrica | Artigo, Tabela 6 | Execução do grupo |
|---|---:|---:|
| Acurácia | 84.57% | 87.27% |
| Precisão IA | 83.15% | 87.57% |
| Recall IA | 86.34% | 86.87% |
| F1 IA | 84.71% | 87.22% |

Diferença observada de acurácia: +2.70 pontos percentuais. Comparação descritiva, não evidência de superioridade: o código recorrente e seus pesos não foram disponibilizados; alguns parâmetros e a divisão específica não são verificáveis. Os valores publicados de precisão/recall/F1 foram transcritos da linha correspondente; a agregação exata da fonte não pôde ser reproduzida. Nenhum resultado publicado foi apresentado como treino local.

## Matriz de confusão (3.000 textos)

| Rótulo da base | Previsto humano | Previsto IA |
|---|---:|---:|
| Humano | 1315 | 185 |
| IA | 197 | 1303 |

Foram 185 humanos classificados como IA e 197 textos de IA classificados como humanos. Essas observações reforçam que a interface não comprova autoria.

## Auditoria e análise complementar

- 185 registros excedentes por repetição depois da limpeza.
- 36 grupos de textos com rótulos conflitantes depois da limpeza.
- 23 textos sem tokens após a limpeza, mantidos como UNK na avaliação para preservar as linhas.
- Sobreposição de hashes únicos: treino/validação 49; treino/teste 28; validação/teste 6. Um mesmo hash pode aparecer em mais de uma interseção.
- Teste sem hashes presentes em treino/validação: 2968 linhas; acurácia 87.33%; F1 IA 87.29%. Foram excluídas 32 linhas, não 34 hashes somados.
- Uma semente e uma base; nenhuma avaliação externa independente. Remover hashes vistos não elimina paráfrases, temas, prompts ou autores repetidos.

## Interpretação da curva

A perda do treino caiu a cada época, mas a melhor perda de validação ocorreu na primeira. A parada antecipada após três épocas sem melhora recuperou esse checkpoint. Não foi usado o último modelo simplesmente por ter treinado por mais tempo.

## Arquivos verificáveis

- `resultados/seed123/historico.json`: perdas e duração de cada época.
- `resultados/seed123/relatorio.json`: métricas calculadas, sem arredondamento para slides.
- `resultados/seed123/predicoes_teste.csv`: resultado por linha, rótulo e pontuação.
- `resultados/seed123/ids_*.csv`: índices e hashes das divisões.
- `resultados/seed123/modelo.pt`: pesos do checkpoint escolhido.
- `resultados/testes.log`: testes de protocolo, rede, serialização e servidor.
- `resultados/teste_interface.json`: verificações no navegador com o modelo real.
- `resultados/ambiente.txt`: versões do ambiente.

Dados originais: https://github.com/shamylafirdoos/Gpt-vs-Human-Text-Classification
Artigo: https://doi.org/10.1038/s41598-025-27377-z
