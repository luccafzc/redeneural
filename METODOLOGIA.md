# Reconstrução de Word2Vec + BiLSTM

Artigo: Masih et al. (2025), *Classifying human vs. AI text with machine learning and explainable transformer models*, Scientific Reports 15, 43310. DOI: https://doi.org/10.1038/s41598-025-27377-z.

O escopo é um experimento recorrente do artigo: **Word2Vec + BiLSTM, dimensão 200, semente 123** (Tabela 6). Esta entrega não reproduz a bateria completa, RoBERTa, LIME, SHAP, pruning ou calibração dos transformers. A rede é realmente treinada com retropropagação; a interface carrega seus pesos.

## Proveniência e dados

- Base original dos autores: 20.000 registros, 10.000 humanos (0) e 10.000 IA (1), em inglês.
- Repositório: https://github.com/shamylafirdoos/Gpt-vs-Human-Text-Classification
- Revisão: `82bf5888efba65bc9aed64f6b02f198d493f806d`.
- SHA-256 do CSV: `f9ab6ff46e38f4930fd3375e6ca5a56134c064855c029e5ed2ad08b3fa9e5c6b`.
- São preservadas as 20.000 linhas. Aplicam-se as divisões estratificadas 70/15/15 usadas no notebook dos autores, com semente 123. Esse notebook contém transformers; não se afirma que seus índices sejam os índices do experimento recorrente publicado.
- Os manifestos registram índices de linha (base zero), hashes e rótulos de cada divisão.
- Duplicatas e conflitos são auditados, mas mantidos no protocolo principal para não alterar silenciosamente a base. O relatório adicional exclui do teste hashes de textos pré-processados presentes no treino/validação. Isso não elimina paráfrases nem garante separação por autor, tema ou prompt.
- Os dados vieram do repositório com licença MIT; a procedência e os termos das fontes que compõem a base devem ser citados conforme o artigo. Não foram traduzidos ou misturados à PT-L2Detect.

## Parâmetros publicados e decisões de reconstrução

| Componente | Artigo/repositório | Implementação |
|---|---|---|
| Rede | BiLSTM | BiLSTM |
| Representação | Word2Vec, dimensão 200 | Word2Vec, dimensão 200 |
| Semente da linha comparada | 123 | 123 |
| Taxa de aprendizado | 0,001 | 0,001 |
| Lote | 32 | 32 |
| Sequência máxima | 200 | 200 tokens após limpeza |
| Otimizador e perda | Adam; CrossEntropyLoss | Adam; CrossEntropyLoss |
| Épocas e parada | até 30; paciência 3 | até 30; paciência 3; monitorar perda de validação |
| Camadas/unidades recorrentes | Não especificadas nas informações disponibilizadas | 1 camada; 64 unidades por direção — escolha local |
| Dropout recorrente | Não especificado | 0 — escolha local, não atribuir ao artigo |
| Classificação | Binária | Concatenar últimos estados → Linear 128×2; softmax na inferência |
| Embeddings originais | Sem arquivo de pesos/receita completa para a linha recorrente | Skip-gram ajustado só no treino: janela 5, 5 épocas, 5 negativos, min_count 2, um worker; depois ajustável pela BiLSTM |
| Vocabulário | Sem limite especificado para esta linha | máximo 30.000, frequência mínima 2; PAD e UNK reservados |
| Plataforma | Keras/TensorFlow para recorrentes; Kaggle | PyTorch 2.5.1, CPU; escolha de implementação |
| Divisão recorrente exata | Sem índices disponibilizados | 70/15/15 estratificado conforme notebook público |
| Experimentos | Vários modelos, dimensões e sementes | Uma configuração, definida antes do teste |

O cabeçalho de dimensão/semente da tabela online aparece em ordem inconsistente com as linhas. A seleção 123/200 foi confirmada pelo texto da seção de resultados do artigo. A linha Word2Vec correspondente publica acurácia 0,8457, precisão 0,8315, recall 0,8634 e F1 0,8471. As métricas são referências publicadas, não números produzidos por esta execução.

## Arquitetura e aprendizado

Cada texto passa por minúsculas, remoção de pontuação ASCII e stopwords inglesas do NLTK, e tokenização por espaços. A ordem exata da remoção de pontuação é uma escolha documentada, pois o código recorrente dos autores não foi publicado. O vocabulário e o Word2Vec usam somente o treino. A sequência é truncada em 200 tokens. Textos sem tokens após a limpeza são representados por UNK na avaliação, preservando as linhas; a interface solicita outro texto nesse caso.

O embedding converte IDs em vetores de 200 números. Duas LSTMs percorrem a sequência em sentidos opostos, cada uma com 64 unidades. As portas usam sigmoid, e o estado candidato e a saída usam tanh. Os últimos estados das duas direções formam um vetor de 128 números. A camada linear produz dois logits. CrossEntropyLoss recebe logits durante o treino; softmax só transforma os logits em pontuações para avaliação/interface.

`pack_padded_sequence` impede que o preenchimento seja tratado como palavras. Os embeddings começam com Word2Vec treinado no corpus de treino e continuam ajustáveis. Os gradientes atualizam os pesos via Adam. A menor perda de validação define o checkpoint salvo; três épocas sem melhora encerram o ajuste. Somente então o checkpoint escolhido é avaliado no teste.

## Comparação e limites

As diferenças entre a execução e o artigo podem vir dos pesos Word2Vec, arquitetura não especificada, processamento, divisão, inicialização e implementação. A comparação é descritiva; não demonstra superioridade estatística nem reprodução exata. Uma única semente não permite estimar variabilidade entre sementes. Os números do teste sem hashes vistos são uma análise complementar, não outro benchmark original.

O modelo trabalha com os primeiros 200 tokens após limpeza e foi estudado em inglês. Não há validação de detecção geral nem garantia de autoria. A interface informa truncamento e fração de palavras desconhecidas. As pontuações softmax não são calibradas. Consultas não modificam os pesos nem são armazenadas.

## Limitações da fonte e da execução

O repositório público contém um notebook de transformers e o CSV, mas não o código recorrente, seus pesos ou todos os parâmetros. Não se completaram essas lacunas com afirmações inventadas sobre os autores. O download de pesos do Hugging Face foi negado pela rede deste ambiente; a reconstrução recorrente foi escolhida dentro das famílias avaliadas pelo artigo. A versão original do artigo está referenciada por DOI e link de PDF. O download binário do PDF também foi bloqueado neste ambiente; o atalho não é uma cópia local do PDF.
