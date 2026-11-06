# Roteiro da apresentação — seis integrantes



Divisão: João (1–4), Leonardo (5–6), Gabriel (7–8), Gustavo (9–11), Felipe (12 e demonstração), Alexandre (13–14).

Tempo sugerido: 15–20 minutos, incluindo a demonstração. A composição de seis foi mantida conforme solicitado; o limite do enunciado é cinco.



## Slide 1 — Capa

**João Vitor Frare**

Apresente os seis integrantes, a disciplina e o objetivo: reconstruir um experimento neural do artigo. Delimite o trabalho à BiLSTM, sem afirmar reprodução integral.



## Slide 2 — O artigo que orienta o experimento

**João Vitor Frare**

Identifique o artigo, ano, periódico e autores. Ele compara modelos clássicos, recorrentes e transformers. Nosso recorte é a linha Word2Vec + BiLSTM, semente 123 e dimensão 200 da Tabela 6.



## Slide 3 — O problema e o objetivo

**João Vitor Frare**

Explique a relevância da identificação de padrões de escrita. Os rótulos são os da base. A meta é estudar reprodução de um método neural, não oferecer prova de autoria nem um detector universal.



## Slide 4 — Da metodologia original ao nosso recorte

**João Vitor Frare**

O artigo inclui redes recorrentes e transformers. Mantemos base e hiperparâmetros publicados da BiLSTM. O código recorrente não está no repositório; unidades, pesos Word2Vec e alguns detalhes foram reconstruídos. A implementação não é uma cópia exata.



## Slide 5 — Os dados são os mesmos do repositório

**Leonardo Forgiarini**

Mostre dados/shuffled_data.csv. São 20.000 linhas, 10.000 por classe, em inglês. O checksum garante a versão. A divisão 70/15/15 segue o notebook público, mas os índices específicos do experimento recorrente não foram publicados.



## Slide 6 — Preparação: do texto aos vetores

**Leonardo Forgiarini**

Explique que a limpeza segue os passos descritos no artigo. O vocabulário e o Word2Vec são ajustados só no treino para não aprender com o teste. Cada palavra vira um vetor de 200 números; isso não é TF-IDF. Textos longos são truncados após a limpeza.



## Slide 7 — A arquitetura neural implementada

**Gabriel Silva**

Embedding: 200 valores por token. Uma camada bidirecional com 64 unidades em cada sentido. As portas da LSTM usam sigmoid, o estado candidato usa tanh. Concatenação dos estados finais: 128 valores. Linear 128 para 2 logits; softmax na inferência. As 64 unidades e a cabeça linear são decisões locais, pois os autores não especificaram esses detalhes.



## Slide 8 — Como o modelo aprendeu

**Gabriel Silva**

Apresente os parâmetros publicados. Adam ajusta pesos para reduzir entropia cruzada. A curva revela sobreajuste: a perda do treino continua caindo e a validação piora. Recuperamos a época 1 e paramos após três épocas sem melhora. O teste só foi executado depois da escolha.



## Slide 9 — Resultado real nos 3.000 textos de teste

**Gustavo Pigatto**

Explique a matriz: linhas são rótulos da base e colunas as previsões. Falsos positivos são humanos classificados como IA; falsos negativos são IA classificada como humana. A acurácia sozinha não mostra os dois tipos de erro.



## Slide 10 — Artigo × execução do grupo

**Gustavo Pigatto**

Compare apenas a linha Word2Vec + BiLSTM, dimensão 200 e semente 123. A diferença de acurácia é descritiva: não houve teste pareado com previsões dos autores, e os detalhes da rede/embeddings/divisão diferem. Não dizer que superamos cientificamente o artigo. A agregação exata de P/R/F1 da fonte não foi reconstruída.



## Slide 11 — Auditoria: o que limita a comparação

**Gustavo Pigatto**

Há 185 linhas excedentes por repetição depois da limpeza e 36 grupos com rótulos conflitantes. Mantivemos as linhas no protocolo principal e auditamos a sobreposição. O teste adicional remove 32 linhas cujos hashes aparecem em treino/validação. Isso não remove paráfrases ou autores repetidos. Uma execução não mede variação entre sementes.



## Slide 12 — Demonstração da rede em funcionamento

**Felipe Lucca**

Abra Iniciar_Interface.bat. Mostre um exemplo rotulado da validação e clique em analisar. Explique: interface envia o texto; Predictor faz limpeza e codificação; BiLSTM.forward calcula logits; softmax gera pontuação; navegador mostra a classe. Consultas não treinam a rede. Os exemplos são escolhidos sem usar seus acertos. O código da rede está em modelo.py, e o treino em treinar.py.



## Slide 13 — Dificuldades e decisões de implementação

**Alexandre Lampert**

Explique as lacunas do repositório e por que 64 unidades é uma escolha local. Os pesos dos transformers não puderam ser baixados. O escopo recorrente foi viável em CPU. Comente o sobreajuste e a parada antecipada. Próximos passos devem ser apresentados como futuros, sem insinuar que já foram executados.



## Slide 14 — O que conseguimos reproduzir

**Alexandre Lampert**

Conclua: rede neural implementada e treinada, dados originais, parâmetros publicados, comparação e interface. Os números locais são reais, mas o experimento é uma reconstrução parcial. A composição de seis integrantes foi mantida por solicitação, embora o roteiro diga até cinco. Isso consta na documentação, não foi tratado como autorização do professor.



## Perguntas prováveis



**É uma rede neural?** Sim: embedding ajustável, camada LSTM bidirecional e camada linear treinadas por retropropagação.

**Por que não RoBERTa?** O artigo também avalia redes recorrentes. O recorte escolhido foi Word2Vec + BiLSTM; os downloads de pesos transformer estavam bloqueados.

**É reprodução exata?** Não. É uma reconstrução parcial com base e parâmetros publicados; faltam detalhes recorrentes dos autores, explicitados em METODOLOGIA.md.

**Funciona em português?** Não há validação em português. A interface está em português, mas o modelo e os exemplos são em inglês.

**O resultado melhor prova superioridade?** Não: protocolos e detalhes diferem, há uma única semente e faltam previsões originais para comparação pareada.

**Por que parou na época 4?** A perda de validação não melhorou por três épocas; recuperou o checkpoint da época 1.

**Por que existem duplicatas?** Preservamos a base para documentar a reconstrução. Apresentamos auditoria e um teste complementar sem hashes vistos. Isso não prova independência de fontes ou autores.

**O que significam as ativações?** Sigmoid controla portas da LSTM; tanh limita os estados; softmax transforma os dois logits em pontuações normalizadas.

**A interface aprende com as consultas?** Não. Ela usa pesos salvos em modo de inferência e não armazena o texto.

