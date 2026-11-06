# Correspondência com o roteiro da disciplina

| Exigência | Evidência e limite |
|---|---|
| Identificar artigo, autores, ano e publicação | Slide 2; fontes/referencia_artigo.json; DOI e atalho para PDF original |
| Problema e objetivos | Slides 3 e 4 |
| Metodologia original | Slides 4 a 8; METODOLOGIA.md |
| Arquitetura neural, camadas e ativações | Slide 7; modelo.py; embedding 200 → BiLSTM 64+64 → linear 128×2, sigmoid/tanh e softmax |
| Mesma base quando disponível | CSV dos autores com revisão e SHA-256 conferidos; slide 5 |
| Parâmetros publicados | config.json; slide 8; Tabela 4 do artigo |
| Treinar/testar e registrar resultados | treinar.py; pesos, histórico, previsões, IDs, auditoria e métricas em resultados/seed123 |
| Comparação com experimento original | Slide 10; RESULTADOS.md; linha Word2Vec + BiLSTM / 123 / 200 da Tabela 6 |
| Diferenças e dificuldades | Slides 11 e 13; METODOLOGIA.md |
| Demonstração prática | app.py; web/; slide 12 com captura real; modelo incluído |
| Conclusão | Slide 14 |
| Código, dados, apresentação e resultados | Incluídos no pacote |
| Artigo original | Link e atalho fornecidos; PDF binário não incluído devido a bloqueio de download. Baixar o original antes da entrega formal. |
| Limite de integrantes | Mantidos seis, conforme pedido. O limite de cinco do enunciado continua sendo uma exceção, sem presumir autorização do professor. |
| Reprodução exata da arquitetura | A família BiLSTM é a mesma; detalhes ausentes da fonte foram reconstruídos e documentados. Não se afirma igualdade exata de arquitetura, pesos, partições recorrentes ou resultados. |

O material permite apresentar uma implementação neural real baseada no artigo e discutir as diferenças. O escopo é um experimento recorrente específico, e não a reprodução de toda a bateria do artigo. A aceitação desse recorte e das lacunas documentadas cabe ao professor.
