# Guia para explicar o código

Fluxo de inferência: clique em Analisar → fetch em web/app.js → Handler.do_POST em app.py → Predictor.predict → tokenize → encode → collate → BiLSTM.forward → softmax → retorno para a interface.

Fluxo de treinamento: prepare_data → split_indices → tokenize → vocabulário/Word2Vec apenas no treino → BiLSTM → CrossEntropyLoss.backward → Adam.step → evaluate na validação → salvar melhor checkpoint → evaluate no teste → metrics.

| Arquivo | Função | Linha da definição |
|---|---|---|
| modelo.py | `tokenize()` | 12 |
| modelo.py | `encode()` | 18 |
| modelo.py | `collate()` | 22 |
| modelo.py | `__init__()` | 32 |
| modelo.py | `forward()` | 40 |
| modelo.py | `__init__()` | 49 |
| modelo.py | `predict()` | 60 |
| treinar.py | `save_json()` | 26 |
| treinar.py | `split_indices()` | 30 |
| treinar.py | `metrics()` | 36 |
| treinar.py | `evaluate()` | 48 |
| treinar.py | `prepare_data()` | 60 |
| treinar.py | `train()` | 75 |
| app.py | `create_server()` | 12 |
| app.py | `log_message()` | 21 |
| app.py | `send()` | 24 |
| app.py | `valid_host()` | 34 |
| app.py | `do_GET()` | 37 |
| app.py | `do_POST()` | 45 |

As linhas são de definição. A ordem de execução segue as chamadas. Funções de biblioteca, como backward e step, não têm definição nestes arquivos.
