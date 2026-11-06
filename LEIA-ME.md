# Texto & Rede — experimento neural para apresentação

Projeto novo e independente de `rede-neural`. Reconstrói o experimento **Word2Vec + BiLSTM** do artigo de Masih et al. (2025), com dados originais e parâmetros publicados. As decisões necessárias por falta de detalhes dos autores estão em `METODOLOGIA.md`. Não é reprodução integral de todos os experimentos do artigo.

## Abrir a demonstração no Windows

1. Extraia o ZIP inteiro para uma pasta (não execute dentro do ZIP).
2. Instale Python 3.12, caso ainda não esteja instalado.
3. Execute `Instalar.bat` uma vez, com internet. Ele cria um ambiente e instala as dependências.
4. Execute `Iniciar_Interface.bat`. O navegador abrirá http://127.0.0.1:8766.
5. Mantenha o terminal aberto. Para encerrar, pressione Ctrl+C.

O pacote inclui os pesos treinados em `resultados/seed123/modelo.pt`: não é necessário treinar para demonstrar. Cole texto **em inglês**, idioma da base original. Os botões carregam exemplos da validação com rótulos dos autores; não são selecionados pelo acerto. Podem ocorrer erros — mostre e discuta isso na apresentação.

## Linux/macOS ou terminal

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

No macOS, instale `torch==2.5.1` pelo índice padrão do pip, sem `--index-url` de CPU Linux/Windows. Para alterar a porta: `python app.py --port 8767`. Para não abrir o navegador automaticamente: `python app.py --no-browser`.

## Refazer o experimento

Preserve/renomeie a pasta `resultados/seed123` antes de repetir a mesma semente. O programa recusa sobrescrever um experimento já iniciado. Os resultados fornecidos correspondem à semente 123. Outras sementes são novos experimentos e não estão implicitamente executadas.

```bash
python treinar.py
python -m unittest discover -s tests -v
```

A base original está em `dados/shuffled_data.csv`. Se ausente, o código baixa a revisão fixada e verifica seu SHA-256. Todo treino usa CPU por padrão e registra o tempo. Os IDs de treino/validação/teste, histórico, auditoria, previsões e relatório ficam em `resultados/seed123`. O código avalia o teste somente após selecionar o checkpoint pela validação.

Para executar outra semente: `python treinar.py --seed 42`; depois `python app.py --model-dir resultados/seed42`. Para reproduzir o ambiente medido, consulte `resultados/ambiente.txt`.

## Arquivos principais

- `modelo.py`: limpeza, codificação, arquitetura BiLSTM e inferência.
- `treinar.py`: leitura da base, divisões, Word2Vec, aprendizado e avaliação.
- `config.json`: configuração do experimento.
- `app.py` e `web/`: servidor e interface local.
- `tests/`: verificações de divisões, máscaras, gradientes, métricas, pesos salvos e API.
- `slides/`: apresentação em PDF e PowerPoint; imagens dos resultados e da interface.
- `ROTEIRO_APRESENTACAO.md`: divisão de falas entre os seis integrantes.
- `fontes/referencia_artigo.json`: identificação, URLs e linha publicada para comparação.
- `fontes/Artigo_original.url`: atalho para o PDF original; não é o PDF armazenado localmente.

## Entrega e escopo acadêmico

São entregues código, pesos treinados, dados originais, resultados e slides. O artigo pode ser obtido pelo DOI https://doi.org/10.1038/s41598-025-27377-z ou pelo atalho em fontes. O PDF original não pôde ser baixado neste ambiente; inclua-o na entrega à disciplina.

O grupo foi mantido com seis integrantes por solicitação. O enunciado limita a cinco, portanto a composição continua uma exceção ao roteiro. A reconstrução de uma arquitetura recorrente específica, com lacunas documentadas, precisa ser apresentada com esse escopo; não afirmar fidelidade integral ao artigo ou validação em português.

Os slides são gerados com `python gerar_material.py` após o treinamento e a captura da interface. Dependências adicionais em `requirements-slides.txt`. Esse comando usa os relatórios reais para preencher gráficos e números; não inventa resultados ausentes.
