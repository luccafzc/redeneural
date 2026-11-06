"""Reconstrução documentada do experimento recorrente de Masih et al. (2025)."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from modelo import ROOT, BiLSTM, tokenize, encode, collate

DATA_SHA256 = 'f9ab6ff46e38f4930fd3375e6ca5a56134c064855c029e5ed2ad08b3fa9e5c6b'
COMMIT = '82bf5888efba65bc9aed64f6b02f198d493f806d'
DATA_URL = f'https://raw.githubusercontent.com/shamylafirdoos/Gpt-vs-Human-Text-Classification/{COMMIT}/shuffled_data.csv'


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def split_indices(labels, seed):
    train, remainder = train_test_split(np.arange(len(labels)), test_size=.3, stratify=labels, random_state=seed)
    val, test = train_test_split(remainder, test_size=.5, stratify=np.asarray(labels)[remainder], random_state=seed)
    return {'train': train, 'val': val, 'test': test}


def metrics(y, probabilities):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    pred = probabilities.argmax(axis=1)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'precision_ai': float(precision_score(y, pred, zero_division=0)),
            'recall_ai': float(recall_score(y, pred, zero_division=0)),
            'f1_ai': float(f1_score(y, pred, zero_division=0)),
            'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'roc_auc': float(roc_auc_score(y, probabilities[:, 1])) if len(set(y)) == 2 else None}


def evaluate(model, loader):
    model.eval()
    labels, probabilities, total_loss = [], [], 0.0
    with torch.inference_mode():
        for x, lengths, y in loader:
            logits = model(x, lengths)
            total_loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
            probabilities.extend(torch.softmax(logits, dim=1).tolist())
            labels.extend(y.tolist())
    return total_loss / len(labels), np.asarray(labels), np.asarray(probabilities)


def prepare_data(path):
    path = Path(path)
    if not path.exists():
        from urllib.request import urlretrieve
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != DATA_SHA256:
        raise ValueError('A base não corresponde à revisão dos autores. Não iniciar o treinamento.')
    frame = pd.read_csv(path, keep_default_na=False)
    if set(frame.columns) != {'Data', 'Labels'} or not frame.Labels.isin([0, 1]).all():
        raise ValueError('Esquema ou rótulos inválidos.')
    return frame


def train(args):
    config = json.loads((ROOT / 'config.json').read_text())
    if args.seed is not None: config['seed'] = args.seed
    seed = config['seed']
    out = ROOT / 'resultados' / f'seed{seed}'
    if out.exists() and (out / 'metadata.json').exists():
        raise ValueError(f'Experimento já iniciado em {out}. Escolha outra pasta ou preserve/renomeie o resultado antes de refazer.')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    frame = prepare_data(args.data)
    labels = frame.Labels.to_numpy()
    stopwords = set((ROOT / 'fontes' / 'stopwords-en.txt').read_text().splitlines())
    tokens = [tokenize(text, stopwords) for text in frame.Data]
    parts = split_indices(labels, seed)
    # Nenhum vocabulário ou embedding é ajustado com validação/teste.
    counts = Counter(w for i in parts['train'] for w in tokens[i])
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    for word, n in counts.most_common():
        if n >= config['min_count'] and len(vocabulary) < config['max_vocab']:
            vocabulary[word] = len(vocabulary)
    corpus = [tokens[i] for i in parts['train']]
    print('Treinando Word2Vec apenas no treino...', flush=True)
    w2v = Word2Vec(corpus, vector_size=config['embedding_dim'], window=config['word2vec_window'],
                    sg=config['word2vec_sg'], negative=config['word2vec_negative'],
                    min_count=config['min_count'], epochs=config['word2vec_epochs'], seed=seed, workers=1,
                    hashfxn=lambda word: int(hashlib.sha256(word.encode()).hexdigest()[:8], 16))
    model = BiLSTM(len(vocabulary), config)
    with torch.no_grad():
        for word, index in vocabulary.items():
            if word in w2v.wv: model.embedding.weight[index].copy_(torch.from_numpy(w2v.wv[word].copy()))
        model.embedding.weight[0].zero_()
        model.embedding.weight[1].copy_(model.embedding.weight[2:].mean(dim=0))
    del w2v
    encoded = [encode(ts, vocabulary, config['max_length']) for ts in tokens]
    datasets = {name: [(encoded[i], int(labels[i])) for i in ids] for name, ids in parts.items()}
    loaders = {name: DataLoader(ds, batch_size=config['batch_size'], shuffle=name == 'train',
                                collate_fn=collate, num_workers=0) for name, ds in datasets.items()}
    canonical = [' '.join(t) for t in tokens]
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in canonical]
    sets = {name: {hashes[i] for i in ids} for name, ids in parts.items()}
    audit = {'n_original': len(frame), 'class_counts': frame.Labels.value_counts().sort_index().to_dict(),
             'duplicates_after_preprocessing': len(hashes) - len(set(hashes)),
             'conflicting_groups_after_preprocessing': int(pd.DataFrame({'hash': hashes, 'label': labels}).groupby('hash').label.nunique().gt(1).sum()),
             'overlap_unique_hashes': {a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
             'empty_after_preprocessing': sum(not t for t in tokens),
             'partitions': {name: {'n':len(ids),'ai':int(labels[ids].sum()),'truncated':sum(len(tokens[i])>config['max_length'] for i in ids)} for name,ids in parts.items()},
             'policy':'Divisões por linhas como no notebook dos autores. Duplicatas mantidas e auditadas; avaliação adicional exclui hashes do teste presentes no treino/validação.'}
    save_json(out/'auditoria.json',audit)
    for name,ids in parts.items():
        pd.DataFrame({'row_id':ids,'text_hash':[hashes[i] for i in ids],'label':labels[ids]}).to_csv(out/f'ids_{name}.csv',index=False)
    metadata = {'config':config,'vocabulary':vocabulary,'stopwords':sorted(stopwords),
                'dataset_sha256':DATA_SHA256,'authors_commit':COMMIT,'parameter_count':sum(p.numel() for p in model.parameters()),
                'versions':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__},
                'architecture':'Embedding Word2Vec ajustável → BiLSTM (64 unidades por direção) → Linear 128×2 → softmax na inferência',
                'device':'cpu','torch_threads':torch.get_num_threads()}
    save_json(out/'metadata.json', metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    best_loss, bad_epochs, history = float('inf'), 0, []
    print(f'BiLSTM: {metadata["parameter_count"]:,} parâmetros; {audit["partitions"]}',flush=True)
    for epoch in range(1, config['max_epochs']+1):
        model.train(); loss_sum=0.; epoch_start=time.perf_counter()
        for batch_index,(x,lengths,y) in enumerate(loaders['train']):
            optimizer.zero_grad(set_to_none=True)
            logits=model(x,lengths)
            loss=torch.nn.functional.cross_entropy(logits,y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(y)
            if (batch_index+1)%100==0: print(f'Época {epoch}: lote {batch_index+1}/{len(loaders["train"])}',flush=True)
        val_loss, val_y, val_prob=evaluate(model,loaders['val'])
        row={'epoch':epoch,'train_loss':loss_sum/len(datasets['train']),'val_loss':val_loss,
             'val_accuracy':float(accuracy_score(val_y,val_prob.argmax(axis=1))),
             'seconds':time.perf_counter()-epoch_start}
        history.append(row);save_json(out/'historico.json',history)
        print(json.dumps(row),flush=True)
        if val_loss < best_loss:
            best_loss=val_loss;bad_epochs=0;best_epoch=epoch
            torch.save(model.state_dict(),out/'modelo.pt')
        else:
            bad_epochs+=1
            if bad_epochs>=config['patience']:break
    # O teste é avaliado apenas após a seleção do checkpoint pela validação.
    model.load_state_dict(torch.load(out/'modelo.pt',weights_only=True))
    test_loss,y,p=evaluate(model,loaders['test'])
    test_ids=parts['test']
    unseen=np.array([hashes[i] not in sets['train']|sets['val'] for i in test_ids])
    report={'status':'completed','seed':seed,'epochs_run':len(history),'best_epoch':best_epoch,
            'selection':'menor perda de validação; parada antecipada após 3 épocas sem melhora',
            'elapsed_seconds':time.perf_counter()-started,'test_loss':test_loss,
            'test':metrics(y,p),'test_without_seen_hashes':metrics(y[unseen],p[unseen]),
            'paper_reference':{'model':'BiLSTM','seed':123,'embedding_dim':200,'accuracy':.8457,
                               'source':'Texto da seção Performance evaluation, referência à Tabela 6',
                               'comparison':'Descritiva: detalhes recorrentes não publicados foram reconstruídos; não são os mesmos pesos/divisões garantidos.'},
            'scope':'Reconstrução parcial do experimento BiLSTM; não reproduz a bateria completa nem os transformers.',
            'score':'softmax sem calibração; não é certeza de autoria'}
    save_json(out/'relatorio.json',report)
    pd.DataFrame({'row_id':test_ids,'text_hash':[hashes[i] for i in test_ids], 'label':y,
                  'prediction':p.argmax(axis=1),'score_ai':p[:,1],'unseen_hash':unseen}).to_csv(out/'predicoes_teste.csv',index=False)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--data',default=str(ROOT/'dados'/'shuffled_data.csv'))
    train(parser.parse_args())
