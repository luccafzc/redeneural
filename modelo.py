"""BiLSTM: rede neural recorrente. Inferência não modifica os pesos."""
import json
import string
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parent


def tokenize(text, stopwords):
    """Minúsculas, pontuação ASCII removida e stopwords inglesas (artigo)."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if w not in stopwords]


def encode(tokens, vocabulary, max_length):
    return [vocabulary.get(t, 1) for t in tokens[:max_length]] or [1]


def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'], padding_idx=0)
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_size'],
                            num_layers=config['num_layers'], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(config['hidden_size'] * 2, 2)

    def forward(self, ids, lengths):
        embedded = self.embedding(ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))


class Predictor:
    def __init__(self, folder=None):
        folder = Path(folder or ROOT / 'resultados' / 'seed123')
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        self.config = metadata['config']
        self.vocabulary = metadata['vocabulary']
        self.stopwords = set(metadata['stopwords'])
        torch.set_num_threads(self.config['threads'])
        self.model = BiLSTM(len(self.vocabulary), self.config)
        self.model.load_state_dict(torch.load(folder / 'modelo.pt', map_location='cpu', weights_only=True))
        self.model.eval()

    def predict(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 100000:
            raise ValueError('Informe um texto não vazio de até 100.000 caracteres.')
        tokens = tokenize(text, self.stopwords)
        if not tokens:
            raise ValueError('O texto não contém palavras utilizáveis após a limpeza.')
        ids = encode(tokens, self.vocabulary, self.config['max_length'])
        if all(i == 1 for i in ids):
            raise ValueError('Nenhuma palavra foi reconhecida. Use um texto em inglês.')
        x, lengths, _ = collate([(ids, 0)])
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(x, lengths), dim=1)[0].tolist()
        label = int(probabilities[1] > probabilities[0])
        return {'label': label, 'prediction': ['Humano', 'IA'][label], 'score_ai': probabilities[1],
                'tokens_total': len(tokens), 'tokens_analyzed': len(ids),
                'truncated': len(tokens) > self.config['max_length'],
                'unknown_fraction': sum(i == 1 for i in ids) / len(ids),
                'model': 'Word2Vec + BiLSTM', 'language': 'en',
                'notice': 'Pontuação experimental, sem calibração; não comprova autoria. Modelo estudado em inglês.'}
