import json
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modelo import BiLSTM, tokenize, encode, collate, Predictor, ROOT
from treinar import split_indices, metrics


class ProtocolTests(unittest.TestCase):
    def test_split_disjoint_balanced_reproducible(self):
        y=np.tile([0,1],1000)
        parts=split_indices(y,123)
        self.assertEqual([len(parts[x]) for x in ('train','val','test')],[1400,300,300])
        self.assertEqual(len(set().union(*(set(v) for v in parts.values()))),2000)
        for a,b in [('train','val'),('train','test'),('val','test')]:self.assertFalse(set(parts[a])&set(parts[b]))
        for name,ids in parts.items():
            self.assertEqual(int(y[ids].sum()),len(ids)//2)
            np.testing.assert_array_equal(ids,split_indices(y,123)[name])

    def test_clean_and_unknown_and_truncation(self):
        tokens=tokenize('The TEXT, is useful!',{'the','is'})
        self.assertEqual(tokens,['text','useful'])
        self.assertEqual(encode(tokens,{'<PAD>':0,'<UNK>':1,'text':2},1),[2])
        self.assertEqual(encode(['unseen'],{},200),[1])

    def test_padding_does_not_change_prediction(self):
        torch.manual_seed(1);torch.set_num_threads(1)
        config={'embedding_dim':8,'hidden_size':4,'num_layers':1,'dropout':0.0}
        m=BiLSTM(12,config).eval()
        x,l,_=collate([([2,3],0)])
        xb,lb,_=collate([([2,3],0),([4,5,6,7,8],1)])
        with torch.no_grad():torch.testing.assert_close(m(x,l)[0],m(xb,lb)[0],rtol=1e-5,atol=1e-6)

    def test_training_changes_recurrent_weights(self):
        config={'embedding_dim':8,'hidden_size':4,'num_layers':1,'dropout':0.0}
        m=BiLSTM(12,config);old=m.lstm.weight_ih_l0.detach().clone()
        x,l,y=collate([([2,3],0),([4,5,6],1)])
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        loss=torch.nn.functional.cross_entropy(m(x,l),y)
        loss.backward();opt.step()
        self.assertGreater(float((old-m.lstm.weight_ih_l0).abs().max().detach()),0)

    def test_metrics_include_both_error_types(self):
        r=metrics([0,0,1,1],[[.9,.1],[.1,.9],[.8,.2],[.2,.8]])
        self.assertEqual(r['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(r['accuracy'],.5)


@unittest.skipUnless((ROOT/'resultados/seed123/relatorio.json').exists(),'Treinamento ainda não concluído')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.predictor=Predictor()

    def test_saved_model_matches_reported_test_prediction(self):
        import pandas as pd
        from treinar import prepare_data
        f=prepare_data(ROOT/'dados/shuffled_data.csv')
        predictions=pd.read_csv(ROOT/'resultados/seed123/predicoes_teste.csv')
        row=predictions.iloc[0]
        text=f.iloc[int(row.row_id)].Data
        score=self.predictor.predict(text)['score_ai']
        self.assertAlmostEqual(score,row.score_ai,places=5)

    def test_invalid_input(self):
        for value in ['',None,12,'a'*100001,'the and is']:
            with self.assertRaises(ValueError):self.predictor.predict(value)

    def test_http_real_model_and_bad_requests(self):
        import threading,urllib.request,urllib.error
        from app import create_server
        server=create_server(0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            self.assertEqual(urllib.request.urlopen(base).status,200)
            def post(payload,origin=None):
                headers={'Content-Type':'application/json','X-Local-UI':'1'}
                if origin:headers['Origin']=origin
                req=urllib.request.Request(base+'/api/analyze',data=json.dumps(payload).encode(),headers=headers)
                return urllib.request.urlopen(req)
            result=json.load(post({'text':'People discuss education and technology in different ways.'}))
            self.assertIn(result['label'],[0,1]);self.assertTrue(0<=result['score_ai']<=1)
            for payload in [{},[],{'text':''},{'text':12}]:
                with self.assertRaises(urllib.error.HTTPError) as e:post(payload)
                self.assertEqual(e.exception.code,400)
            with self.assertRaises(urllib.error.HTTPError) as e:post({'text':'text'},'https://example.com')
            self.assertEqual(e.exception.code,403)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
