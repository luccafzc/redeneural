"""Interface local da rede neural. Execute: python app.py."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from modelo import ROOT, Predictor


def create_server(port=8766, model_dir=None):
    folder=Path(model_dir or ROOT/'resultados'/'seed123')
    predictor=Predictor(folder)
    lock=threading.Lock()
    report=json.loads((folder/'relatorio.json').read_text(encoding='utf-8'))
    examples_path=ROOT/'web'/'exemplos.json'
    examples=json.loads(examples_path.read_text(encoding='utf-8')) if examples_path.exists() else []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Não registrar os textos enviados.

        def send(self,status,body,kind='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.valid_host():return self.send(403,{'error':'Abra a interface pelo endereço local.'})
            if self.path=='/api/info':
                return self.send(200,{'model':'Word2Vec + BiLSTM','report':report,'examples':examples})
            assets={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if self.path not in assets:return self.send(404,{'error':'Página não encontrada.'})
            name,kind=assets[self.path];self.send(200,(ROOT/'web'/name).read_bytes(),kind)

        def do_POST(self):
            if not self.valid_host():return self.send(403,{'error':'Endereço inválido.'})
            origin=self.headers.get('Origin')
            if origin is not None and origin != 'http://'+self.headers.get('Host',''):
                return self.send(403,{'error':'Origem inválida.'})
            if self.path!='/api/analyze':return self.send(404,{'error':'Rota inválida.'})
            if self.headers.get('X-Local-UI')!='1':return self.send(403,{'error':'Use a interface local.'})
            if self.headers.get_content_type()!='application/json':return self.send(415,{'error':'Envie JSON.'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=650000:return self.send(413,{'error':'Texto muito grande.'})
                self.connection.settimeout(15)
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Informe um objeto com o campo text.')
                with lock:result=predictor.predict(data.get('text'))
                self.send(200,result)
            except (ValueError,UnicodeError) as exc:self.send(400,{'error':str(exc)})
            except TimeoutError:self.send(408,{'error':'Tempo esgotado ao receber o texto.'})

    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--model-dir',default=None)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    try:server=create_server(args.port,args.model_dir)
    except FileNotFoundError:
        raise SystemExit('Modelo/relatório ausente. Execute python treinar.py e aguarde a conclusão.')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'Interface: {url}\nPressione Ctrl+C para encerrar.',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
