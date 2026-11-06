"""Gera slides editáveis e PDF a partir dos resultados reais do treinamento."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitz
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'slides';OUT.mkdir(exist_ok=True)
R=json.loads((ROOT/'resultados/seed123/relatorio.json').read_text())
A=json.loads((ROOT/'resultados/seed123/auditoria.json').read_text())
M=json.loads((ROOT/'resultados/seed123/metadata.json').read_text())
H=json.loads((ROOT/'resultados/seed123/historico.json').read_text())
REF=json.loads((ROOT/'fontes/referencia_artigo.json').read_text())
assert R['status']=='completed','É preciso concluir o treinamento antes de montar os slides.'
T=R['test'];P=REF['selected_experiment']
NAVY='#193B48';INK='#213C4D';TEAL='#318775';PALE='#E8F0EB';BG='#F7F6F2';MUTED='#647783';ORANGE='#C68045'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})

def charts():
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    ax.plot([r['epoch'] for r in H],[r['train_loss'] for r in H],marker='o',lw=2.5,color=TEAL,label='Treino')
    ax.plot([r['epoch'] for r in H],[r['val_loss'] for r in H],marker='o',lw=2.5,color=ORANGE,label='Validação')
    ax.axvline(R['best_epoch'],color=MUTED,ls='--',alpha=.6,label='Checkpoint escolhido')
    ax.set(xlabel='Época',ylabel='Entropia cruzada',xticks=range(1,len(H)+1));ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.savefig(OUT/'curva_treinamento.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.8),layout='constrained')
    cm=np.array(T['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=1500)
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',fontsize=25,color='white' if v>800 else INK)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Humano','IA'],yticklabels=['Humano','IA'],xlabel='Classe prevista',ylabel='Rótulo da base')
    fig.savefig(OUT/'matriz_confusao.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    names=['Acurácia','Precisão IA','Recall IA','F1 IA'];x=np.arange(4)
    local=[T[k]*100 for k in ['accuracy','precision_ai','recall_ai','f1_ai']]
    original=[P[k]*100 for k in ['accuracy','precision','recall','f1']]
    for shift,vals,color,label in [(-.18,original,MUTED,'Artigo — Tabela 6'),(.18,local,TEAL,'Execução do grupo')]:
        bars=ax.bar(x+shift,vals,.34,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.2f}%' for v in vals],padding=3,fontsize=10)
    ax.set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Percentual');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(OUT/'comparacao.png',dpi=180);plt.close(fig)

for name,filename in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(ROOT/'fontes'/'fonts'/filename)))

class Deck:
    def __init__(self):
        self.prs=Presentation();self.prs.slide_width=Inches(13.333333);self.prs.slide_height=Inches(7.5)
        self.pdf=canvas.Canvas(str(OUT/'Apresentacao-BiLSTM.pdf'),pagesize=(1200,675))
        self.pdf.setTitle('Detecção de textos — reconstrução Word2Vec + BiLSTM')
        self.n=0;self.notes=[]
    def rect(self,x,y,w,h,color,line=None):
        shape=self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x/90),Inches(y/90),Inches(w/90),Inches(h/90))
        shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(color[1:]);shape.line.fill.background()
        self.pdf.setFillColor(color);self.pdf.rect(x,675-y-h,w,h,fill=1,stroke=0)
    def text(self,s,x,y,w,size=23,color=INK,bold=False,leading=1.3):
        font='DVB' if bold else 'DV';lines=[]
        for paragraph in s.split('\n'):
            line=''
            for word in paragraph.split():
                candidate=(line+' '+word).strip()
                if pdfmetrics.stringWidth(candidate,font,size)>w and line:lines.append(line);line=word
                else:line=candidate
            lines.append(line)
        for j,line in enumerate(lines):
            yy=y+j*size*leading
            assert yy+size < 670,(self.n,s,yy)
            box=self.slide.shapes.add_textbox(Inches(x/90),Inches(yy/90),Inches((w+10)/90),Inches(size*1.45/90))
            tf=box.text_frame;tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0;tf.word_wrap=False
            p=tf.paragraphs[0];p.text=line;p.font.name='DejaVu Sans';p.font.size=Pt(size*.8);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color[1:]);p.space_after=Pt(0)
            self.pdf.setFillColor(color);self.pdf.setFont(font,size);self.pdf.drawString(x,675-yy-size,line)
        return len(lines)*size*leading
    def image(self,path,x,y,w,h):
        im=Image.open(path);iw,ih=im.size;scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
        x+=(w-ww)/2;y+=(h-hh)/2
        self.slide.shapes.add_picture(str(path),Inches(x/90),Inches(y/90),width=Inches(ww/90),height=Inches(hh/90))
        self.pdf.drawImage(str(path),x,675-y-hh,width=ww,height=hh,mask='auto')
    def new(self,title,speaker,notes,source='Masih et al. (2025) · DOI: 10.1038/s41598-025-27377-z',dark=False):
        if self.n:self.pdf.showPage()
        self.slide=self.prs.slides.add_slide(self.prs.slide_layouts[6]);self.n+=1
        self.rect(0,0,1200,675,NAVY if dark else BG)
        self.text('TEXTO & REDE  /  RÉPLICA ACADÊMICA',48,25,800,11, '#A9CCBF' if dark else TEAL,True)
        self.text(speaker.upper(),820,25,330,11,'#A9CCBF' if dark else MUTED,True)
        if title:self.text(title,48,70,1100,34,'#FFFFFF' if dark else INK,True)
        self.rect(48,632,1104,1,'#41616A' if dark else '#D7DFDA')
        self.text(source,48,645,1050,10,'#B1C8C4' if dark else MUTED)
        self.text(f'{self.n:02}',1120,641,40,15,'#B1C8C4' if dark else MUTED,True)
        self.slide.notes_slide.notes_text_frame.text=notes+'\n\nFonte: '+source
        self.notes.append((self.n,title,speaker,notes))
    def card(self,x,y,w,h,tag,title,body):
        self.rect(x,y,w,h,'#FFFFFF');self.rect(x,y,5,h,TEAL)
        self.text(tag,x+23,y+20,w-45,12,TEAL,True)
        title_height=self.text(title,x+23,y+54,w-45,27,INK,True)
        self.text(body,x+23,y+54+title_height+16,w-45,19,MUTED)
    def bullet(self,title,body,x,y,w):
        self.text(title,x,y,w,23,INK,True)
        self.text(body,x,y+39,w,19,MUTED)
    def finish(self):
        self.pdf.save();self.prs.save(OUT/'Apresentacao-BiLSTM.pptx')


def build():
    charts();d=Deck()
    d.new('', 'João Vitor Frare', 'Apresente os seis integrantes, a disciplina e o objetivo: reconstruir um experimento neural do artigo. Delimite o trabalho à BiLSTM, sem afirmar reprodução integral.',dark=True)
    d.text('Textos humanos\nou gerados por IA?',55,125,1090,55,'#FFFFFF',True)
    d.text('Reconstrução experimental com Word2Vec + BiLSTM',59,285,1060,27,'#AFD1C0')
    d.rect(59,351,1080,2,'#41616A')
    d.text('Alexandre Lampert · Felipe Lucca · Gabriel Silva',59,385,1060,21,'#FFFFFF')
    d.text('Gustavo Pigatto · Leonardo Forgiarini · João Vitor Frare',59,420,1060,21,'#FFFFFF')
    d.text('Ciência da Computação | UNOESC — Disciplina: Inteligência Artificial\nProfessor: Vinícius Almeida dos Santos',59,510,1060,18,'#B8CDC8')

    d.new('O artigo que orienta o experimento','João Vitor Frare','Identifique o artigo, ano, periódico e autores. Ele compara modelos clássicos, recorrentes e transformers. Nosso recorte é a linha Word2Vec + BiLSTM, semente 123 e dimensão 200 da Tabela 6.')
    d.text(REF['title'],50,146,1070,30,INK,True)
    d.text('Adven Masih · Bushra Afzal · Shamyla Firdoos · Jabar Mahmood\nAitizaz Ali · Mohamed Shabbir Abdulnabi · Daniel Musafiri Balungu',50,262,1080,18,MUTED)
    d.card(50,365,525,205,'PUBLICAÇÃO','Scientific Reports','Volume 15 · Artigo 43310 · 2025\nDOI: 10.1038/s41598-025-27377-z')
    d.card(600,365,550,205,'RECORTE ESCOLHIDO','Word2Vec + BiLSTM','Dimensão 200 · Semente 123\nExperimento recorrente da Tabela 6')

    d.new('O problema e o objetivo','João Vitor Frare','Explique a relevância da identificação de padrões de escrita. Os rótulos são os da base. A meta é estudar reprodução de um método neural, não oferecer prova de autoria nem um detector universal.')
    d.card(50,165,345,320,'PROBLEMA','Textos semelhantes','A escrita gerada por IA pode se aproximar da escrita humana e dificultar a distinção.')
    d.card(425,165,345,320,'TAREFA','Duas classes','Receber um texto e prever um dos rótulos: humano (0) ou IA (1).')
    d.card(800,165,350,320,'OBJETIVO','Reproduzir e avaliar','Treinar uma rede do artigo, medir seus erros e discutir diferenças nos resultados.')
    d.text('A classificação estima padrões da base; não comprova a autoria de um texto.',54,545,1080,22,TEAL,True)

    d.new('Da metodologia original ao nosso recorte','João Vitor Frare','O artigo inclui redes recorrentes e transformers. Mantemos base e hiperparâmetros publicados da BiLSTM. O código recorrente não está no repositório; unidades, pesos Word2Vec e alguns detalhes foram reconstruídos. A implementação não é uma cópia exata.')
    d.bullet('No artigo','Dados rotulados → processamento → modelos neurais → treinamento → teste e comparação.',55,165,1060)
    d.bullet('Neste trabalho','Base original → Word2Vec → BiLSTM → comparação com a linha correspondente da Tabela 6.',55,285,1060)
    d.rect(50,420,1100,145,PALE)
    d.text('Reconstrução parcial, com escolhas explícitas',74,441,1030,27,TEAL,True)
    d.text('Os autores não disponibilizaram o código recorrente completo. As lacunas estão documentadas em METODOLOGIA.md.',74,488,1030,21,INK)

    d.new('Os dados são os mesmos do repositório','Leonardo Forgiarini','Mostre dados/shuffled_data.csv. São 20.000 linhas, 10.000 por classe, em inglês. O checksum garante a versão. A divisão 70/15/15 segue o notebook público, mas os índices específicos do experimento recorrente não foram publicados.','Base: repositório dos autores, revisão 82bf5888… · Auditoria: resultados/seed123/auditoria.json')
    d.text('20.000',55,155,600,74,TEAL,True);d.text('textos em inglês · 50% humano / 50% IA',60,249,1030,24,INK)
    for x,w,number,label in [(55,735,'14.000','TREINO · 70%'),(805,165,'3.000','VALIDAÇÃO · 15%'),(985,165,'3.000','TESTE · 15%')]:
        d.rect(x,340,w,95,TEAL if x==55 else NAVY);d.text(number,x+15,355,w-25,29,'#FFFFFF',True)
        d.text(label,x,454,w,12,MUTED,True)
    d.text('Divisões estratificadas · Semente 123 · Índices e hashes registrados',55,532,1100,23,INK)

    d.new('Preparação: do texto aos vetores','Leonardo Forgiarini','Explique que a limpeza segue os passos descritos no artigo. O vocabulário e o Word2Vec são ajustados só no treino para não aprender com o teste. Cada palavra vira um vetor de 200 números; isso não é TF-IDF. Textos longos são truncados após a limpeza.')
    steps=[('01','Limpeza','Minúsculas, pontuação e stopwords inglesas.'),('02','Tokens','Separação por espaços; limite de 200 tokens.'),('03','Word2Vec','Vetores de 200 dimensões, aprendidos só no treino.')]
    for i,(tag,title,body) in enumerate(steps):d.card(50+i*375,170,350,290,tag,title,body)
    d.text(f'Vocabulário: {len(M["vocabulary"]):,} entradas, incluindo PAD e UNK'.replace(',','.'),55,508,1100,25,TEAL,True)
    d.text('Word2Vec skip-gram: 5 épocas, janela 5; configurações locais documentadas.',55,555,1100,18,MUTED)

    d.new('A arquitetura neural implementada','Gabriel Silva','Embedding: 200 valores por token. Uma camada bidirecional com 64 unidades em cada sentido. As portas da LSTM usam sigmoid, o estado candidato usa tanh. Concatenação dos estados finais: 128 valores. Linear 128 para 2 logits; softmax na inferência. As 64 unidades e a cabeça linear são decisões locais, pois os autores não especificaram esses detalhes.')
    blocks=[('Entrada','até 200 tokens'),('Embedding','200 dimensões'),('BiLSTM','64 + 64 unidades'),('Linear','128 → 2 logits'),('Softmax','Humano / IA')]
    for i,(title,body) in enumerate(blocks):
        x=50+i*225;d.rect(x,192,202,145,NAVY if i==2 else PALE)
        d.text(title,x+13,217,176,23,'#FFFFFF' if i==2 else INK,True)
        d.text(body,x+13,269,176,16,'#C6E0D5' if i==2 else MUTED)
        if i<4:d.text('→',x+206,240,22,18,TEAL)
    d.bullet('Memória em duas direções','A rede percorre a sequência da esquerda para a direita e no sentido inverso.',55,390,1060)
    d.text(f'{M["parameter_count"]:,} parâmetros treináveis · Portas sigmoid · Estado tanh'.replace(',','.'),55,530,1080,23,TEAL,True)
    d.text('64 unidades por direção e uma camada: escolhas de reconstrução, não dados publicados.',55,577,1080,15,MUTED)

    d.new('Como o modelo aprendeu','Gabriel Silva','Apresente os parâmetros publicados. Adam ajusta pesos para reduzir entropia cruzada. A curva revela sobreajuste: a perda do treino continua caindo e a validação piora. Recuperamos a época 1 e paramos após três épocas sem melhora. O teste só foi executado depois da escolha.','Parâmetros: Tabela 4 do artigo · Curvas: resultados/seed123/historico.json')
    d.image(OUT/'curva_treinamento.png',390,155,755,430)
    d.bullet('Adam · 0,001','Lote 32\nCrossEntropyLoss\nAté 30 épocas\nPaciência: 3',55,170,310)
    d.text(f'{R["epochs_run"]} épocas executadas\nCheckpoint: época {R["best_epoch"]}\nTempo: {R["elapsed_seconds"]/60:.1f} min em CPU',55,415,320,21,TEAL,True)

    d.new('Resultado real nos 3.000 textos de teste','Gustavo Pigatto','Explique a matriz: linhas são rótulos da base e colunas as previsões. Falsos positivos são humanos classificados como IA; falsos negativos são IA classificada como humana. A acurácia sozinha não mostra os dois tipos de erro.','Fonte: resultados/seed123/relatorio.json e predicoes_teste.csv · Execução local')
    d.image(OUT/'matriz_confusao.png',490,145,650,455)
    d.text(f'{100*T["accuracy"]:.2f}%',55,157,430,64,TEAL,True);d.text('Acurácia',58,244,400,22,MUTED)
    d.text(f'Precisão IA   {100*T["precision_ai"]:.2f}%\nRecall IA      {100*T["recall_ai"]:.2f}%\nF1 IA           {100*T["f1_ai"]:.2f}%',58,315,400,25,INK)
    d.text(f'{T["confusion_matrix"][0][1]} humanos apontados como IA\n{T["confusion_matrix"][1][0]} textos de IA apontados como humanos',58,470,420,19,ORANGE,True)

    d.new('Artigo × execução do grupo','Gustavo Pigatto','Compare apenas a linha Word2Vec + BiLSTM, dimensão 200 e semente 123. A diferença de acurácia é descritiva: não houve teste pareado com previsões dos autores, e os detalhes da rede/embeddings/divisão diferem. Não dizer que superamos cientificamente o artigo. A agregação exata de P/R/F1 da fonte não foi reconstruída.','Artigo: Tabela 6, Word2Vec + BiLSTM / 123 / 200 · Grupo: relatorio.json')
    d.image(OUT/'comparacao.png',50,145,1090,390)
    delta=100*(T['accuracy']-P['accuracy'])
    d.text(f'Diferença observada de acurácia: {delta:+.2f} pontos percentuais.',55,551,1100,24,TEAL,True)
    d.text('Comparação descritiva: pesos, arquitetura detalhada e divisão original não são idênticos ou verificáveis.',55,591,1100,15,MUTED)

    d.new('Auditoria: o que limita a comparação','Gustavo Pigatto','Há 185 linhas excedentes por repetição depois da limpeza e 36 grupos com rótulos conflitantes. Mantivemos as linhas no protocolo principal e auditamos a sobreposição. O teste adicional remove 32 linhas cujos hashes aparecem em treino/validação. Isso não remove paráfrases ou autores repetidos. Uma execução não mede variação entre sementes.','Fonte: resultados/seed123/auditoria.json e relatorio.json')
    d.card(50,165,350,260,'DADOS','185 repetições','Linhas excedentes após limpeza; 36 grupos de textos com rótulos conflitantes.')
    d.card(425,165,350,260,'TESTE COMPLEMENTAR','32 linhas excluídas',f'2.968 textos sem hashes vistos. Acurácia: {100*R["test_without_seen_hashes"]["accuracy"]:.2f}%.')
    d.card(800,165,350,260,'GENERALIZAÇÃO','Uma configuração','Inglês, uma semente e uma base. Não demonstra desempenho geral ou certeza de autoria.')
    d.text('Possíveis causas das diferenças: embeddings, unidades da LSTM, limpeza, divisão e inicialização.',55,485,1090,24,INK,True)
    d.text('A avaliação complementar não substitui um teste externo independente.',55,574,1100,20,MUTED)

    d.new('Demonstração da rede em funcionamento','Felipe Lucca','Abra Iniciar_Interface.bat. Mostre um exemplo rotulado da validação e clique em analisar. Explique: interface envia o texto; Predictor faz limpeza e codificação; BiLSTM.forward calcula logits; softmax gera pontuação; navegador mostra a classe. Consultas não treinam a rede. Os exemplos são escolhidos sem usar seus acertos. O código da rede está em modelo.py, e o treino em treinar.py.','Captura real da interface local · Modelo carregado: resultados/seed123/modelo.pt')
    d.image(OUT/'interface.png',48,142,790,465)
    d.bullet('1. Inserir','Texto em inglês ou exemplo da validação.',870,174,280)
    d.bullet('2. Analisar','Limpeza → vetores → BiLSTM → pontuação.',870,307,280)
    d.bullet('3. Interpretar','A previsão pode errar; score não é prova de autoria.',870,442,280)

    d.new('Dificuldades e decisões de implementação','Alexandre Lampert','Explique as lacunas do repositório e por que 64 unidades é uma escolha local. Os pesos dos transformers não puderam ser baixados. O escopo recorrente foi viável em CPU. Comente o sobreajuste e a parada antecipada. Próximos passos devem ser apresentados como futuros, sem insinuar que já foram executados.')
    d.bullet('Código recorrente não disponibilizado','Reconstruímos a BiLSTM e registramos as decisões que não aparecem no artigo.',55,165,1080)
    d.bullet('Recursos de execução','Treinamento em CPU. Downloads dos pesos de transformers bloqueados no ambiente.',55,285,1080)
    d.bullet('Sobreajuste e qualidade da base','A validação evitou usar o último checkpoint; a auditoria revelou repetições e conflitos.',55,405,1080)
    d.text('Próximos passos: outras sementes, teste externo e confirmação dos detalhes com os autores.',55,560,1080,22,TEAL,True)

    d.new('O que conseguimos reproduzir','Alexandre Lampert','Conclua: rede neural implementada e treinada, dados originais, parâmetros publicados, comparação e interface. Os números locais são reais, mas o experimento é uma reconstrução parcial. A composição de seis integrantes foi mantida por solicitação, embora o roteiro diga até cinco. Isso consta na documentação, não foi tratado como autorização do professor.',dark=True)
    d.text('Uma rede treinada.\nResultados verificáveis.',55,150,1090,48,'#FFFFFF',True)
    d.text('Base original · BiLSTM · Treino e teste · Interface · Comparação',59,300,1070,25,'#AFD1C0')
    d.text('A reconstrução permitiu estudar aprendizado, sobreajuste e erros de classificação. As lacunas da fonte impedem afirmar reprodução exata.',59,380,1040,27,'#FFFFFF')
    d.text('Fontes: artigo de Masih et al. (2025), repositório dos autores e relatórios desta execução.\nCódigo, pesos, divisões, previsões e apresentação acompanham a entrega.',59,525,1050,18,'#B8CDC8')
    d.finish()
    notes=['# Roteiro da apresentação — seis integrantes','',
           'Divisão: João (1–4), Leonardo (5–6), Gabriel (7–8), Gustavo (9–11), Felipe (12 e demonstração), Alexandre (13–14).',
           'Tempo sugerido: 15–20 minutos, incluindo a demonstração. A composição de seis foi mantida conforme solicitado; o limite do enunciado é cinco.','']
    for number,title,speaker,note in d.notes:notes += [f'## Slide {number} — {title or "Capa"}',f'**{speaker}**',note,'']
    notes += ['## Perguntas prováveis','',
              '**É uma rede neural?** Sim: embedding ajustável, camada LSTM bidirecional e camada linear treinadas por retropropagação.',
              '**Por que não RoBERTa?** O artigo também avalia redes recorrentes. O recorte escolhido foi Word2Vec + BiLSTM; os downloads de pesos transformer estavam bloqueados.',
              '**É reprodução exata?** Não. É uma reconstrução parcial com base e parâmetros publicados; faltam detalhes recorrentes dos autores, explicitados em METODOLOGIA.md.',
              '**Funciona em português?** Não há validação em português. A interface está em português, mas o modelo e os exemplos são em inglês.',
              '**O resultado melhor prova superioridade?** Não: protocolos e detalhes diferem, há uma única semente e faltam previsões originais para comparação pareada.',
              '**Por que parou na época 4?** A perda de validação não melhorou por três épocas; recuperou o checkpoint da época 1.',
              '**Por que existem duplicatas?** Preservamos a base para documentar a reconstrução. Apresentamos auditoria e um teste complementar sem hashes vistos. Isso não prova independência de fontes ou autores.',
              '**O que significam as ativações?** Sigmoid controla portas da LSTM; tanh limita os estados; softmax transforma os dois logits em pontuações normalizadas.',
              '**A interface aprende com as consultas?** Não. Ela usa pesos salvos em modo de inferência e não armazena o texto.','']
    (ROOT/'ROTEIRO_APRESENTACAO.md').write_text('\n\n'.join(notes),encoding='utf-8')
    pdf=fitz.open(OUT/'Apresentacao-BiLSTM.pdf')
    assert len(pdf)==14
    contact=Image.new('RGB',(1200,4*169),'#FFFFFF')
    for i,page in enumerate(pdf):
        pix=page.get_pixmap(matrix=fitz.Matrix(.25,.25));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        contact.paste(im,((i%4)*300,(i//4)*169))
    contact.save(OUT/'visao-geral.png')
    print('Slides gerados: 14 páginas em PDF e PowerPoint editável.')


if __name__=='__main__':build()
