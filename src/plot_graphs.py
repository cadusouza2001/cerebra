import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

TRAIN_LOG = 'training_log.csv'
EVAL_FILE = 'evaluation_results.jsonl'
EMB_FILE = 'doc_embeddings.npz'
DATASET_FILE = 'qa_dataset/spark_qa_generative_dataset.jsonl'


def plot_loss_curve(log_path=TRAIN_LOG):
    """Plota curva de perda treino/val"""
    df = pd.read_csv(log_path)
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='treino')
    plt.plot(df['epoch'], df['val_loss'], label='validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.title('Curva de Perda')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_token_distribution(dataset_path=DATASET_FILE):
    data = [json.loads(l) for l in open(dataset_path, 'r', encoding='utf-8')]
    lengths = [len(d['question'].split()) for d in data]
    plt.figure()
    sns.histplot(lengths, bins=20)
    plt.xlabel('Tokens por pergunta')
    plt.title('Distribuição de tamanho das perguntas')
    plt.tight_layout()
    plt.show()


def plot_confusion(eval_path=EVAL_FILE, top_k=20):
    records = [json.loads(l) for l in open(eval_path, 'r', encoding='utf-8')]
    gold = []
    pred = []
    for r in records:
        g_tokens = r['expected'].split()
        p_tokens = r['prediction'].split()
        for g, p in zip(g_tokens, p_tokens):
            gold.append(g)
            pred.append(p)
    # Seleciona tokens mais frequentes
    tokens = pd.Series(gold + pred).value_counts().index[:top_k]
    gold_filt = [t if t in tokens else '<other>' for t in gold]
    pred_filt = [t if t in tokens else '<other>' for t in pred]
    cm = confusion_matrix(gold_filt, pred_filt, labels=list(tokens) + ['<other>'])
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, xticklabels=list(tokens)+['<other>'], yticklabels=list(tokens)+['<other>'], annot=False, cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Esperado')
    plt.title('Matriz de Confusão de Tokens')
    plt.tight_layout()
    plt.show()


def plot_embeddings(emb_path=EMB_FILE):
    data = np.load(emb_path)['vectors']
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data)
    plt.figure()
    plt.scatter(proj[:,0], proj[:,1], s=10, alpha=0.6)
    plt.title('Projeção 2D dos Embeddings')
    plt.tight_layout()
    plt.show()


def show_examples(eval_path=EVAL_FILE, n=5):
    records = [json.loads(l) for l in open(eval_path, 'r', encoding='utf-8')]
    for r in records[:n]:
        print('Q:', r['question'])
        print('GT:', r['expected'])
        print('Pred:', r['prediction'])
        print('-'*20)


if __name__ == '__main__':
    plot_loss_curve()
    plot_token_distribution()
    plot_confusion()
    plot_embeddings()
    show_examples()
