from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from .bert_encoder import BertTagClassifier
from .fusion_model import FusionModel
from .gnn_model import GNNClassifier
from .metrics import multilabel_metrics
from .train import MusicGraphDataset, device_graph
from .utils import ensure_dir, save_json


def load_model(checkpoint, device):
    state=torch.load(checkpoint, map_location=device, weights_only=False); cfg=state["config"]; n=len(state["vocab"]); task=state["task"]
    if task=="bert": model=BertTagClassifier(n, hidden_size=cfg["model"]["text_hidden"], model_name=cfg["model"]["text_model"], local_files_only=True)
    elif task=="gnn": model=GNNClassifier(n, hidden_dim=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"])
    else: model=FusionModel(n, text_hidden=cfg["model"]["text_hidden"], graph_hidden=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"], cross_attention=not state.get("early_concat",False), local_files_only=True)
    model.load_state_dict(state["model"]); return model.to(device).eval(), state


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--manifest", default="data/processed/manifest.jsonl"); ap.add_argument("--task", choices=["bert","gnn","fusion"], required=True); args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,state=load_model(args.checkpoint,device); test=MusicGraphDataset(args.manifest,"test",state["vocab"]); logits=[]; ys=[]; embeddings=[]; cases=[]
    with torch.no_grad():
        for graph in test:
            graph=device_graph(graph,device)
            if args.task=="bert": out=model([graph["text"]]); emb=model.encoder([graph["text"]])
            elif args.task=="gnn": out=model(graph).unsqueeze(0); emb=model.encoder(graph).unsqueeze(0)
            else: r=model(graph,graph["text"],return_attention=True); out=r["logits"]; emb=r["embedding"]
            probs=torch.sigmoid(out)[0].cpu().numpy(); logits.append(out[0].cpu().numpy()); ys.append(graph["y"].cpu().numpy()); embeddings.append(emb[0].cpu().numpy())
            top=np.argsort(probs)[-3:][::-1]; cases.append({"track_id":graph["track_id"],"text":graph["text"],"predictions":[{"label":state["vocab"][i],"score":float(probs[i])} for i in top]})
    metrics=multilabel_metrics(np.asarray(logits),np.asarray(ys)); output=ensure_dir("results"); save_json(metrics,output/f"{args.task}_test_metrics.json"); save_json({"case_studies":cases[:3]},output/f"{args.task}_case_studies.json"); print(metrics)
    if len(embeddings)>=3:
        matrix = np.asarray(embeddings)
        try:
            from sklearn.manifold import TSNE
            points = TSNE(perplexity=min(5, len(matrix)-1), init="random", random_state=42).fit_transform(matrix)
            title = f"{args.task} t-SNE embeddings"
        except ImportError:
            # Keeps the demo usable before optional analysis dependencies are installed.
            _, _, vectors = np.linalg.svd(matrix - matrix.mean(0), full_matrices=False)
            points, title = (matrix - matrix.mean(0)) @ vectors[:2].T, f"{args.task} PCA embeddings (install scikit-learn for t-SNE)"
        plt.scatter(points[:,0],points[:,1],c=np.argmax(ys,axis=1)); plt.title(title); plt.savefig(output/f"{args.task}_tsne.png",dpi=160,bbox_inches="tight"); plt.close()

if __name__ == "__main__": main()
