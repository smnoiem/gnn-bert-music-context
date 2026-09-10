from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml

from .bert_encoder import BertTagClassifier
from .data import MusicGraphDataset
from .fusion_model import FusionModel
from .gnn_model import GNNClassifier
from .metrics import multilabel_metrics
from .utils import ensure_dir, save_json, seed_everything


def device_graph(graph, device): return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in graph.items()}

def run_epoch(model, data, optimizer, task, device):
    training = optimizer is not None; model.train(training); criterion = nn.BCEWithLogitsLoss(); losses=[]; logits=[]; targets=[]
    for graph in data:
        graph = device_graph(graph, device); y = graph["y"].unsqueeze(0).to(device)
        if task == "bert": prediction = model([graph["text"]])
        elif task == "gnn": prediction = model(graph).unsqueeze(0)
        else: prediction = model(graph, graph["text"])["logits"]
        loss = criterion(prediction, y)
        if task == "fusion" and "emotion" in graph:
            output = model(graph, graph["text"]); loss = loss + .1 * nn.functional.mse_loss(output["emotion"], graph["emotion"].unsqueeze(0))
        if training: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item()); logits.append(prediction.detach().cpu().numpy()[0]); targets.append(y.cpu().numpy()[0])
    return float(np.mean(losses)), multilabel_metrics(np.asarray(logits), np.asarray(targets))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task", choices=["bert","gnn","fusion"], required=True); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--manifest", default="data/processed/manifest.jsonl"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--epochs", type=int); ap.add_argument("--early-concat", action="store_true"); ap.add_argument("--run-name"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_everything(cfg["seed"])
    if args.synthetic: args.manifest="data/processed/manifest.jsonl"
    train=MusicGraphDataset(args.manifest, "train"); val=MusicGraphDataset(args.manifest, "val", train.vocab)
    num_labels=len(train.vocab); model_args=dict(num_labels=num_labels, text_hidden=cfg["model"]["text_hidden"], graph_hidden=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"], dropout=cfg["model"]["dropout"], model_name=cfg["model"]["text_model"], freeze=cfg["training"]["freeze_text_encoder"])
    if args.task == "bert": model=BertTagClassifier(num_labels, hidden_size=cfg["model"]["text_hidden"], model_name=cfg["model"]["text_model"], freeze=cfg["training"]["freeze_text_encoder"], local_files_only=args.synthetic)
    elif args.task == "gnn": model=GNNClassifier(num_labels, hidden_dim=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"])
    else: model=FusionModel(**model_args, cross_attention=not args.early_concat, local_files_only=args.synthetic)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    results=ensure_dir("results"); run_name=args.run_name or args.task; history=[]; best=-1
    for epoch in range(args.epochs or cfg["training"]["epochs"]):
        tl,tm=run_epoch(model, train, opt, args.task, device); vl,vm=run_epoch(model, val, None, args.task, device); row={"epoch":epoch+1,"train_loss":tl,"val_loss":vl,**{f"train_{k}":v for k,v in tm.items()},**{f"val_{k}":v for k,v in vm.items()}}; history.append(row); print(row)
        if vm["macro_f1"] > best: best=vm["macro_f1"]; torch.save({"model":model.state_dict(),"vocab":train.vocab,"config":cfg,"task":args.task,"early_concat":args.early_concat}, results / f"{run_name}_best.pt")
    save_json({"task":args.task,"labels":train.vocab,"history":history}, results / f"{run_name}_metrics.json")
    # Required learning curves: metrics are also retained as JSON for the report table.
    import matplotlib.pyplot as plt
    epochs = [x["epoch"] for x in history]
    plt.figure(figsize=(6, 4)); plt.plot(epochs, [x["train_macro_f1"] for x in history], label="train macro-F1"); plt.plot(epochs, [x["val_macro_f1"] for x in history], label="validation macro-F1"); plt.plot(epochs, [x["val_micro_f1"] for x in history], label="validation micro-F1"); plt.xlabel("epoch"); plt.ylabel("F1"); plt.legend(); plt.tight_layout(); plt.savefig(results / f"{run_name}_f1_curve.png", dpi=160); plt.close()

if __name__ == "__main__": main()
