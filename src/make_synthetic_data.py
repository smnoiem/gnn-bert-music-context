from __future__ import annotations

import argparse, json
import logging
from pathlib import Path
import numpy as np
import torch

from .utils import configure_logging, progress


LABELS=["rock","jazz","electronic","calm","energetic","melancholic"]
WORDS={"rock":"distorted guitar driving drums","jazz":"swing piano brass improvisation","electronic":"synth bass electronic beat","calm":"soft ambient gentle slow","energetic":"fast energetic dance rhythm","melancholic":"sad reflective minor mood"}

def main():
 configure_logging()
 p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=24); p.add_argument("--output",default="data/processed"); a=p.parse_args(); rng=np.random.default_rng(42); root=Path(a.output); graphs=root/"graphs"; graphs.mkdir(parents=True,exist_ok=True); rows=[]
 logger = logging.getLogger("music-context")
 logger.info("Generating %d synthetic graph samples in %s", a.count, root)
 for i in progress(range(a.count), desc="Generating graphs", total=a.count):
  tags=[LABELS[i%3],LABELS[3+i%3]]; n=4+i%5; x=torch.tensor(rng.normal(size=(n,32)),dtype=torch.float32); edge=[]
  for j in range(n): edge.extend([(j,j),(j,max(0,j-1)),(j,min(n-1,j+1))])
  path=graphs/f"synthetic_{i:03d}.pt"; torch.save({"x":x,"edge_index":torch.tensor(edge,dtype=torch.long).T},path)
  split="train" if i<int(a.count*.65) else "val" if i<int(a.count*.8) else "test"; rows.append({"track_id":f"synthetic_{i:03d}","graph":str(path),"text":" ".join(WORDS[t] for t in tags),"labels":"|".join(tags),"split":split,"artist_id":f"artist_{i}"})
 with open(root/"manifest.jsonl","w") as f:
  for row in rows: f.write(json.dumps(row)+"\n")
 logger.info("Wrote %d graph samples and %s", a.count, root / "manifest.jsonl")
if __name__=="__main__": main()
