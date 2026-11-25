import argparse
from .data import save
p=argparse.ArgumentParser(); p.add_argument("--samples",type=int,default=1200); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
print("Created",save(n=a.samples,seed=a.seed))
