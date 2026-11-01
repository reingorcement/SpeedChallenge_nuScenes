"""Explicit bounded preflight or later authorized common pilot-cache extraction."""
import argparse
from purevision_v5.fusion_abc_cache import extract_cache

def main():
    p=argparse.ArgumentParser()
    for k in ['checkpoint','manifest','selection','flow','rgb-frames','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--scope',choices=['preflight','pilot'],default='preflight')
    p.add_argument('--allow-pilot-cache',action='store_true')
    p.add_argument('--train-samples',type=int,default=8);p.add_argument('--dev-samples',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if min(args.train_samples,args.dev_samples,args.batch_size,args.threads)<1:p.error('Counts must be positive')
    print(extract_cache(args)['identity'])

if __name__=='__main__':main()
