"""Prepare the single full-data 2x3 cache, preserving the original pilot paths."""
import argparse
from purevision_v5.fusion_full_cache import preflight,extract_full


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','manifest','flow','rgb-frames','reference-cache','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--scope',choices=['preflight','full'],default='preflight')
    p.add_argument('--allow-full-cache',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-receipt');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--shard-size',type=int,default=4096);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.threads<1:p.error('threads must be positive')
    if a.scope=='full' and not a.preflight_receipt:p.error('--full requires --preflight-receipt')
    value=preflight(a) if a.scope=='preflight' else extract_full(a)
    print(value.get('identity',value.get('output_cache_identity')),flush=True)


if __name__=='__main__':main()
