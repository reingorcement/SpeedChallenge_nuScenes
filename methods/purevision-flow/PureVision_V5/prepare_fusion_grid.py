"""Extract the authorized 4x6 pilot directly from frozen maps, verifying 2x3 reuse."""
import argparse
from purevision_v5.fusion_grid_cache import extract_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'manifest', 'selection', 'flow', 'rgb-frames', 'reference-cache', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--grid', nargs=2, type=int, default=[4, 6])
    parser.add_argument('--scope', choices=['preflight', 'pilot'], default='preflight')
    parser.add_argument('--allow-pilot-cache', action='store_true')
    parser.add_argument('--train-samples', type=int, default=2)
    parser.add_argument('--dev-samples', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    print(extract_cache(parser.parse_args())['identity'])


if __name__ == '__main__':
    main()
