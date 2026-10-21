import argparse
from pathlib import Path
from purevision_v4.evaluation import evaluate, macro_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate V4 body-reference predictions with independent labels")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    report = evaluate(args.predictions, args.truth, args.output, args.methods)
    print(report.query("subset == 'all' and population == 'own_valid'").to_string(index=False))


if __name__ == "__main__":
    main()
