import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quatsch", help="increase output verbosity")
args = parser.parse_args()
if args.quatsch:
    print(f"verbosity turned on. {args.quatsch}")
