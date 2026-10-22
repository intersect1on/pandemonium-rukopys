import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['gpu', 'mps', 'cuda', 'cpu'], default='gpu')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
if args.epochs <= 0:
    parser.error('epochs must be positive')
from cv import train_detector
train_detector(vars(args))
