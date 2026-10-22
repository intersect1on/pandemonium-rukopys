import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODELS = ['rf-nano', 'rf-small', 'rtv2-r18', 'dfine-small', 'deimv2-nano', 'yolo26s']
parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
parser.add_argument('--minutes', type=float, default=50)
parser.add_argument('--device', default='gpu', choices=['gpu', 'mps', 'cuda', 'cpu'])
parser.add_argument('--resume-existing', action='store_true')
parser.add_argument('--evaluate-only', action='store_true')
parser.add_argument('--report', action='store_true')
args = parser.parse_args()
if args.minutes <= 0:
    parser.error('minutes must be positive')
if args.report:
    from detectors import write_comparison_report
    print(write_comparison_report().round(4).to_string(index=False))
elif len(args.models) > 1:
    failures = []
    for name in args.models:
        command = [sys.executable, __file__, '--models', name, '--minutes', str(args.minutes), '--device', args.device]
        command += [flag for enabled, flag in ((args.resume_existing, '--resume-existing'),
                    (args.evaluate_only, '--evaluate-only')) if enabled]
        if subprocess.run(command, cwd=ROOT).returncode:
            failures.append(name)
    print('Failed candidates:', failures, flush=True)
    sys.exit(bool(failures))
else:
    from detectors import evaluate_variant, train_budget, train_yolo
    name = args.models[0]
    directory = ROOT / 'models' / f'benchmark-{name}'
    state = directory / ('weights/training_state.pt' if name == 'yolo26s' else 'last/state.pt')
    config = dict(model=name, minutes=args.minutes, device=args.device, size=640, seed=42,
                  train_pages=None, val_pages=None, pilot=False, resume=args.resume_existing and state.exists())
    print(f'{name}: {"resume" if config["resume"] else "pretrained weights"}; {args.minutes:g} cumulative minutes', flush=True)
    if args.evaluate_only:
        evaluate_variant(config)
    else:
        (train_yolo if name == 'yolo26s' else train_budget)(config)
