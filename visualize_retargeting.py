"""Display glove, robot and predicted fingertip point clouds.
Author: Chenxi Wang, Ying Feng
"""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    from anydexrt.config import load_config
    from anydexrt.inference.retargeting import load_model, visualize_predictions
    model = load_model(load_config(args.config), args.checkpoint, args.device)
    visualize_predictions(model, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
