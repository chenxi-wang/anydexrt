"""Collect robot references, glove motion, anchors, or contact samples.
Author: Chenxi Wang, Ying Feng
"""
import argparse
from pathlib import Path
from anydexrt.arguments import add_glove_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['robot', 'motion', 'anchors', 'contact'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    add_glove_arguments(parser)
    args = parser.parse_args(argv)
    from anydexrt.config import load_config
    from anydexrt.collection.pipeline import collect
    collect(load_config(args.config), args.data_root, args.stage,
            args.glove_serial, args.address, args.hand_type)


if __name__ == '__main__':
    main()
