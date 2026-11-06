"""Visualize live glove retargeting with optional contact correction.
Author: Chenxi Wang, Ying Feng
"""
import argparse
from pathlib import Path
from anydexrt.arguments import add_glove_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--contact-checkpoint', type=Path)
    parser.add_argument('--contact-data-dir', type=Path)
    parser.add_argument('--contact-threshold', type=float, default=0.5,
                        help='Confidence fusion cutoff in [0, 1)')
    add_glove_arguments(parser, required=True)
    args = parser.parse_args(argv)
    if args.contact_checkpoint and not args.contact_data_dir:
        parser.error('--contact-data-dir is required with --contact-checkpoint')
    if not 0 <= args.contact_threshold < 1:
        parser.error('--contact-threshold must be in [0, 1)')
    from anydexrt.config import load_config
    from anydexrt.inference.retargeting import load_model, live
    config = load_config(args.config)
    model = load_model(config, args.checkpoint, args.device)
    contact_model = None
    if args.contact_checkpoint:
        import torch
        from anydexrt.models.contact import ContactClassifier
        contact_model = ContactClassifier().to(args.device)
        contact_model.load_state_dict(torch.load(args.contact_checkpoint, map_location=args.device, weights_only=True))
        contact_model.eval()
    live(config, model, args.data_dir, args.glove_serial, args.address,
         args.hand_type, contact_model, args.contact_data_dir,
         contact_threshold=args.contact_threshold)


if __name__ == '__main__':
    main()
