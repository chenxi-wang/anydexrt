"""Train the retargeting model on CUDA.
Author: Chenxi Wang, Ying Feng
"""
import argparse
from pathlib import Path
from anydexrt.arguments import add_training_arguments, prepare_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, epochs=5)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args(argv)
    prepare_training(parser, args)
    from anydexrt.config import load_config, hand_spec
    from anydexrt.models.retargeting import MappingModel
    from anydexrt.dataset.retargeting import HandGlovePairDataset
    config = load_config(args.config)
    model = MappingModel(config['finger_num'])
    dataset = HandGlovePairDataset(args.data_dir, hand_spec(config), num_repeat=args.repeats)
    from anydexrt.training.retargeting import train
    train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size,
          save_ckpt=args.checkpoint, num_workers=args.workers)


if __name__ == '__main__':
    main()
