"""Train the contact model on CUDA.
Author: Chenxi Wang, Ying Feng
"""
import argparse
from anydexrt.arguments import add_training_arguments, prepare_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, epochs=20)
    parser.add_argument('--samples', type=int, default=1000000)
    args = parser.parse_args(argv)
    prepare_training(parser, args)
    from anydexrt.models.contact import ContactClassifier
    from anydexrt.dataset.contact import ContactDataset
    model = ContactClassifier()
    dataset = ContactDataset(args.data_dir, max_capacity=args.samples)
    from anydexrt.training.contact import train
    train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size,
          save_ckpt=args.checkpoint, num_workers=args.workers)


if __name__ == '__main__':
    main()
