# io module?
import argparse
import torch


def create_argparser():
    """Creates argument parser and parses the provided arguments.

    Returns
    -------
    argparse.ArgumentParser
        argument parser for DNNs.
    """
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser


parser = create_argparser()

BATCH_SIZE = 16
EPOCHS = 600

def get_args(batch_size=BATCH_SIZE, epochs=EPOCHS, log_interval=10, no_cuda=False):
    """Helper function to create arg."""

    args = ['--batch-size', str(batch_size), 
            '--seed', '43', 
            '--epochs', str(epochs), 
            '--log-interval', str(log_interval)]
    if no_cuda:
        args.append('--no-cuda')
    args = parser.parse_args(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    return args

