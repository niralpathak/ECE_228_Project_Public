import os
import argparse
from net_custom_2 import Network_Custom_2

def main():
    parser = argparse.ArgumentParser(description="Run keras")
    parser.add_argument("--batch", type=int, required=True, \
        help="batch size")
    parser.add_argument("--epochs", type=int, required=True, \
        help="number of epochs for train")
    parser.add_argument("--init_lr", type=float, default = 1e-3, \
        help="initial learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint",\
        help="path to checkpoint directory")
    parser.add_argument("--training_imgs", type=str, required=True,\
        help="path to training images directory")
    parser.add_argument("--training_labels", type=str, required=True,\
        help="path to training labels directory")
    parser.add_argument("--testing_imgs", type=str, required=True,\
        help="path to testing images directory")
    parser.add_argument("--sample_submission", type=str, required=True,\
        help="path to sample_submission")
    parser.add_argument("--patience", type=int, required=True,\
        help="patience for decreases in validation loss")

    args = vars(parser.parse_args())
    solver = Network_Custom_2(args)
    solver.train()
    solver.test()

if __name__ == "__main__":
    main()