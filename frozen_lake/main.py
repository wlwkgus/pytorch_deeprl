import argparse

from trainer import ConcreteTrainer

parser = argparse.ArgumentParser(description='template')

parser.add_argument('--is_train', type=bool, default=True, metavar='N', help='is train')

args = parser.parse_args()

# Deep Q Learning of Frozen Lake.
# TODO : Add Double Q Learning.


def main():
    if args.is_train:
        t = ConcreteTrainer()
        t.run(epochs=1)
    else:
        pass

if __name__ == '__main__':
    main()
