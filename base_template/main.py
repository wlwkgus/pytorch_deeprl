import argparse

from trainer import ConcreteTrainer

parser = argparse.ArgumentParser(description='template')

parser.add_argument('--is_train', type=bool, default=True, metavar='N', help='is train')

args = parser.parse_args()


def main():
    if args.is_train:
        t = ConcreteTrainer()
        t.run(epochs=2)
    else:
        pass

if __name__ == '__main__':
    main()
