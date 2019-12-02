import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    chars = set()

    for path in args.paths:
        with open(path) as file:
            for line in file:
                for char in line.strip():
                    chars.add(char)

    print(chars)
