import sys
from typing import TextIO


def main(f: TextIO):
    f.readline()

    for line in f:
        k, d0, d1 = [int(raw) for raw in line.split()]


if __name__ == "__main__":
    main(sys.stdin)
