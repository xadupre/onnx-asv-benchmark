import argparse
from pathlib import Path

from result_shards import merge_shards


def main():
    parser = argparse.ArgumentParser(
        description="Merge independently published ASV result shards."
    )
    parser.add_argument("source", type=Path, help="Directory containing the shards.")
    parser.add_argument("destination", type=Path, help="Merged ASV results directory.")
    args = parser.parse_args()
    merge_shards(args.source, args.destination)


if __name__ == "__main__":
    main()
