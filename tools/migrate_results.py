import argparse
from pathlib import Path

from result_shards import migrate_legacy_results, migrate_shard_hierarchy


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy flat ASV results to anonymous shards."
    )
    parser.add_argument("results", type=Path, help="Published ASV results directory.")
    args = parser.parse_args()
    migrated = migrate_legacy_results(args.results)
    migrated.update(migrate_shard_hierarchy(args.results / "shards"))
    if migrated:
        print("\n".join(sorted(migrated)))
    else:
        print("No legacy results to migrate.")


if __name__ == "__main__":
    main()
