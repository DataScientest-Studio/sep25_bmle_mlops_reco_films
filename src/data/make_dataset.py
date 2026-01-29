import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

# ✅ build_features est dans src/features
from src.features.build_features import build_user_movie_matrices_from_db


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_filepath", type=click.Path(file_okay=False))
def main(input_filepath, output_filepath):
    """
    Build content-based recommendation datasets from a SQLite database.
    """
    logger = logging.getLogger(__name__)

    db_path = Path(input_filepath)
    out_dir = Path(output_filepath)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using SQLite DB: %s", db_path)
    logger.info("Output directory: %s", out_dir)

    # Orchestration only: the heavy lifting is in build_features.py
    build_user_movie_matrices_from_db(
        db_path=db_path,
        out_dir=out_dir,
        ratings_table="raw_ratings",
        movies_table="raw_movies",
        logger=logger,
    )

    logger.info("✅ Dataset creation finished successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load .env if present (paths, configs, etc.)
    load_dotenv(find_dotenv())

    main()
