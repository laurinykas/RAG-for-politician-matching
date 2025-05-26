import json
import re
import os
import argparse
import random
import math
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_splitter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> list:
    """Loads data from a JSON file, expecting a list."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.warning(f"Data in {file_path} is not a list. Wrapping it in a list.")
                return [data]
            return data
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return []


def save_json_file(data: list, file_path: str):
    """Saves data to a JSON file."""
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved data to {file_path} ({len(data)} items)")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}", exc_info=True)


def get_keywords_from_arg(keywords_arg: str) -> list:
    """Parses keywords from command-line argument (comma-separated or file path)."""
    if not keywords_arg:
        return []
    # Convert to lowercase immediately for consistent comparison later
    if os.path.isfile(keywords_arg):
        try:
            with open(keywords_arg, 'r', encoding='utf-8') as f:
                return [line.strip().lower() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading keywords from file {keywords_arg}: {e}")
            return []
    else:
        return [kw.strip().lower() for kw in keywords_arg.split(',') if kw.strip()]


def split_data(articles: list, keywords_to_stratify: list,
               train_ratio: float, val_ratio: float) -> tuple[list, list, list]:
    """
    Splits articles into training, validation, and testing sets.
    Articles with exactly one keyword from keywords_to_stratify (found in their 'anonymized.keywords' field)
    are split according to train_ratio and val_ratio per keyword.
    Other articles are pooled and split randomly according to the same ratios.
    The test set gets the remainder.
    """
    if not articles:
        return [], [], []

    single_keyword_articles = defaultdict(list)
    other_articles = []  # Articles with 0 or >1 of the specified keywords

    logger.info(
        f"Categorizing {len(articles)} articles based on {len(keywords_to_stratify)} predefined keywords found in article's 'keywords' field.")

    for article in articles:
        article_keywords_field = []
        if isinstance(article, dict):
            article_keywords_field = article.get("anonymized", {}).get("keywords", [])
            if not isinstance(article_keywords_field, list):
                logger.warning(
                    f"Article's 'keywords' field is not a list: {article_keywords_field}. Treating as no keywords for this article.")
                article_keywords_field = []
            article_keywords_lower = {str(kw).lower() for kw in article_keywords_field if kw}
        else:
            logger.warning(f"Encountered non-dict article: {type(article)}. Adding to 'other_articles'.")
            other_articles.append(article)
            continue

        found_stratification_kws = article_keywords_lower.intersection(set(keywords_to_stratify))

        if len(found_stratification_kws) == 1:
            keyword = list(found_stratification_kws)[0]
            single_keyword_articles[keyword].append(article)
        else:
            other_articles.append(article)

    train_set = []
    validation_set = []
    test_set = []

    # Helper function to split a list of items
    def _split_group(items: list, group_name: str):
        if not items:
            return

        num_total = len(items)
        random.shuffle(items)

        # Calculate split indices
        # Train set gets floor(num_total * train_ratio)
        # Validation set gets floor(num_total * (train_ratio + val_ratio)) - train_set_size
        # Test set gets the rest
        split_idx_train = int(num_total * train_ratio)
        split_idx_val = int(num_total * (train_ratio + val_ratio))

        current_train = items[:split_idx_train]
        current_val = items[split_idx_train:split_idx_val]
        current_test = items[split_idx_val:]

        train_set.extend(current_train)
        validation_set.extend(current_val)
        test_set.extend(current_test)

        logger.info(
            f"    Split for '{group_name}': "
            f"{len(current_train)} train, {len(current_val)} validation, {len(current_test)} test "
            f"(Total: {num_total})"
        )

    logger.info("Splitting single-keyword articles (stratified)...")
    for keyword, kw_articles in single_keyword_articles.items():
        logger.info(f"  Keyword '{keyword}': {len(kw_articles)} articles.")
        _split_group(kw_articles, f"Keyword: {keyword}")

    logger.info(f"Splitting other articles ({len(other_articles)} articles) randomly...")
    if other_articles:
        _split_group(other_articles, "Other articles")

    # Final shuffle of the combined sets
    random.shuffle(train_set)
    random.shuffle(validation_set)
    random.shuffle(test_set)

    logger.info(f"Total training articles: {len(train_set)}")
    logger.info(f"Total validation articles: {len(validation_set)}")
    logger.info(f"Total testing articles: {len(test_set)}")

    return train_set, validation_set, test_set


def main():
    parser = argparse.ArgumentParser(
        description="Split JSON data into training, validation, and testing sets based on keywords found in article's 'keywords' field.")
    parser.add_argument("--input-file", required=True,
                        help="Path to the input JSON file containing a list of articles (each with an 'anonymized.keywords' list).")
    parser.add_argument("--keywords", required=True,
                        help="Comma-separated list of specific keywords OR path to a text file with one keyword per line. These are the keywords to stratify by if found in an article's 'keywords' field.")
    parser.add_argument("--train-output-file", required=True,
                        help="Path to save the training set JSON file.")
    parser.add_argument("--validation-output-file", required=True,
                        help="Path to save the validation set JSON file.")
    parser.add_argument("--test-output-file", required=True,
                        help="Path to save the testing set JSON file.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Proportion of data to allocate to the training set (e.g., 0.8 for 80%). Default is 0.8.")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Proportion of data to allocate to the validation set (e.g., 0.1 for 10%). Default is 0.1. Test set gets the remainder.")

    args = parser.parse_args()

    if not (0 < args.train_ratio < 1):
        logger.error("Train ratio must be between 0 (exclusive) and 1 (exclusive).")
        return
    if not (0 <= args.val_ratio < 1): # val_ratio can be 0
        logger.error("Validation ratio must be between 0 (inclusive) and 1 (exclusive).")
        return
    if args.train_ratio + args.val_ratio >= 1.0:
        logger.error(f"The sum of train_ratio ({args.train_ratio}) and val_ratio ({args.val_ratio}) must be less than 1.0.")
        return

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    logger.info(f"Starting dataset splitting process with ratios: "
                f"Train={args.train_ratio:.2f}, Validation={args.val_ratio:.2f}, Test={test_ratio:.2f}")
    logger.info(f"Input data file: {args.input_file}")

    articles = load_json_file(args.input_file)
    if not articles:
        logger.error("No articles loaded. Exiting.")
        return

    keywords_for_stratification = get_keywords_from_arg(args.keywords)
    if not keywords_for_stratification:
        logger.warning(
            "No keywords provided for stratification via --keywords. Will perform a simple random split of all data based on specified ratios.")
    else:
        logger.info(f"Keywords for stratification (from --keywords argument): {keywords_for_stratification}")

    train_data, val_data, test_data = split_data(
        articles, keywords_for_stratification, args.train_ratio, args.val_ratio
    )

    save_json_file(train_data, args.train_output_file)
    save_json_file(val_data, args.validation_output_file)
    save_json_file(test_data, args.test_output_file)

    logger.info("Dataset splitting complete.")


if __name__ == "__main__":
    # Set a fixed seed for reproducibility of shuffles if needed, e.g., for debugging.
    random.seed(42)
    main()