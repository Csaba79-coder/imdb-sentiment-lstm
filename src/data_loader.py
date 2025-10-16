"""
Data loader module for IMDB Sentiment Analysis.
Handles loading, validation, cleaning, and splitting of the dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH_CLEANED, VALIDATION_SPLIT, RANDOM_SEED


def check_data_exists():
    """
    Check if the dataset file exists.

    Returns:
        bool: True if file exists, False otherwise
    """
    if not os.path.exists(DDATA_PATH_CLEANED):
        print(f"❌ Error: Dataset not found at {DATA_PATH_CLEANED}")
        print("Please download the IMDB dataset from Kaggle and place it in the data/ folder.")
        return False
    print(f"✅ Dataset found at {DDATA_PATH_CLEANED}")
    return True


def clean_and_prepare_csv():
    """
    Clean the malformed CSV file and create a properly formatted version.

    Problem: The original CSV has reviews containing commas, making standard
    CSV parsing fail. Reviews and sentiments are in one cell.

    Solution: Split each line by the LAST comma, since sentiment (positive/negative)
    is always the last field after the final comma.

    Returns:
        str: Path to the cleaned CSV file
        None: If cleaning fails
    """
    print("🔧 Cleaning malformed CSV...")

    # Output path for cleaned file
    cleaned_path = DATA_PATH.replace('.csv', '_cleaned.csv')

    try:
        # Read the raw file line by line
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Prepare cleaned data storage
        cleaned_rows = []

        # Add proper header
        cleaned_rows.append(['review', 'sentiment'])

        # Process each data row (skip first line which is header)
        skipped = 0
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()

            # Skip empty lines
            if not line:
                skipped += 1
                continue

            # Find the LAST comma - that separates review from sentiment
            # Example: "Great movie, loved it, highly recommend,positive"
            #          review: "Great movie, loved it, highly recommend"
            #          sentiment: "positive"
            last_comma_idx = line.rfind(',')

            if last_comma_idx == -1:  # No comma found - invalid line
                skipped += 1
                continue

            # Split at the last comma
            review = line[:last_comma_idx].strip()
            sentiment = line[last_comma_idx + 1:].strip()

            # Validate sentiment - must be 'positive' or 'negative'
            if sentiment not in ['positive', 'negative']:
                skipped += 1
                continue

            # Remove any leading/trailing quotes from review
            review = review.strip('"').strip("'")

            # Add cleaned row
            cleaned_rows.append([review, sentiment])

        # Create DataFrame from cleaned data
        df_cleaned = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])

        # Write cleaned CSV
        print(f"  Writing cleaned CSV to: {cleaned_path}")
        df_cleaned.to_csv(cleaned_path, index=False, encoding='utf-8', quoting=1)  # quoting=1 means QUOTE_ALL

        print(f"✅ CSV cleaned successfully!")
        print(f"   Original rows: {len(lines) - 1}")
        print(f"   Cleaned rows: {len(df_cleaned)}")
        print(f"   Skipped rows: {skipped}")

        return cleaned_path

    except Exception as e:
        print(f"❌ Error cleaning CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_data():
    """
    Load the IMDB dataset from CSV file.
    If the original CSV is malformed, automatically clean it first.

    Returns:
        pd.DataFrame: Cleaned dataframe with 'review' and 'sentiment' columns
        None: If file doesn't exist or loading fails
    """
    # Check if file exists
    if not check_data_exists():
        return None

    try:
        print("Loading dataset...")

        # Attempt to load the original CSV
        try:
            df = pd.read_csv(DATA_PATH, encoding='utf-8')

            # Check if CSV is malformed (too many columns due to commas in text)
            if len(df.columns) > 2:
                print("  ⚠️ CSV appears malformed (too many columns detected)")
                print("  🔧 Attempting to clean and reformat...")

                # Clean the CSV
                cleaned_path = clean_and_prepare_csv()
                if cleaned_path is None:
                    return None

                # Load the cleaned CSV
                df = pd.read_csv(cleaned_path, encoding='utf-8')

        except pd.errors.ParserError as pe:
            # CSV parsing failed - malformed file
            print(f"  ⚠️ CSV parsing error: {pe}")
            print("  🔧 Attempting to clean and reformat...")

            # Clean the CSV
            cleaned_path = clean_and_prepare_csv()
            if cleaned_path is None:
                return None

            # Load the cleaned CSV
            df = pd.read_csv(cleaned_path, encoding='utf-8')

        # Verify we have the correct columns
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            print(f"❌ Expected columns 'review' and 'sentiment', got: {df.columns.tolist()}")
            return None

        # Keep only necessary columns
        df = df[['review', 'sentiment']]

        # Remove rows with missing values
        print("  Removing missing values...")
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        if rows_before != rows_after:
            print(f"  Removed {rows_before - rows_after} rows with missing data")

        # Remove duplicate rows
        print("  Removing duplicates...")
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        if duplicates > 0:
            print(f"  Removed {duplicates} duplicate rows")

        # Reset index to have clean sequential numbering
        df = df.reset_index(drop=True)

        print(f"✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(df)}")

        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def explore_data(df):
    """
    Print comprehensive information about the dataset.
    Includes: sample rows, shape, missing values, sentiment distribution,
    and review length statistics.

    Args:
        df (pd.DataFrame): The dataset to explore
    """
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    # Display first few rows
    print("\nFirst 3 samples:")
    print(df.head(3))

    # Display dataset info
    print("\nDataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Sentiment distribution (class balance)
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())

    # Calculate review length statistics (number of words)
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    print("\nReview length statistics (words):")
    print(df['review_length'].describe())

    print("=" * 60 + "\n")


def prepare_data(df):
    """
    Prepare data for training: extract features and convert labels to binary.

    Args:
        df (pd.DataFrame): The dataset

    Returns:
        tuple: (reviews, labels)
               - reviews: array of review texts
               - labels: binary array (1=positive, 0=negative)
    """
    # Extract reviews (feature/input data)
    reviews = df['review'].values

    # Convert sentiment strings to binary labels
    # positive -> 1, negative -> 0
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).values

    # Display preparation results
    print(f"✅ Data prepared:")
    print(f"   Reviews: {len(reviews)}")
    print(f"   Positive: {np.sum(labels)} ({np.sum(labels) / len(labels) * 100:.1f}%)")
    print(f"   Negative: {len(labels) - np.sum(labels)} ({(len(labels) - np.sum(labels)) / len(labels) * 100:.1f}%)")

    return reviews, labels


def split_data(reviews, labels):
    """
    Split data into training and validation sets.
    Uses stratified split to maintain class balance in both sets.

    Args:
        reviews (np.array): Review texts
        labels (np.array): Sentiment labels (binary)

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        reviews,
        labels,
        test_size=VALIDATION_SPLIT,  # Percentage for validation (from config)
        random_state=RANDOM_SEED,  # For reproducibility
        stratify=labels  # Keep same positive/negative ratio in both sets
    )

    print(f"\n✅ Data split complete:")
    print(f"   Training samples: {len(X_train)} ({(1 - VALIDATION_SPLIT) * 100:.0f}%)")
    print(f"   Validation samples: {len(X_val)} ({VALIDATION_SPLIT * 100:.0f}%)")

    return X_train, X_val, y_train, y_val


def load_and_prepare_data():
    """
    Main pipeline function: loads, cleans, explores, and splits the data.
    This is the primary function called from main.py.

    Pipeline steps:
    1. Load data (with automatic cleaning if needed)
    2. Explore data (print statistics)
    3. Prepare features and labels
    4. Split into train/validation sets

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
               - X_train: training reviews
               - X_val: validation reviews
               - y_train: training labels
               - y_val: validation labels
        None: if any step fails
    """
    print("\n" + "🚀 Starting data loading and preparation...\n")

    # Step 1: Load data
    df = load_data()
    if df is None:
        return None

    # Step 2: Explore data (print overview)
    explore_data(df)

    # Step 3: Prepare features and labels
    reviews, labels = prepare_data(df)

    # Step 4: Split into train/validation sets
    X_train, X_val, y_train, y_val = split_data(reviews, labels)

    print("\n✅ Data loading complete!\n")

    return X_train, X_val, y_train, y_val


# Test the module if run directly
if __name__ == "__main__":
    """
    Test block - runs only when this file is executed directly.
    Usage: python src/data_loader.py
    """
    print("Testing data_loader module...")
    result = load_and_prepare_data()

    if result is not None:
        X_train, X_val, y_train, y_val = result
        print("\n✅ Test successful!")
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")