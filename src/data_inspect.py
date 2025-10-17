"""
Data inspection module for IMDB dataset.
Shows the RAW content and creates formatted CSV.
"""

import os
import csv
import re
from src.config import DATA_PATH_ORIGINAL, DATA_PATH_FORMATTED


def clean_text(text):
    """
    Clean review text by removing HTML tags and extra whitespace.

    Args:
        text (str): Raw review text

    Returns:
        str: Cleaned review text
    """
    # Remove HTML tags (like <br />, <br>, </br>, etc.)
    # Replace with space to avoid joining words
    text = re.sub(r'<[^>]+>', ' ', text)  # ← SPACE, not empty string!

    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def inspect_csv():
    """
    Show the EXACT raw content of the CSV file.
    No truncation, no formatting - just the raw text.
    """

    print("=" * 70)
    print("CSV INSPECTION - RAW CONTENT")
    print("=" * 70)
    print(f"\nFile: {DATA_PATH_ORIGINAL}\n")

    if not os.path.exists(DATA_PATH_ORIGINAL):
        print(f"❌ File not found")
        return

    try:
        with open(DATA_PATH_ORIGINAL, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Total lines: {len(lines)}\n")
        print("=" * 70)

        # Show first 50 lines EXACTLY as they are
        for i in range(min(50, len(lines))):
            print(f"Line {i + 1}:")
            print(lines[i])  # Raw line with everything
            print("-" * 70)

    except Exception as e:
        print(f"❌ Error: {e}")


def create_formatted_csv():
    """
    Create a formatted CSV with two columns: review and sentiment.

    Process:
    1. Read original CSV line by line
    2. Split each line at the LAST comma
    3. Clean sentiment (remove ;;;;;; and quotes)
    4. Clean review (remove HTML tags, quotes, extra whitespace)
    5. Save to formatted CSV
    6. Verify by reading back and showing first 50 rows

    Returns:
        str: Path to formatted CSV file
    """

    print("\n" + "=" * 70)
    print("CREATING FORMATTED CSV")
    print("=" * 70)

    try:
        # Read original file
        print(f"\n📖 Reading: {DATA_PATH_ORIGINAL}")
        with open(DATA_PATH_ORIGINAL, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"   Total lines: {len(lines)}")

        # Prepare formatted data
        formatted_rows = []
        skipped = 0

        # Process each line (skip header)
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()

            # Skip empty lines
            if not line:
                skipped += 1
                continue

            # Find LAST comma (separator between review and sentiment)
            last_comma_idx = line.rfind(',')

            if last_comma_idx == -1:
                skipped += 1
                continue

            # Split at last comma
            review = line[:last_comma_idx]
            sentiment = line[last_comma_idx + 1:]

            # Clean sentiment:
            # 1. Remove semicolons (;;;;;;)
            sentiment = sentiment.replace(';', '').strip()

            # 2. Remove quotes from sentiment
            sentiment = sentiment.strip('"').strip("'")

            # Clean review:
            # 1. Remove surrounding quotes if present
            if review.startswith('"') and review.endswith('"'):
                review = review[1:-1]

            # 2. Remove HTML tags and clean whitespace
            review = clean_text(review)

            # Validate sentiment
            if sentiment not in ['positive', 'negative']:
                skipped += 1
                continue

            # Skip if review is empty after cleaning
            if not review:
                skipped += 1
                continue

            # Add to formatted data
            formatted_rows.append([review, sentiment])

        # Write formatted CSV
        print(f"\n💾 Writing formatted CSV: {DATA_PATH_FORMATTED}")
        with open(DATA_PATH_FORMATTED, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['review', 'sentiment'])  # Header
            writer.writerows(formatted_rows)

        print(f"✅ Formatted CSV created!")
        print(f"   Total rows written: {len(formatted_rows)}")
        print(f"   Skipped rows: {skipped}")

        # Verify by reading back
        print(f"\n" + "=" * 70)
        print("VERIFICATION - FIRST 50 ROWS FROM FORMATTED CSV")
        print("=" * 70 + "\n")

        with open(DATA_PATH_FORMATTED, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Header: {header}\n")
            print("-" * 70)

            for i, row in enumerate(reader, start=1):
                if i > 50:
                    break

                if len(row) == 2:
                    review, sentiment = row
                    # Show first 100 chars of review
                    review_preview = review[:100] + "..." if len(review) > 100 else review
                    print(f"Row {i}:")
                    print(f"  Review: {review_preview}")
                    print(f"  Sentiment: {sentiment}")
                    print("-" * 70)

        print(f"\n✅ Verification complete!")
        return DATA_PATH_FORMATTED

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_files():
    """
    Compare original and formatted CSV files.
    Shows line counts and statistics for both files.
    """

    print("\n" + "=" * 70)
    print("FILE COMPARISON")
    print("=" * 70)

    try:
        # Check if original file exists
        if not os.path.exists(DATA_PATH_ORIGINAL):
            print(f"❌ Original file not found: {DATA_PATH_ORIGINAL}")
            return

        # Check if formatted file exists
        if not os.path.exists(DATA_PATH_FORMATTED):
            print(f"❌ Formatted file not found: {DATA_PATH_FORMATTED}")
            return

        # Count lines in original file
        print(f"\n📄 ORIGINAL FILE: {DATA_PATH_ORIGINAL}")
        with open(DATA_PATH_ORIGINAL, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        original_total = len(original_lines)
        original_data = original_total - 1  # Exclude header

        print(f"   Total lines: {original_total}")
        print(f"   Header: 1")
        print(f"   Data rows: {original_data}")

        # Count lines in formatted file
        print(f"\n📄 FORMATTED FILE: {DATA_PATH_FORMATTED}")
        with open(DATA_PATH_FORMATTED, 'r', encoding='utf-8') as f:
            formatted_lines = f.readlines()

        formatted_total = len(formatted_lines)
        formatted_data = formatted_total - 1  # Exclude header

        print(f"   Total lines: {formatted_total}")
        print(f"   Header: 1")
        print(f"   Data rows: {formatted_data}")

        # Calculate difference
        print(f"\n📊 COMPARISON:")
        difference = original_data - formatted_data
        percentage = (formatted_data / original_data * 100) if original_data > 0 else 0

        print(f"   Original data rows: {original_data}")
        print(f"   Formatted data rows: {formatted_data}")
        print(f"   Difference: {difference} rows")
        print(f"   Retention rate: {percentage:.2f}%")

        if difference > 0:
            print(f"\n⚠️  {difference} rows were skipped during formatting")
        elif difference == 0:
            print(f"\n✅ All rows successfully formatted!")
        else:
            print(f"\n⚠️  WARNING: More rows in formatted than original!")

        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run inspection and formatting pipeline.
    """
    # Step 1: Inspect original CSV
    print("STEP 1: Inspecting original CSV\n")
    inspect_csv()

    # Step 2: Create formatted CSV
    print("\n\nSTEP 2: Creating formatted CSV\n")
    formatted_path = create_formatted_csv()

    # Step 3: Compare files
    if formatted_path:
        print("\n\nSTEP 3: Comparing files\n")
        compare_files()

        print(f"\n🎉 SUCCESS! Formatted file created at:")
        print(f"   {formatted_path}")


if __name__ == "__main__":
    main()
"""
if __name__ == "__main__":
    # Step 1: Inspect original CSV
    print("STEP 1: Inspecting original CSV\n")
    inspect_csv()

    # Step 2: Create formatted CSV
    print("\n\nSTEP 2: Creating formatted CSV\n")
    formatted_path = create_formatted_csv()

    # Step 3: Compare files
    if formatted_path:
        print("\n\nSTEP 3: Comparing files\n")
        compare_files()

        print(f"\n🎉 SUCCESS! Formatted file created at:")
        print(f"   {formatted_path}")

"""

