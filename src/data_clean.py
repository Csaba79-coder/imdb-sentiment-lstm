"""
Data Cleaning and Exploratory Data Analysis (EDA) module for IMDB dataset.
Performs data cleaning, statistical analysis, and creates visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from src.config import DATA_PATH_FORMATTED, DATA_PATH_CLEANED, VIZ_EDA

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """
    Load the formatted dataset.

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    print(f"\n📖 Reading: {DATA_PATH_FORMATTED}\n")

    try:
        df = pd.read_csv(DATA_PATH_FORMATTED)
        print(f"✅ Data loaded successfully!")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def check_missing_values(df):
    """
    Check for missing values in the dataset.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with missing value information
    """
    print("\n" + "=" * 70)
    print("CHECKING MISSING VALUES")
    print("=" * 70 + "\n")

    missing = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })

    print(missing_df)

    if missing.sum() == 0:
        print("\n✅ No missing values found!")
    else:
        print(f"\n⚠️  Found {missing.sum()} missing values")

    return missing_df


def check_duplicates(df):
    """
    Check and remove duplicate rows.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    print("\n" + "=" * 70)
    print("CHECKING DUPLICATES")
    print("=" * 70 + "\n")

    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")

    if duplicates > 0:
        print(f"\n⚠️  Removing {duplicates} duplicate rows...")
        df_clean = df.drop_duplicates()
        print(f"✅ Duplicates removed!")
        print(f"   Rows before: {len(df)}")
        print(f"   Rows after: {len(df_clean)}")
        return df_clean
    else:
        print("\n✅ No duplicates found!")
        return df


def validate_sentiment_values(df):
    """
    Validate sentiment values and remove invalid rows.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with only valid sentiment values
    """
    print("\n" + "=" * 70)
    print("VALIDATING SENTIMENT VALUES")
    print("=" * 70 + "\n")

    # Check for empty strings
    empty_sentiment = df['sentiment'].str.strip() == ""
    empty_count = empty_sentiment.sum()

    # Check for invalid values (not 'positive' or 'negative')
    valid_sentiments = ['positive', 'negative']
    invalid_sentiment = ~df['sentiment'].isin(valid_sentiments)
    invalid_count = invalid_sentiment.sum()

    print(f"Empty sentiment values: {empty_count}")
    print(f"Invalid sentiment values: {invalid_count}")

    if empty_count > 0 or invalid_count > 0:
        print(f"\n⚠️  Found {empty_count + invalid_count} invalid rows")
        print(f"   Removing invalid rows...")
        df_clean = df[df['sentiment'].isin(valid_sentiments)].copy()
        print(f"✅ Invalid rows removed!")
        print(f"   Rows before: {len(df)}")
        print(f"   Rows after: {len(df_clean)}")
        return df_clean
    else:
        print("\n✅ All sentiment values are valid!")
        return df


def add_text_length_feature(df):
    """
    Add text length feature (word count and character count).

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with added features
    """
    print("\n" + "=" * 70)
    print("ADDING TEXT LENGTH FEATURES")
    print("=" * 70 + "\n")

    df['review_length'] = df['review'].apply(len)
    df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

    print("✅ Text length features added!")
    print(f"   - review_length: Character count")
    print(f"   - word_count: Word count")

    return df


def detect_outliers(df):
    """
    Detect outliers in text length using IQR method.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with outlier information
    """
    print("\n" + "=" * 70)
    print("DETECTING OUTLIERS (IQR METHOD)")
    print("=" * 70 + "\n")

    Q1 = df['word_count'].quantile(0.25)
    Q3 = df['word_count'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]

    print(f"Q1 (25th percentile): {Q1:.0f} words")
    print(f"Q3 (75th percentile): {Q3:.0f} words")
    print(f"IQR: {IQR:.0f} words")
    print(f"Lower bound: {lower_bound:.0f} words")
    print(f"Upper bound: {upper_bound:.0f} words")
    print(f"\n📊 Total outliers detected: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)")

    # We keep outliers for sentiment analysis (they might contain important info)
    print("\n💡 Note: Outliers are kept in the dataset (may contain valuable sentiment info)")

    return df


def descriptive_statistics(df):
    """
    Calculate and display descriptive statistics.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70 + "\n")

    print("📊 OVERALL STATISTICS:")
    print(f"   Total reviews: {len(df)}")
    print(f"   Unique reviews: {df['review'].nunique()}")
    print(f"   Duplicate reviews: {len(df) - df['review'].nunique()}")

    print("\n📊 SENTIMENT DISTRIBUTION:")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    print(f"\nBalance: {sentiment_counts.values[0] / sentiment_counts.values[1]:.2f}:1")

    print("\n📊 TEXT LENGTH STATISTICS:")
    print(df[['review_length', 'word_count']].describe())


def plot_sentiment_distribution(df):
    """
    Plot sentiment distribution.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n📊 Creating sentiment distribution plot...")

    plt.figure(figsize=(8, 6))
    sentiment_counts = df['sentiment'].value_counts()

    colors = ['#2ecc71', '#e74c3c']  # Green for positive, Red for negative
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
    plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        plt.text(i, count + 100, f'{count}\n({count / len(df) * 100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(VIZ_EDA, 'sentiment_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_text_length_distribution(df):
    """
    Plot text length distribution (histogram and boxplot).

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n📊 Creating text length distribution plots...")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Word count histogram
    axes[0].hist(df['word_count'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Word Count', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    axes[0].axvline(df['word_count'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {df["word_count"].mean():.0f}')
    axes[0].axvline(df['word_count'].median(), color='green', linestyle='--',
                    linewidth=2, label=f'Median: {df["word_count"].median():.0f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Character count histogram
    axes[1].hist(df['review_length'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Character Count', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Character Count Distribution', fontsize=14, fontweight='bold')
    axes[1].axvline(df['review_length'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {df["review_length"].mean():.0f}')
    axes[1].axvline(df['review_length'].median(), color='green', linestyle='--',
                    linewidth=2, label=f'Median: {df["review_length"].median():.0f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(VIZ_EDA, 'text_length_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Word count boxplot by sentiment
    df.boxplot(column='word_count', by='sentiment', ax=axes[0],
               patch_artist=True, grid=False)
    axes[0].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Word Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Word Count by Sentiment', fontsize=14, fontweight='bold')
    plt.sca(axes[0])
    plt.xticks([1, 2], ['negative', 'positive'])

    # Character count boxplot by sentiment
    df.boxplot(column='review_length', by='sentiment', ax=axes[1],
               patch_artist=True, grid=False)
    axes[1].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Character Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Character Count by Sentiment', fontsize=14, fontweight='bold')
    plt.sca(axes[1])
    plt.xticks([1, 2], ['negative', 'positive'])

    plt.suptitle('')  # Remove the automatic title
    plt.tight_layout()
    save_path = os.path.join(VIZ_EDA, 'text_length_boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_wordcloud(df, sentiment, max_words=100):
    """
    Create word cloud for specific sentiment.

    Args:
        df (pd.DataFrame): Input dataframe
        sentiment (str): 'positive' or 'negative'
        max_words (int): Maximum number of words in word cloud
    """
    print(f"\n📊 Creating word cloud for {sentiment} reviews...")

    # Filter by sentiment
    text = ' '.join(df[df['sentiment'] == sentiment]['review'].values)

    # Create word cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis' if sentiment == 'positive' else 'Reds',
        max_words=max_words,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)

    # Plot
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {sentiment.capitalize()} Reviews',
              fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout(pad=0)

    save_path = os.path.join(VIZ_EDA, f'wordcloud_{sentiment}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_top_words(df, sentiment, top_n=20):
    """
    Plot top N most frequent words for specific sentiment.

    Args:
        df (pd.DataFrame): Input dataframe
        sentiment (str): 'positive' or 'negative'
        top_n (int): Number of top words to display
    """
    print(f"\n📊 Creating top words plot for {sentiment} reviews...")

    # Filter by sentiment and get all words
    text = ' '.join(df[df['sentiment'] == sentiment]['review'].values)
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # Count word frequencies
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)

    # Plot
    words_list = [word for word, count in top_words]
    counts_list = [count for word, count in top_words]

    plt.figure(figsize=(12, 8))
    color = '#2ecc71' if sentiment == 'positive' else '#e74c3c'
    plt.barh(range(len(words_list)), counts_list, color=color, alpha=0.7)
    plt.yticks(range(len(words_list)), words_list)
    plt.xlabel('Frequency', fontsize=12, fontweight='bold')
    plt.ylabel('Words', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Frequent Words - {sentiment.capitalize()} Reviews',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(VIZ_EDA, f'top_words_{sentiment}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def save_cleaned_data(df):
    """
    Save cleaned dataset to CSV.

    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("\n" + "=" * 70)
    print("SAVING CLEANED DATA")
    print("=" * 70 + "\n")

    # Drop temporary columns used for analysis
    df_to_save = df[['review', 'sentiment']].copy()

    print(f"💾 Saving cleaned data to: {DATA_PATH_CLEANED}")
    df_to_save.to_csv(DATA_PATH_CLEANED, index=False)
    print(f"✅ Cleaned data saved successfully!")
    print(f"   Total rows: {len(df_to_save)}")
    print(f"   Columns: {list(df_to_save.columns)}")


def main():
    """
    Main function to run the entire data cleaning and EDA pipeline.
    """
    print("\n" + "=" * 70)
    print("DATA CLEANING & EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Load data
    df = load_data()
    if df is None:
        return

    # Data cleaning
    check_missing_values(df)
    df = check_duplicates(df)
    df = validate_sentiment_values(df)  # ← ÚJ VALIDÁCIÓ!
    df = add_text_length_feature(df)
    df = detect_outliers(df)

    # Descriptive statistics
    descriptive_statistics(df)

    # Visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    plot_sentiment_distribution(df)
    plot_text_length_distribution(df)
    create_wordcloud(df, 'positive')
    create_wordcloud(df, 'negative')
    plot_top_words(df, 'positive')
    plot_top_words(df, 'negative')

    # Save cleaned data
    save_cleaned_data(df)

    print("\n" + "=" * 70)
    print("🎉 DATA CLEANING & EDA COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📂 Cleaned data saved to: {DATA_PATH_CLEANED}")
    print(f"📊 Visualizations saved to: {VIZ_EDA}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

