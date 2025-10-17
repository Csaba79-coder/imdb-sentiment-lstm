"""
Data preprocessing module for IMDB Sentiment Analysis.
Tokenizes text, creates sequences, and prepares data for LSTM training.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config import (
    DATA_PATH_CLEANED,
    VOCAB_SIZE,
    MAX_LENGTH,
    MODEL_DIR,
    TOKENIZER_PATH,
    VIZ_PREPROCESSING,
    RANDOM_SEED
)
from src.data_loader import load_and_prepare_data

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


def create_tokenizer(texts, vocab_size):
    """
    Create and fit a Keras Tokenizer on the training texts.

    Args:
        texts (np.array): Array of text reviews
        vocab_size (int): Maximum vocabulary size (most frequent words)

    Returns:
        Tokenizer: Fitted Keras Tokenizer object
    """
    print("\n" + "=" * 70)
    print("CREATING TOKENIZER")
    print("=" * 70 + "\n")

    print(f"🔧 Creating tokenizer with vocab_size={vocab_size:,}")

    # Create tokenizer
    # num_words: maximum vocabulary size (keeps only most frequent words)
    # oov_token: token for out-of-vocabulary words
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token='<OOV>',  # Out-of-vocabulary token
        lower=True,  # Convert to lowercase
        char_level=False  # Word-level tokenization
    )

    # Fit tokenizer on texts (learn vocabulary)
    print(f"📚 Fitting tokenizer on {len(texts):,} reviews...")
    tokenizer.fit_on_texts(texts)

    # Get vocabulary info
    word_index = tokenizer.word_index
    total_words = len(word_index)

    print(f"\n✅ Tokenizer created successfully!")
    print(f"   Total unique words found: {total_words:,}")
    print(f"   Vocabulary size (kept): {vocab_size:,}")
    print(f"   OOV token: <OOV>")

    # Show most common words
    print(f"\n📊 Top 10 most frequent words:")
    sorted_words = sorted(word_index.items(), key=lambda x: x[1])[:10]
    for word, index in sorted_words:
        print(f"   {index:3d}. {word}")

    return tokenizer


def texts_to_sequences(tokenizer, texts):
    """
    Convert texts to sequences of integers using the tokenizer.

    Args:
        tokenizer (Tokenizer): Fitted Keras Tokenizer
        texts (np.array): Array of text reviews

    Returns:
        list: List of integer sequences
    """
    print("\n" + "=" * 70)
    print("CONVERTING TEXTS TO SEQUENCES")
    print("=" * 70 + "\n")

    print(f"🔢 Converting {len(texts):,} texts to integer sequences...")

    # Convert texts to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Calculate statistics
    seq_lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(seq_lengths)
    median_length = np.median(seq_lengths)
    min_length = np.min(seq_lengths)
    max_length = np.max(seq_lengths)

    print(f"\n✅ Conversion complete!")
    print(f"   Total sequences: {len(sequences):,}")
    print(f"   Avg sequence length: {avg_length:.1f} tokens")
    print(f"   Median length: {median_length:.0f} tokens")
    print(f"   Min length: {min_length} tokens")
    print(f"   Max length: {max_length} tokens")

    return sequences, seq_lengths


def pad_sequences_to_length(sequences, max_length):
    """
    Pad sequences to a fixed length.

    Args:
        sequences (list): List of integer sequences
        max_length (int): Target length for all sequences

    Returns:
        np.array: Padded sequences array
    """
    print("\n" + "=" * 70)
    print("PADDING SEQUENCES")
    print("=" * 70 + "\n")

    print(f"⚖️  Padding sequences to length={max_length}")
    print(f"   Padding type: 'post' (add zeros at the end)")
    print(f"   Truncating type: 'post' (cut from the end)")

    # Pad sequences
    # padding='post': add zeros at the end
    # truncating='post': cut from the end if too long
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    # Calculate statistics
    original_lengths = [len(seq) for seq in sequences]
    too_short = sum(1 for length in original_lengths if length < max_length)
    too_long = sum(1 for length in original_lengths if length > max_length)
    exact = sum(1 for length in original_lengths if length == max_length)

    print(f"\n✅ Padding complete!")
    print(f"   Output shape: {padded.shape}")
    print(f"   Sequences padded (too short): {too_short:,} ({too_short / len(sequences) * 100:.1f}%)")
    print(f"   Sequences truncated (too long): {too_long:,} ({too_long / len(sequences) * 100:.1f}%)")
    print(f"   Sequences unchanged (exact): {exact:,} ({exact / len(sequences) * 100:.1f}%)")

    return padded


def save_tokenizer(tokenizer, path):
    """
    Save tokenizer to disk using pickle.

    Args:
        tokenizer (Tokenizer): Fitted tokenizer
        path (str): Path to save tokenizer
    """
    print("\n" + "=" * 70)
    print("SAVING TOKENIZER")
    print("=" * 70 + "\n")

    print(f"💾 Saving tokenizer to: {path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save tokenizer using pickle
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Tokenizer saved successfully!")
    print(f"   File size: {os.path.getsize(path) / 1024:.2f} KB")


def save_preprocessed_data(X_train, X_val, y_train, y_val):
    """
    Save preprocessed arrays to disk using NumPy.

    Args:
        X_train (np.array): Training sequences
        X_val (np.array): Validation sequences
        y_train (np.array): Training labels
        y_val (np.array): Validation labels
    """
    print("\n" + "=" * 70)
    print("SAVING PREPROCESSED DATA")
    print("=" * 70 + "\n")

    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(DATA_PATH_CLEANED)
    os.makedirs(data_dir, exist_ok=True)

    # Define paths
    paths = {
        'X_train': os.path.join(data_dir, 'X_train_preprocessed.npy'),
        'X_val': os.path.join(data_dir, 'X_val_preprocessed.npy'),
        'y_train': os.path.join(data_dir, 'y_train.npy'),
        'y_val': os.path.join(data_dir, 'y_val.npy')
    }

    # Save arrays
    print("💾 Saving preprocessed arrays:")

    np.save(paths['X_train'], X_train)
    print(f"   ✅ X_train: {paths['X_train']}")
    print(f"      Shape: {X_train.shape}, Size: {X_train.nbytes / 1024 / 1024:.2f} MB")

    np.save(paths['X_val'], X_val)
    print(f"   ✅ X_val: {paths['X_val']}")
    print(f"      Shape: {X_val.shape}, Size: {X_val.nbytes / 1024 / 1024:.2f} MB")

    np.save(paths['y_train'], y_train)
    print(f"   ✅ y_train: {paths['y_train']}")
    print(f"      Shape: {y_train.shape}, Size: {y_train.nbytes / 1024:.2f} KB")

    np.save(paths['y_val'], y_val)
    print(f"   ✅ y_val: {paths['y_val']}")
    print(f"      Shape: {y_val.shape}, Size: {y_val.nbytes / 1024:.2f} KB")

    total_size = sum([
        X_train.nbytes, X_val.nbytes,
        y_train.nbytes, y_val.nbytes
    ]) / 1024 / 1024

    print(f"\n✅ All data saved successfully!")
    print(f"   Total size: {total_size:.2f} MB")


def plot_sequence_length_distribution(seq_lengths_train, seq_lengths_val, max_length):
    """
    Plot sequence length distribution before padding.

    Args:
        seq_lengths_train (list): Training sequence lengths
        seq_lengths_val (list): Validation sequence lengths
        max_length (int): Maximum sequence length (padding target)
    """
    print("\n📊 Creating sequence length distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Training set
    axes[0].hist(seq_lengths_train, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(max_length, color='red', linestyle='--', linewidth=2,
                    label=f'Max Length: {max_length}')
    axes[0].axvline(np.mean(seq_lengths_train), color='green', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(seq_lengths_train):.0f}')
    axes[0].set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Set - Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Validation set
    axes[1].hist(seq_lengths_val, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].axvline(max_length, color='red', linestyle='--', linewidth=2,
                    label=f'Max Length: {max_length}')
    axes[1].axvline(np.mean(seq_lengths_val), color='green', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(seq_lengths_val):.0f}')
    axes[1].set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Set - Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(VIZ_PREPROCESSING, 'sequence_length_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {save_path}")


def plot_vocabulary_stats(tokenizer):
    """
    Plot vocabulary statistics.

    Args:
        tokenizer (Tokenizer): Fitted tokenizer
    """
    print("\n📊 Creating vocabulary statistics plot...")

    word_counts = tokenizer.word_counts
    sorted_counts = sorted(word_counts.values(), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 50 words frequency
    axes[0].bar(range(min(50, len(sorted_counts))), sorted_counts[:50],
                color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Word Rank', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Top 50 Most Frequent Words', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Word frequency distribution (log scale)
    axes[1].plot(range(len(sorted_counts)), sorted_counts, color='darkblue', linewidth=2)
    axes[1].set_xlabel('Word Rank', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    axes[1].set_title('Word Frequency Distribution (Zipf\'s Law)', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(VIZ_PREPROCESSING, 'vocabulary_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {save_path}")


def preprocess_pipeline():
    """
    Main preprocessing pipeline.

    Steps:
    1. Load cleaned data
    2. Create and fit tokenizer
    3. Convert texts to sequences
    4. Pad sequences to fixed length
    5. Save tokenizer and preprocessed data
    6. Create visualizations

    Returns:
        tuple: (X_train, X_val, y_train, y_val, tokenizer)
    """
    print("\n" + "=" * 70)
    print("🚀 DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # Step 1: Load cleaned data
    result = load_and_prepare_data(verbose=False)
    if result is None:
        print("\n❌ Failed to load data!")
        return None

    X_train_text, X_val_text, y_train, y_val = result

    # Step 2: Create and fit tokenizer on training data
    tokenizer = create_tokenizer(X_train_text, VOCAB_SIZE)

    # Step 3: Convert texts to sequences
    train_sequences, train_seq_lengths = texts_to_sequences(tokenizer, X_train_text)
    val_sequences, val_seq_lengths = texts_to_sequences(tokenizer, X_val_text)

    # Step 4: Pad sequences to fixed length
    X_train = pad_sequences_to_length(train_sequences, MAX_LENGTH)
    X_val = pad_sequences_to_length(val_sequences, MAX_LENGTH)

    # Step 5: Save tokenizer and preprocessed data
    save_tokenizer(tokenizer, TOKENIZER_PATH)
    save_preprocessed_data(X_train, X_val, y_train, y_val)

    # Step 6: Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    plot_sequence_length_distribution(train_seq_lengths, val_seq_lengths, MAX_LENGTH)
    plot_vocabulary_stats(tokenizer)

    print("\n" + "=" * 70)
    print("🎉 PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Tokenizer saved: {TOKENIZER_PATH}")
    print(f"📊 Visualizations saved: {VIZ_PREPROCESSING}")
    print(f"💾 Preprocessed data saved in: {os.path.dirname(DATA_PATH_CLEANED)}")
    print("\n" + "=" * 70 + "\n")

    return X_train, X_val, y_train, y_val, tokenizer


def main():
    """
    Main function to run preprocessing pipeline.
    """
    result = preprocess_pipeline()

    if result is not None:
        X_train, X_val, y_train, y_val, tokenizer = result
        print("\n✅ Preprocessing successful!")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Validation data shape: {X_val.shape}")
        print(f"   Vocabulary size: {len(tokenizer.word_index):,}")
    else:
        print("\n❌ Preprocessing failed!")


if __name__ == "__main__":
    main()

