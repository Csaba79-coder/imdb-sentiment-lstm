"""
Configuration file for IMDB Sentiment Analysis LSTM project.
All hyperparameters and settings are centralized here for easy modification.
"""

import os

# ============================================================================
# PROJECT ROOT - AUTOMATIC DETECTION
# ============================================================================

# Get the directory where this config.py file is located (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up to get project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Path to the dataset (now relative to PROJECT_ROOT)
DATA_PATH_ORIGINAL = os.path.join(PROJECT_ROOT, 'data', 'imdb_dataset.csv')
DATA_PATH_FORMATTED = os.path.join(PROJECT_ROOT, 'data', 'imdb_dataset_formatted.csv')
DATA_PATH_CLEANED = os.path.join(PROJECT_ROOT, 'data', 'imdb_dataset_cleaned.csv')

# Maximum number of words in vocabulary (most frequent words)
VOCAB_SIZE = 10000

# Maximum length of each review (sequences will be padded/truncated)
MAX_LENGTH = 200

# Percentage of data used for validation
VALIDATION_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Dimension of word embedding vectors
EMBEDDING_DIM = 128

# Number of units in LSTM layer
LSTM_UNITS = 128

# Dropout rate to prevent overfitting (0.0 - 1.0)
DROPOUT = 0.5

# Recurrent dropout (dropout applied to recurrent connections)
RECURRENT_DROPOUT = 0.2

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Number of samples per gradient update
BATCH_SIZE = 64

# Number of complete passes through the training dataset
EPOCHS = 10

# Learning rate for optimizer
LEARNING_RATE = 0.001

# Early stopping patience (stop if no improvement after N epochs)
EARLY_STOPPING_PATIENCE = 3

# ============================================================================
# MODEL SAVING
# ============================================================================

# Directory to save trained models
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Model filename
MODEL_NAME = 'lstm_sentiment_model.h5'

# Full path to save the model
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Path to save tokenizer
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pickle')

# ============================================================================
# VISUALIZATION PATHS
# ============================================================================

# Root directory for all visualizations
VIZ_ROOT = os.path.join(PROJECT_ROOT, 'visualizations')

# EDA visualizations (exploratory data analysis)
VIZ_EDA = os.path.join(VIZ_ROOT, 'eda')

# Preprocessing visualizations
VIZ_PREPROCESSING = os.path.join(VIZ_ROOT, 'preprocessing')

# Training history plots
VIZ_TRAINING = os.path.join(VIZ_ROOT, 'training')

# ============================================================================
# PREPROCESSING
# ============================================================================

# Whether to convert text to lowercase
LOWERCASE = True

# Whether to remove HTML tags
REMOVE_HTML = True

# Whether to remove punctuation
REMOVE_PUNCTUATION = False  # Keep punctuation (! and ? can indicate sentiment)

# Whether to remove stopwords
REMOVE_STOPWORDS = False  # Stopwords can be important for sentiment

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Verbosity level for training (0 = silent, 1 = progress bar, 2 = one line per epoch)
VERBOSE = 1

# Whether to plot training history
PLOT_HISTORY = True

# Directory for plots and visualizations (DEPRECATED - use VIZ_TRAINING)
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """
    Create necessary directories if they don't exist.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH_CLEANED), exist_ok=True)

    # Create visualization directories
    os.makedirs(VIZ_EDA, exist_ok=True)
    os.makedirs(VIZ_PREPROCESSING, exist_ok=True)
    os.makedirs(VIZ_TRAINING, exist_ok=True)


def print_config():
    """
    Print all configuration parameters.
    Useful for logging and debugging.
    """
    print("=" * 60)
    print("CONFIGURATION PARAMETERS")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Original: {DATA_PATH_ORIGINAL}")
    print(f"Dataset Formatted: {DATA_PATH_FORMATTED}")
    print(f"Dataset Cleaned: {DATA_PATH_CLEANED}")
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Max Sequence Length: {MAX_LENGTH}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}")
    print(f"LSTM Units: {LSTM_UNITS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Visualization Root: {VIZ_ROOT}")
    print("=" * 60)


# Create directories when config is imported
create_directories()