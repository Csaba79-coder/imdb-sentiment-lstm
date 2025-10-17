"""
LSTM Model Architecture for IMDB Sentiment Analysis.
Implements a many-to-one LSTM network for binary sentiment classification.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from src.config import (
    VOCAB_SIZE,
    MAX_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS,
    DROPOUT,
    RECURRENT_DROPOUT,
    LEARNING_RATE,
    VIZ_TRAINING
)


def build_lstm_model(vocab_size=VOCAB_SIZE,
                     max_length=MAX_LENGTH,
                     embedding_dim=EMBEDDING_DIM,
                     lstm_units=LSTM_UNITS,
                     dropout=DROPOUT,
                     recurrent_dropout=RECURRENT_DROPOUT):
    """
    Build a many-to-one LSTM model for sentiment classification.

    Architecture:
        Input (batch_size, max_length)
            ↓
        Embedding Layer (vocab_size, embedding_dim)
            ↓
        LSTM Layer (lstm_units, dropout, recurrent_dropout)
            ↓
        Dropout Layer (dropout)
            ↓
        Dense Output Layer (1 unit, sigmoid activation)
            ↓
        Output (batch_size, 1) - probability [0, 1]

    Args:
        vocab_size (int): Size of vocabulary (number of unique words)
        max_length (int): Maximum sequence length (padded/truncated)
        embedding_dim (int): Dimension of word embedding vectors
        lstm_units (int): Number of LSTM units
        dropout (float): Dropout rate for regularization (0.0-1.0)
        recurrent_dropout (float): Recurrent dropout rate (0.0-1.0)

    Returns:
        Sequential: Compiled Keras model
    """
    print("\n" + "=" * 70)
    print("BUILDING LSTM MODEL")
    print("=" * 70 + "\n")

    # Initialize Sequential model
    model = Sequential(name='LSTM_Sentiment_Classifier')

    # 1. Embedding Layer
    # Converts integer sequences into dense vectors of fixed size
    # Input shape: (batch_size, max_length)
    # Output shape: (batch_size, max_length, embedding_dim)
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name='embedding_layer'
    ))
    print(f"✅ Added Embedding Layer:")
    print(f"   Input: vocab_size={vocab_size:,}, max_length={max_length}")
    print(f"   Output: embedding_dim={embedding_dim}")

    # 2. LSTM Layer (Many-to-One)
    # Processes the sequence and returns only the last output
    # return_sequences=False: only return output at the last timestep
    # Input shape: (batch_size, max_length, embedding_dim)
    # Output shape: (batch_size, lstm_units)
    model.add(LSTM(
        units=lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,  # Many-to-one: return only last output
        name='lstm_layer'
    ))
    print(f"\n✅ Added LSTM Layer:")
    print(f"   Units: {lstm_units}")
    print(f"   Dropout: {dropout}")
    print(f"   Recurrent Dropout: {recurrent_dropout}")
    print(f"   Architecture: Many-to-One (return_sequences=False)")

    # 3. Dropout Layer (additional regularization)
    # Randomly sets a fraction of input units to 0 during training
    model.add(Dropout(dropout, name='dropout_layer'))
    print(f"\n✅ Added Dropout Layer:")
    print(f"   Rate: {dropout}")

    # 4. Dense Output Layer
    # Binary classification with sigmoid activation
    # Output: probability between 0 (negative) and 1 (positive)
    # Input shape: (batch_size, lstm_units)
    # Output shape: (batch_size, 1)
    model.add(Dense(
        units=1,
        activation='sigmoid',
        name='output_layer'
    ))
    print(f"\n✅ Added Dense Output Layer:")
    print(f"   Units: 1 (binary classification)")
    print(f"   Activation: sigmoid")
    print(f"   Output: probability [0=negative, 1=positive]")

    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE COMPLETE")
    print("=" * 70 + "\n")

    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the model with optimizer, loss function, and metrics.

    Args:
        model (Sequential): Keras model to compile
        learning_rate (float): Learning rate for Adam optimizer

    Returns:
        Sequential: Compiled model
    """
    print("=" * 70)
    print("COMPILING MODEL")
    print("=" * 70 + "\n")

    # Adam optimizer with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Binary crossentropy loss (for binary classification)
    # Metrics: accuracy
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model compiled successfully!")
    print(f"   Optimizer: Adam (learning_rate={learning_rate})")
    print(f"   Loss: binary_crossentropy")
    print(f"   Metrics: accuracy")
    print("\n" + "=" * 70 + "\n")

    return model


def print_model_summary(model):
    """
    Print detailed model summary with layer information.

    Args:
        model (Sequential): Keras model
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70 + "\n")

    # Build model before summary (required for Keras 3.x)
    model.build(input_shape=(None, MAX_LENGTH))

    model.summary()

    # Calculate total parameters
    total_params = model.count_params()

    print("\n" + "=" * 70)
    print(f"Total Parameters: {total_params:,}")
    print("=" * 70 + "\n")


def save_model_architecture(model, save_path=None):
    """
    Save model architecture as JSON and visualization as PNG.

    Args:
        model (Sequential): Keras model
        save_path (str): Directory to save files (default: VIZ_TRAINING)
    """
    if save_path is None:
        save_path = VIZ_TRAINING

    os.makedirs(save_path, exist_ok=True)

    print("=" * 70)
    print("SAVING MODEL ARCHITECTURE")
    print("=" * 70 + "\n")

    # 1. Save model architecture as JSON
    json_path = os.path.join(save_path, 'model_architecture.json')
    with open(json_path, 'w') as f:
        f.write(model.to_json())
    print(f"✅ Architecture JSON saved: {json_path}")

    # 2. Save model config as readable JSON
    config_path = os.path.join(save_path, 'model_config.json')
    config = {
        'name': model.name,
        'layers': len(model.layers),
        'parameters': model.count_params(),
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape)
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Model config saved: {config_path}")

    # 3. Save model visualization as PNG
    try:
        plot_path = os.path.join(save_path, 'model_architecture.png')
        plot_model(
            model,
            to_file=plot_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',  # Top to Bottom
            expand_nested=True,
            dpi=150
        )
        print(f"✅ Model visualization saved: {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not save visualization: {e}")
        print("   (graphviz may not be installed)")

    print("\n" + "=" * 70 + "\n")


def build_bidirectional_lstm_model(vocab_size=VOCAB_SIZE,
                                   max_length=MAX_LENGTH,
                                   embedding_dim=EMBEDDING_DIM,
                                   lstm_units=LSTM_UNITS,
                                   dropout=DROPOUT,
                                   recurrent_dropout=RECURRENT_DROPOUT):
    """
    Build a Bidirectional LSTM model (optional, more powerful version).
    Processes sequences in both forward and backward directions.

    Args:
        vocab_size (int): Size of vocabulary
        max_length (int): Maximum sequence length
        embedding_dim (int): Dimension of word embeddings
        lstm_units (int): Number of LSTM units
        dropout (float): Dropout rate
        recurrent_dropout (float): Recurrent dropout rate

    Returns:
        Sequential: Compiled Keras model
    """
    print("\n" + "=" * 70)
    print("BUILDING BIDIRECTIONAL LSTM MODEL")
    print("=" * 70 + "\n")

    model = Sequential(name='BiLSTM_Sentiment_Classifier')

    # Embedding Layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name='embedding_layer'
    ))
    print(f"✅ Added Embedding Layer (vocab={vocab_size:,}, dim={embedding_dim})")

    # Bidirectional LSTM Layer
    model.add(Bidirectional(
        LSTM(
            units=lstm_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False
        ),
        name='bidirectional_lstm_layer'
    ))
    print(f"✅ Added Bidirectional LSTM Layer (units={lstm_units})")

    # Dropout Layer
    model.add(Dropout(dropout, name='dropout_layer'))
    print(f"✅ Added Dropout Layer (rate={dropout})")

    # Output Layer
    model.add(Dense(1, activation='sigmoid', name='output_layer'))
    print(f"✅ Added Dense Output Layer (sigmoid)")

    print("\n" + "=" * 70 + "\n")

    return model


def create_and_compile_model(bidirectional=False):
    """
    Create and compile LSTM model (standard or bidirectional).

    Args:
        bidirectional (bool): Use bidirectional LSTM if True

    Returns:
        Sequential: Compiled Keras model
    """
    if bidirectional:
        model = build_bidirectional_lstm_model()
    else:
        model = build_lstm_model()

    model = compile_model(model)
    print_model_summary(model)
    save_model_architecture(model)

    return model


def main():
    """
    Main function to test model creation.
    """
    print("\n" + "=" * 70)
    print("TESTING MODEL CREATION")
    print("=" * 70)

    # Create standard LSTM model
    print("\n📌 Creating Standard LSTM Model...")
    model = create_and_compile_model(bidirectional=False)

    print("\n✅ Model creation successful!")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

