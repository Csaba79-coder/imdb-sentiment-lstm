"""
Training script for LSTM sentiment analysis model.
Loads preprocessed data, trains model, and saves results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.config import (
    DATA_PATH_CLEANED,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    VIZ_TRAINING,
    VERBOSE
)
from src.model import create_and_compile_model

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def load_preprocessed_data():
    """
    Load preprocessed training and validation data.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70 + "\n")

    data_dir = os.path.dirname(DATA_PATH_CLEANED)

    # Define paths
    paths = {
        'X_train': os.path.join(data_dir, 'X_train_preprocessed.npy'),
        'X_val': os.path.join(data_dir, 'X_val_preprocessed.npy'),
        'y_train': os.path.join(data_dir, 'y_train.npy'),
        'y_val': os.path.join(data_dir, 'y_val.npy')
    }

    # Check if files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"❌ Error: {name} not found at {path}")
            print("Please run data_preprocess.py first!")
            return None

    # Load arrays
    print("📂 Loading arrays:")
    X_train = np.load(paths['X_train'])
    print(f"   ✅ X_train: {X_train.shape}")

    X_val = np.load(paths['X_val'])
    print(f"   ✅ X_val: {X_val.shape}")

    y_train = np.load(paths['y_train'])
    print(f"   ✅ y_train: {y_train.shape}")

    y_val = np.load(paths['y_val'])
    print(f"   ✅ y_val: {y_val.shape}")

    print(f"\n✅ All data loaded successfully!")
    print("=" * 70 + "\n")

    return X_train, X_val, y_train, y_val


def create_callbacks(model_path=MODEL_SAVE_PATH, patience=EARLY_STOPPING_PATIENCE):
    """
    Create training callbacks for model optimization.

    Args:
        model_path (str): Path to save best model
        patience (int): Early stopping patience (epochs)

    Returns:
        list: List of Keras callbacks
    """
    print("=" * 70)
    print("CREATING TRAINING CALLBACKS")
    print("=" * 70 + "\n")

    # 1. Early Stopping - stop training if val_loss doesn't improve
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    print(f"✅ Early Stopping:")
    print(f"   Monitor: val_loss")
    print(f"   Patience: {patience} epochs")
    print(f"   Restore best weights: True")

    # 2. Model Checkpoint - save best model during training
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    print(f"\n✅ Model Checkpoint:")
    print(f"   Save path: {model_path}")
    print(f"   Monitor: val_accuracy")
    print(f"   Save best only: True")

    # 3. Reduce Learning Rate on Plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    print(f"\n✅ Reduce LR on Plateau:")
    print(f"   Monitor: val_loss")
    print(f"   Factor: 0.5 (halve LR)")
    print(f"   Patience: 2 epochs")

    print("\n" + "=" * 70 + "\n")

    return [early_stop, checkpoint, reduce_lr]


def train_model(model, X_train, y_train, X_val, y_val,
                batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE):
    """
    Train the LSTM model.

    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        batch_size (int): Batch size for training
        epochs (int): Number of epochs
        verbose (int): Verbosity mode (0, 1, or 2)

    Returns:
        History: Keras training history object
    """
    print("=" * 70)
    print("TRAINING MODEL")
    print("=" * 70 + "\n")

    print(f"📊 Training Configuration:")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Training Samples: {len(X_train):,}")
    print(f"   Validation Samples: {len(X_val):,}")
    print(f"   Steps per Epoch: {len(X_train) // batch_size}")
    print(f"   Validation Steps: {len(X_val) // batch_size}")

    # Create callbacks
    callbacks = create_callbacks()

    print("\n🚀 Starting training...\n")
    print("=" * 70 + "\n")

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )

    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70 + "\n")

    return history


def save_training_history(history, save_path=None):
    """
    Save training history as JSON.

    Args:
        history: Keras History object
        save_path (str): Path to save history JSON
    """
    if save_path is None:
        save_path = os.path.join(VIZ_TRAINING, 'training_history.json')

    print("=" * 70)
    print("SAVING TRAINING HISTORY")
    print("=" * 70 + "\n")

    # Convert history to dict
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }

    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=4)

    print(f"✅ Training history saved: {save_path}")
    print("=" * 70 + "\n")


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Keras History object
        save_path (str): Directory to save plots
    """
    if save_path is None:
        save_path = VIZ_TRAINING

    print("=" * 70)
    print("CREATING TRAINING PLOTS")
    print("=" * 70 + "\n")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get training history
    epochs_range = range(1, len(history.history['loss']) + 1)

    # Plot 1: Loss
    axes[0].plot(epochs_range, history.history['loss'],
                 label='Training Loss', linewidth=2, marker='o')
    axes[0].plot(epochs_range, history.history['val_loss'],
                 label='Validation Loss', linewidth=2, marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(epochs_range, history.history['accuracy'],
                 label='Training Accuracy', linewidth=2, marker='o', color='green')
    axes[1].plot(epochs_range, history.history['val_accuracy'],
                 label='Validation Accuracy', linewidth=2, marker='s', color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(save_path, 'training_history.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Training plots saved: {plot_file}")
    print("=" * 70 + "\n")


def print_training_summary(history):
    """
    Print summary of training results.

    Args:
        history: Keras History object
    """
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70 + "\n")

    # Get final metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    # Get best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"📊 Final Epoch Results:")
    print(f"   Training Loss:       {final_loss:.4f}")
    print(f"   Training Accuracy:   {final_acc:.4f} ({final_acc * 100:.2f}%)")
    print(f"   Validation Loss:     {final_val_loss:.4f}")
    print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")

    print(f"\n🏆 Best Validation Accuracy:")
    print(f"   Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"   Epoch: {best_val_acc_epoch}/{len(history.history['loss'])}")

    print(f"\n📈 Training Epochs: {len(history.history['loss'])}")

    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main training pipeline.
    """
    print("\n" + "=" * 70)
    print("🚀 LSTM SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Load preprocessed data
    result = load_preprocessed_data()
    if result is None:
        return
    X_train, X_val, y_train, y_val = result

    # Step 2: Create and compile model
    print("\n📌 Creating LSTM model...")
    model = create_and_compile_model(bidirectional=False)

    # Step 3: Train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Step 4: Save training history
    save_training_history(history)

    # Step 5: Plot training history
    plot_training_history(history)

    # Step 6: Print summary
    print_training_summary(history)

    print("=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Model saved: {MODEL_SAVE_PATH}")
    print(f"📊 Plots saved: {VIZ_TRAINING}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

