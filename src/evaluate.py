"""
Model evaluation script for LSTM sentiment analysis.
Generates confusion matrix, classification report, and sample predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.config import MODEL_SAVE_PATH, DATA_PATH_CLEANED, VIZ_TRAINING

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def load_trained_model(model_path=MODEL_SAVE_PATH):
    """
    Load trained LSTM model.

    Args:
        model_path (str): Path to saved model

    Returns:
        model: Loaded Keras model
    """
    print("\n" + "=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70 + "\n")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first: python src/train.py")
        return None

    print(f"📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print(f"✅ Model loaded successfully!")

    print("\n" + "=" * 70 + "\n")
    return model


def load_validation_data():
    """
    Load preprocessed validation data.

    Returns:
        tuple: (X_val, y_val)
    """
    print("=" * 70)
    print("LOADING VALIDATION DATA")
    print("=" * 70 + "\n")

    data_dir = os.path.dirname(DATA_PATH_CLEANED)

    X_val_path = os.path.join(data_dir, 'X_val_preprocessed.npy')
    y_val_path = os.path.join(data_dir, 'y_val.npy')

    if not os.path.exists(X_val_path) or not os.path.exists(y_val_path):
        print("❌ Error: Validation data not found!")
        print("Please run data_preprocess.py first!")
        return None, None

    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    print(f"✅ X_val loaded: {X_val.shape}")
    print(f"✅ y_val loaded: {y_val.shape}")

    print("\n" + "=" * 70 + "\n")
    return X_val, y_val


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation set.

    Args:
        model: Trained Keras model
        X_val: Validation sequences
        y_val: Validation labels

    Returns:
        tuple: (loss, accuracy, predictions)
    """
    print("=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70 + "\n")

    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

    print(f"📊 Evaluation Results:")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Get predictions
    print(f"\n🔮 Generating predictions...")
    predictions = model.predict(X_val, verbose=0)

    print(f"✅ Predictions generated: {predictions.shape}")

    print("\n" + "=" * 70 + "\n")
    return loss, accuracy, predictions


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        save_path (str): Path to save plot
    """
    print("=" * 70)
    print("CREATING CONFUSION MATRIX")
    print("=" * 70 + "\n")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add counts as text
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                     ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_path is None:
        save_path = VIZ_TRAINING

    plot_file = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved: {plot_file}")
    print("\n" + "=" * 70 + "\n")


def print_classification_report(y_true, y_pred):
    """
    Print classification report with precision, recall, F1-score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70 + "\n")

    target_names = ['Negative', 'Positive']
    report = classification_report(y_true, y_pred, target_names=target_names)

    print(report)
    print("=" * 70 + "\n")


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path (str): Path to save plot
    """
    print("=" * 70)
    print("CREATING ROC CURVE")
    print("=" * 70 + "\n")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = VIZ_TRAINING

    plot_file = os.path.join(save_path, 'roc_curve.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC curve saved: {plot_file}")
    print(f"   AUC Score: {roc_auc:.4f}")
    print("\n" + "=" * 70 + "\n")


def show_sample_predictions(model, X_val, y_val, num_samples=10):
    """
    Show sample predictions with true labels.

    Args:
        model: Trained model
        X_val: Validation sequences
        y_val: True labels
        num_samples (int): Number of samples to show
    """
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70 + "\n")

    # Random sample indices
    indices = np.random.choice(len(X_val), num_samples, replace=False)

    # Get predictions
    samples = X_val[indices]
    predictions = model.predict(samples, verbose=0)

    print(f"Showing {num_samples} random predictions:\n")
    print(f"{'Index':<8} {'True':<12} {'Predicted':<12} {'Confidence':<12} {'Correct':<10}")
    print("-" * 70)

    correct = 0
    for i, idx in enumerate(indices):
        true_label = 'Positive' if y_val[idx] == 1 else 'Negative'
        pred_label = 'Positive' if predictions[i][0] >= 0.5 else 'Negative'
        confidence = predictions[i][0] if predictions[i][0] >= 0.5 else 1 - predictions[i][0]
        is_correct = '✅' if true_label == pred_label else '❌'

        if true_label == pred_label:
            correct += 1

        print(f"{idx:<8} {true_label:<12} {pred_label:<12} {confidence:<12.4f} {is_correct:<10}")

    print("-" * 70)
    print(f"\nSample Accuracy: {correct}/{num_samples} ({correct / num_samples * 100:.1f}%)")
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main evaluation pipeline.
    """
    print("\n" + "=" * 70)
    print("🔍 MODEL EVALUATION PIPELINE")
    print("=" * 70)

    # Step 1: Load trained model
    model = load_trained_model()
    if model is None:
        return

    # Step 2: Load validation data
    X_val, y_val = load_validation_data()
    if X_val is None:
        return

    # Step 3: Evaluate model
    loss, accuracy, predictions = evaluate_model(model, X_val, y_val)

    # Step 4: Convert predictions to binary
    y_pred = (predictions >= 0.5).astype(int).flatten()

    # Step 5: Confusion matrix
    plot_confusion_matrix(y_val, y_pred)

    # Step 6: Classification report
    print_classification_report(y_val, y_pred)

    # Step 7: ROC curve
    plot_roc_curve(y_val, predictions)

    # Step 8: Sample predictions
    show_sample_predictions(model, X_val, y_val, num_samples=15)

    print("=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Visualizations saved: {VIZ_TRAINING}")
    print(f"   - confusion_matrix.png")
    print(f"   - roc_curve.png")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

