import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns  # <-- ADDED SEABORN FOR THE CLEAN HEATMAP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
KEYPOINTS_PATH = "keypoints"
MODEL_PATH     = "action.h5"
ACTIONS_PATH   = "actions.npy"
SEQUENCE_LENGTH = 30
# ──────────────────────────────────────────────────────────────────────────────

def load_data(actions):
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        action_path = os.path.join(KEYPOINTS_PATH, action)
        if not os.path.isdir(action_path):
            continue
        for fname in os.listdir(action_path):
            if not fname.endswith('.npy'):
                continue
            data = np.load(os.path.join(action_path, fname))
            if len(data) < SEQUENCE_LENGTH:
                continue
            sequences.append(data[:SEQUENCE_LENGTH])
            labels.append(label_map[action])

    return np.array(sequences), np.array(labels)


def plot_class_distribution(actions, labels):
    counts = [np.sum(labels == i) for i in range(len(actions))]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(actions, counts, color='steelblue', edgecolor='black', linewidth=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontsize=8)

    ax.set_title('Dataset — Sample Count per Class', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Sign Class', fontsize=11)
    ax.set_ylabel('Number of Sequences', fontsize=11)
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels(actions, rotation=45, ha='right', fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('graph_class_distribution.png', dpi=150)
    plt.close()
    print("✅  Saved: graph_class_distribution.png")


def plot_per_class_accuracy(actions, y_true, y_pred):
    per_class_acc = []
    for i in range(len(actions)):
        mask = y_true == i
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(np.mean(y_pred[mask] == y_true[mask]) * 100)

    # Color coding based on accuracy thresholds
    colors = ['#2ecc71' if a >= 90 else '#e67e22' if a >= 75 else '#e74c3c'
              for a in per_class_acc]

    # Adjusted figsize: Wide (14) and short (6) to fit 38 vertical bars
    fig, ax = plt.subplots(figsize=(14, 6)) 
    
    # Use ax.bar for vertical bars
    bars = ax.bar(actions, per_class_acc, color=colors, edgecolor='black', linewidth=0.5)

    # Add text labels on TOP of each bar
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 1,
                f'{acc:.0f}%', 
                ha='center', va='bottom', fontsize=8, rotation=90)

    # Add Mean Line (Horizontal now)
    mean_acc = np.mean(per_class_acc)
    ax.axhline(y=mean_acc, color='navy', linestyle='--',
               linewidth=1.2, label=f'Mean: {mean_acc:.1f}%')

    # Formatting
    ax.set_title('Per-Class Accuracy (38 ISL Classes)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sign Class', fontsize=11, fontweight='bold')
    
    # Extend Y limit to make room for the text labels on top
    ax.set_ylim(0, 120) 
    
    # Rotate X-axis labels 90 degrees so they are readable
    plt.xticks(rotation=90, fontsize=9)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label='≥ 90% Excellent'),
                       Patch(facecolor='#e67e22', label='75–89% Good'),
                       Patch(facecolor='#e74c3c', label='< 75% Needs work')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('graph_per_class_accuracy_vertical.png', dpi=300) 
    plt.close()
    print("✅  Saved: graph_per_class_accuracy_vertical.png")


# 🚨 THE NEW CONFUSION MATRIX FUNCTION 🚨
def plot_confusion_matrix(actions, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the matrix to show percentages (0.0 to 1.0)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a square figure size perfectly suited for a single column width
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot using Seaborn heatmap (annot=False turns OFF the unreadable numbers)
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', cbar=True,
                xticklabels=actions, yticklabels=actions, ax=ax)
    
    # Formatting the axes to be small but readable
    ax.set_title('Normalized Confusion Matrix', fontsize=11, fontweight='bold', pad=15)
    ax.set_ylabel('True Sign Class', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Sign Class', fontsize=10, fontweight='bold')
    
    # Rotate labels so they don't overlap, and shrink font to size 6
    plt.xticks(rotation=90, fontsize=6) 
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    # Save at 300 DPI for crisp conference printing
    plt.savefig('graph_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅  Saved: graph_confusion_matrix.png (High-Res Single Column Heatmap)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⏳ Loading model and data...")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ '{MODEL_PATH}' not found. Run from your project root folder.")
        exit()
    if not os.path.exists(ACTIONS_PATH):
        print(f"❌ '{ACTIONS_PATH}' not found.")
        exit()

    model   = load_model(MODEL_PATH)
    actions = list(np.load(ACTIONS_PATH, allow_pickle=True))
    print(f"✅ Model loaded. Classes ({len(actions)}): {actions}")

    # ==========================================
    # 🚨 THE FIX: ISOLATING UNSEEN DATA
    # ==========================================
    # 1. Load basic data
    X_raw, y_raw = load_data(actions)

    # 2. Apply the exact same augmentation as train_model.py to match shapes
    np.random.seed(0)
    noise = np.random.normal(loc=0.0, scale=0.05, size=X_raw.shape)
    X_noise = X_raw + noise
    
    X_full = np.concatenate([X_raw, X_noise])
    y_full = np.concatenate([y_raw, y_raw])

    # 3. Perform the EXACT SAME split used in training (SEED=0, test_size=0.15)
    # We throw away the training data (_) and ONLY keep the test data
    _, X_test, _, y_true = train_test_split(X_full, y_full, test_size=0.15, random_state=0)

    print(f"✅ Data loaded. Testing strictly on UNSEEN validation data: {len(X_test)} sequences.")

    print("🔍 Running predictions (this may take a minute)...")
    # 4. Predict ONLY on the unseen test data
    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    overall_acc = np.mean(y_pred == y_true) * 100
    print(f"\n🎯 True Unseen Accuracy: {overall_acc:.2f}%")
    # ==========================================

    print("\n📊 Generating graphs...")
    plot_class_distribution(actions, y_true)
    plot_per_class_accuracy(actions, y_true, y_pred)
    plot_confusion_matrix(actions, y_true, y_pred)

    from sklearn.metrics import classification_report

    print("\n📋 Generating Classification Report...")
    report = classification_report(y_true, y_pred, target_names=actions)
    print(report)

    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("✅  Saved: classification_report.txt")

    print("\n✅ ALL DONE. 4 files saved in your project folder:")
    print("   → graph_class_distribution.png")
    print("   → graph_per_class_accuracy.png")
    print("   → graph_confusion_matrix.png")
    print("   → classification_report.txt")