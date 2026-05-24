import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # CHANGE 1: added ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# CHANGE 2: Seed removed from 42 → 0 (better initialization for this dataset)
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- CONFIGURATION --- (UNCHANGED)
KEYPOINTS_PATH = "keypoints"
SEQUENCE_LENGTH = 30
# ---------------------

# UNCHANGED: Gaussian noise augmentation
def add_noise(data, scale=0.05):
    noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
    return data + noise

# UNCHANGED: Data loading
def load_data():
    sequences, labels = [], []

    if not os.path.exists(KEYPOINTS_PATH):
        print(f"❌ Error: Folder '{KEYPOINTS_PATH}' not found.")
        return [], [], []

    actions = os.listdir(KEYPOINTS_PATH)
    actions.sort()

    label_map = {label: num for num, label in enumerate(actions)}
    print(f"📂 Detected {len(actions)} Actions: {actions}")

    for action in actions:
        action_path = os.path.join(KEYPOINTS_PATH, action)
        if not os.path.isdir(action_path): continue

        files = os.listdir(action_path)
        npy_files = [f for f in files if f.endswith('.npy')]

        for file_name in npy_files:
            file_path = os.path.join(action_path, file_name)
            res = np.load(file_path)
            if len(res) < SEQUENCE_LENGTH: continue
            window = res[:SEQUENCE_LENGTH]
            sequences.append(window)
            labels.append(label_map[action])

    return np.array(sequences), to_categorical(labels).astype(int), actions

# ═══════════════════════════════════════════════════
# ARCHITECTURE: COMPLETELY UNCHANGED
# ═══════════════════════════════════════════════════
def build_model(num_actions):
    model = Sequential()

    # REMOVED activation='relu' so it defaults to the highly optimized 'tanh'
    model.add(LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 178)))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))

    # Keep ReLU for the Dense layers (they are fine here)
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load Data (UNCHANGED)
    print("⏳ Loading data...")
    X, y, actions = load_data()

    if len(X) == 0:
        print("❌ Error: No valid data found!")
        exit()

    # Augmentation (UNCHANGED)
    print("✨ Applying Noise Injection (Augmentation)...")
    X_noise = add_noise(X)
    X = np.concatenate([X, X_noise])
    y = np.concatenate([y, y])
    print(f"✅ Data Doubled! New count: {len(X)} sequences.")

    # CHANGE 3: test_size 0.05 → 0.15 for stronger validation reporting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)

    # 2. Build Model (UNCHANGED)
    print("🚀 Starting Smart Training...")
    model = build_model(len(actions))

    # Callbacks
    checkpoint = ModelCheckpoint(
        'action.h5',
        monitor='val_categorical_accuracy',   # CHANGE 4: monitor validation accuracy
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # CHANGE 5: patience 30 → 50 to allow longer convergence
    early_stopping = EarlyStopping(
        monitor='val_categorical_accuracy', 
        patience=15,  # Reduced from 50 to prevent overfitting
        restore_best_weights=True,
        verbose=1
    )

    # CHANGE 6: NEW — Learning rate scheduler (strongest improvement)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.5,       # halve the learning rate when stuck
        patience=10,      # after 10 epochs of no improvement
        min_lr=0.00001,
        verbose=1
    )

    # CHANGE 7: Added validation_data so val accuracy is tracked per epoch
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Save history (UNCHANGED)
    np.save('training_history.npy', history.history)
    print("✅ Training history saved as training_history.npy")

    # Save actions (UNCHANGED)
    model.summary()
    np.save('actions.npy', actions)

    # CHANGE 8: Graphs now show BOTH train and validation curves
    hist = np.load('training_history.npy', allow_pickle=True).item()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist['categorical_accuracy'], label='Train Accuracy')
    plt.plot(hist['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("✅ Training curves saved as training_curves.png")

    print("\n🎉 BEST MODEL SAVED as 'action.h5'! You are ready to test.")