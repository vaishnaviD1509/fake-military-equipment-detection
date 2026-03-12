import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ============================================================
# CONFIGURATION — CHANGE ONLY THESE
# ============================================================

REAL_TRAIN_FOLDER = r"C:\Users\Sharanabasava\Desktop\Workshop\equipment\real_equipment\train"
REAL_TEST_FOLDER  = r"C:\Users\Sharanabasava\Desktop\Workshop\equipment\real_equipment\test"
FAKE_TRAIN_FOLDER = r"C:\Users\Sharanabasava\Desktop\Workshop\equipment\fake_equipment\train"
FAKE_TEST_FOLDER  = r"C:\Users\Sharanabasava\Desktop\Workshop\equipment\fake_equipment\test"

IMG_SIZE         = 128   # increased from 64 for better detail
EPOCHS           = 50    # increased from 20 for better learning
BATCH_SIZE       = 16
THRESHOLD        = None  # None = auto-detect
MODEL_SAVE_PATH  = "fake_equipment_detector.keras"
THRESHOLD_PATH   = "threshold.json"

# ============================================================
# 1. LOAD IMAGES
# ============================================================

print("=" * 55)
print("  FAKE EQUIPMENT DETECTOR — Autoencoder")
print("=" * 55)

def load_images(folder):
    images    = []
    filenames = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(folder):
        print(f"❌ Folder not found → {folder}")
        return np.array([]), []

    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(valid_ext):
                path = os.path.join(root, f)
                try:
                    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
                    img = img_to_array(img) / 255.0
                    images.append(img)
                    filenames.append(path)
                except Exception as e:
                    print(f"  ⚠️  Skipped {f} — {e}")

    print(f"  ✅ Loaded {len(images)} images from {os.path.basename(folder)}")
    return np.array(images), filenames

print("\n📂 Loading images...")
X_real_train, real_train_files = load_images(REAL_TRAIN_FOLDER)
X_real_test,  real_test_files  = load_images(REAL_TEST_FOLDER)
X_fake_train, fake_train_files = load_images(FAKE_TRAIN_FOLDER)
X_fake_test,  fake_test_files  = load_images(FAKE_TEST_FOLDER)

if len(X_real_train) == 0:
    print("❌ No real training images found. Check REAL_TRAIN_FOLDER path.")
    exit()

print(f"\n📊 Dataset Summary:")
print(f"   REAL train : {len(X_real_train)}  ← used for training")
print(f"   REAL test  : {len(X_real_test)}   ← used for evaluation")
print(f"   FAKE train : {len(X_fake_train)}  ← used for evaluation only")
print(f"   FAKE test  : {len(X_fake_test)}   ← used for evaluation only")

# ============================================================
# 2. BUILD AUTOENCODER
# ============================================================

def build_autoencoder(img_size):
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(2)(x)

    # Decoder
    x = layers.Conv2D(64,  3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64,  3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32,  3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs, decoded, name="Autoencoder")
    model.compile(optimizer='adam', loss='mse')
    return model

print("\n🔨 Building Autoencoder...")
autoencoder = build_autoencoder(IMG_SIZE)
autoencoder.summary()

# ============================================================
# 3. TRAIN ON REAL IMAGES ONLY
# ============================================================

print("\n🔧 Training on REAL images only...")
history = autoencoder.fit(
    X_real_train, X_real_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)
print("  ✅ Training complete\n")

# ============================================================
# 4. SAVE MODEL
# ============================================================

autoencoder.save(MODEL_SAVE_PATH)
print(f"💾 Model saved to: {MODEL_SAVE_PATH}")

# ============================================================
# 5. COMPUTE THRESHOLD
# ============================================================

print("\n🔍 Computing threshold from real training images...")
recon_train  = autoencoder.predict(X_real_train, verbose=0)
errors_train = np.mean((X_real_train - recon_train) ** 2, axis=(1, 2, 3))

if THRESHOLD is None:
    threshold = float(errors_train.mean() + errors_train.std())
else:
    threshold = THRESHOLD

print(f"\n📊 REAL Train Error Stats:")
print(f"   Min       : {errors_train.min():.6f}")
print(f"   Max       : {errors_train.max():.6f}")
print(f"   Mean      : {errors_train.mean():.6f}")
print(f"   Std Dev   : {errors_train.std():.6f}")
print(f"   Threshold : {threshold:.6f}  (mean + 1×std)")

with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": threshold, "img_size": IMG_SIZE}, f)
print(f"💾 Threshold saved to: {THRESHOLD_PATH}")

# ============================================================
# 6. EVALUATE ON ALL TEST + FAKE TRAIN SETS
# ============================================================

all_errors = []
all_labels = []
all_files  = []

def evaluate(images, filenames, true_label):
    if len(images) == 0:
        return
    recon  = autoencoder.predict(images, verbose=0)
    errors = np.mean((images - recon) ** 2, axis=(1, 2, 3))
    for fname, err in zip(filenames, errors):
        pred     = "FAKE" if err > threshold else "REAL"
        expected = "FAKE" if true_label == 1 else "REAL"
        mark     = "✓" if pred == expected else "✗"
        icon     = "⚠️  FAKE" if pred == "FAKE" else "✅ REAL"
        print(f"  {icon:<12}  {err:.6f}   {os.path.basename(fname)} {mark}")
        all_errors.append(err)
        all_labels.append(true_label)
        all_files.append(fname)

print("\n" + "=" * 65)
print(f"  {'RESULT':<12} {'ERROR':>10}   FILENAME")
print("=" * 65)
print("\n── REAL TEST ──")
evaluate(X_real_test, real_test_files, 0)
print("\n── FAKE TRAIN ──")
evaluate(X_fake_train, fake_train_files, 1)
print("\n── FAKE TEST ──")
evaluate(X_fake_test, fake_test_files, 1)
print("=" * 65)

if all_labels:
    preds   = [1 if e > threshold else 0 for e in all_errors]
    correct = sum(p == t for p, t in zip(preds, all_labels))
    total   = len(all_labels)
    print(f"\n🎯 Overall Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

# ============================================================
# 7. PLOT: Training Loss
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
plt.title('Autoencoder Training Loss', fontsize=13)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.close()
print("\n📈 Saved: training_loss.png")

# ============================================================
# 8. PLOT: Error distribution (max 50 images)
# ============================================================

if all_errors:
    sorted_idx = np.argsort(all_errors)
    max_show   = 50

    if len(sorted_idx) > max_show:
        show_idx = np.concatenate([sorted_idx[:max_show//2], sorted_idx[-max_show//2:]])
    else:
        show_idx = sorted_idx

    bar_colors = ['#e74c3c' if all_labels[i] == 1 else '#2ecc71' for i in show_idx]
    bar_labels = [os.path.basename(all_files[i]) for i in show_idx]
    bar_vals   = [all_errors[i] for i in show_idx]

    fig, ax = plt.subplots(figsize=(min(20, max(10, len(show_idx) * 0.4)), 5))
    ax.bar(range(len(bar_vals)), bar_vals, color=bar_colors, edgecolor='white')
    ax.axhline(threshold, color='navy', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold:.4f})')
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title(f'REAL vs FAKE Error Distribution (showing {len(show_idx)} of {len(all_errors)})', fontsize=13)
    fake_patch = mpatches.Patch(color='#e74c3c', label='FAKE')
    real_patch = mpatches.Patch(color='#2ecc71', label='REAL')
    ax.legend(handles=[real_patch, fake_patch])
    plt.tight_layout()
    plt.savefig("error_distribution.png", dpi=150)
    plt.close()
    print("📊 Saved: error_distribution.png")

# ============================================================
# 9. PLOT: real_and_fake_images.png
# ============================================================

n_show      = 5
real_indices = [i for i, l in enumerate(all_labels) if l == 0]
fake_indices = [i for i, l in enumerate(all_labels) if l == 1]

real_sorted = sorted(real_indices, key=lambda i: all_errors[i])[:n_show]
fake_sorted = sorted(fake_indices, key=lambda i: all_errors[i], reverse=True)[:n_show]

n_real = len(real_sorted)
n_fake = len(fake_sorted)
n_cols = max(n_real, n_fake, 1)

if n_real > 0 or n_fake > 0:
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 7))
    fig.suptitle('Real vs Fake Equipment — Model Predictions',
                 fontsize=14, fontweight='bold', y=1.01)

    for i in range(n_cols):
        ax = axes[0, i] if n_cols > 1 else axes[0]
        if i < n_real:
            idx = real_sorted[i]
            img = load_img(all_files[idx], target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            ax.imshow(img)
            ax.set_title(f"✅ REAL\nerr={all_errors[idx]:.4f}",
                         fontsize=8, color='#2ecc71', fontweight='bold')
        ax.axis('off')

    for i in range(n_cols):
        ax = axes[1, i] if n_cols > 1 else axes[1]
        if i < n_fake:
            idx = fake_sorted[i]
            img = load_img(all_files[idx], target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            ax.imshow(img)
            ax.set_title(f"⚠️ FAKE\nerr={all_errors[idx]:.4f}",
                         fontsize=8, color='#e74c3c', fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("real_and_fake_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("🖼️  Saved: real_and_fake_images.png")

print("\n✅ All done!")
print(f"💾 Model      : {MODEL_SAVE_PATH}")
print(f"💾 Threshold  : {THRESHOLD_PATH}  ({threshold:.6f})")
print(f"📸 Plots      : training_loss.png, error_distribution.png, real_and_fake_images.png")