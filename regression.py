import gc
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# SETTINGS (stable training for regression)
# ============================================================

IMG_SIZE = 128
NUM_SAMPLES = 10000      # αν αργεί -> 8000
SEED = 42

AA_FACTOR = 2            # 2: καλό, 4: πιο λείο αλλά πιο αργό
NOISE_STD = 0.01         # global noise σε όλη την εικόνα

# Stable LR per phase
LR_PHASE1 = 2e-4
LR_PHASE2 = 8e-5

# ---- PHASES: Phase 2 much larger ----
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 80
BATCH_SIZE = 64

# BatchNorm stability
BN_MOMENTUM = 0.90
BN_EPS = 1e-3

# Gradient clipping
CLIPNORM = 1.0

# Robust losses
HUBER_I_DELTA = 0.02
HUBER_C_DELTA = 0.02

# Curriculum configs
PHASE1_CFG = {
    "intensity_jitter_std": 0.020,
    "min_visible_intensity": 0.20,
    "global_pixel_noise_std": 0.000,
}
PHASE2_CFG = {
    "intensity_jitter_std": 0.030,
    "min_visible_intensity": 0.18,
    "global_pixel_noise_std": NOISE_STD,
}
EVAL_CFG = {
    "intensity_jitter_std": 0.030,
    "min_visible_intensity": 0.18,
    "global_pixel_noise_std": NOISE_STD,
}

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# GENERATOR HELPERS
# ============================================================

def sample_intensity(rng):
    label = rng.integers(0, 3)
    if label == 0:
        return float(rng.uniform(0.10, 0.35))
    elif label == 1:
        return float(rng.uniform(0.35, 0.60))
    else:
        return float(rng.uniform(0.60, 0.95))

def jitter_intensity(I, rng, std=0.03, min_visible=0.18):
    Ij = I + rng.normal(0.0, std)
    Ij = np.clip(Ij, min_visible, 1.0)
    return float(Ij)

def sample_axes(rng, min_r=5, max_r=25, min_diff=3.0):
    while True:
        a = float(rng.uniform(min_r, max_r))
        b = float(rng.uniform(min_r, max_r))
        if abs(a - b) >= min_diff:
            return a, b

def ellipse_mask(size, cx, cy, a, b, theta):
    y, x = np.ogrid[:size, :size]
    x_shift = x - cx
    y_shift = y - cy

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    xr = cos_t * x_shift + sin_t * y_shift
    yr = -sin_t * x_shift + cos_t * y_shift

    return (xr**2) / (a**2) + (yr**2) / (b**2) <= 1.0

def ellipse_alpha_mask(size, cx, cy, a, b, theta, aa_factor=2):
    if aa_factor <= 1:
        return ellipse_mask(size, cx, cy, a, b, theta).astype(np.float32)

    H = size
    W = size
    f = aa_factor
    Hh, Wh = H * f, W * f

    y_hr, x_hr = np.ogrid[:Hh, :Wh]
    x = (x_hr + 0.5) / f - 0.5
    y = (y_hr + 0.5) / f - 0.5

    x_shift = x - cx
    y_shift = y - cy

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    xr = cos_t * x_shift + sin_t * y_shift
    yr = -sin_t * x_shift + cos_t * y_shift

    mask_hr = ((xr**2) / (a**2) + (yr**2) / (b**2) <= 1.0).astype(np.float32)
    alpha = mask_hr.reshape(H, f, W, f).mean(axis=(1, 3)).astype(np.float32)
    return alpha

def sample_center_inside(rng, a, b, theta, size=128):
    hx = np.sqrt(a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2)
    hy = np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2)
    cx = float(rng.uniform(hx, size - hx))
    cy = float(rng.uniform(hy, size - hy))
    return cx, cy

def sample_latent_two_ellipses(rng, size=128):
    a1, b1 = sample_axes(rng, 5, 25, min_diff=3.0)
    t1 = float(rng.uniform(0, np.pi))
    cx1, cy1 = sample_center_inside(rng, a1, b1, t1, size)
    I1_raw = sample_intensity(rng)
    mask1 = ellipse_mask(size, cx1, cy1, a1, b1, t1)

    while True:
        a2, b2 = sample_axes(rng, 5, 25, min_diff=3.0)
        t2 = float(rng.uniform(0, np.pi))
        cx2, cy2 = sample_center_inside(rng, a2, b2, t2, size)
        I2_raw = sample_intensity(rng)
        mask2 = ellipse_mask(size, cx2, cy2, a2, b2, t2)
        if not np.any(mask1 & mask2):
            break

    latent = np.array([
        a1, b1, t1, cx1, cy1, I1_raw,
        a2, b2, t2, cx2, cy2, I2_raw
    ], dtype=np.float32)
    return latent

def generate_latent_pool(num_samples, size=128, seed=42):
    rng = np.random.default_rng(seed)
    latents = np.zeros((num_samples, 12), dtype=np.float32)
    for i in range(num_samples):
        latents[i] = sample_latent_two_ellipses(rng, size=size)
    return latents

def render_from_latent(latent, cfg, rng, size=128, aa_factor=2):
    a1, b1, t1, cx1, cy1, I1_raw, a2, b2, t2, cx2, cy2, I2_raw = latent

    I1 = jitter_intensity(I1_raw, rng, std=cfg["intensity_jitter_std"], min_visible=cfg["min_visible_intensity"])
    I2 = jitter_intensity(I2_raw, rng, std=cfg["intensity_jitter_std"], min_visible=cfg["min_visible_intensity"])

    if I1 <= I2:
        I_low, I_high = I1, I2
        cx_low, cy_low = cx1, cy1
        cx_high, cy_high = cx2, cy2
        low_params = (a1, b1, t1, cx1, cy1)
        high_params = (a2, b2, t2, cx2, cy2)
    else:
        I_low, I_high = I2, I1
        cx_low, cy_low = cx2, cy2
        cx_high, cy_high = cx1, cy1
        low_params = (a2, b2, t2, cx2, cy2)
        high_params = (a1, b1, t1, cx1, cy1)

    aL, bL, tL, cxL, cyL = low_params
    aH, bH, tH, cxH, cyH = high_params

    alpha_low = ellipse_alpha_mask(size, cxL, cyL, aL, bL, tL, aa_factor=aa_factor)
    alpha_high = ellipse_alpha_mask(size, cxH, cyH, aH, bH, tH, aa_factor=aa_factor)
    alpha_high = alpha_high * (1.0 - alpha_low)

    img = np.zeros((size, size), dtype=np.float32)
    img = img * (1.0 - alpha_low) + I_low * alpha_low
    img = img * (1.0 - alpha_high) + I_high * alpha_high

    std_pix = cfg.get("global_pixel_noise_std", 0.0)
    if std_pix > 0:
        noise = rng.normal(0.0, std_pix, size=(size, size)).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    y = np.array([
        I_low, I_high,
        cx_low / size, cy_low / size,
        cx_high / size, cy_high / size
    ], dtype=np.float32)

    return img.astype(np.float32), y

def build_dataset_from_latents(latents, cfg, size=128, seed=0, aa_factor=2):
    rng = np.random.default_rng(seed)
    N = len(latents)
    X = np.zeros((N, size, size, 1), dtype=np.float32)
    Y = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        img, y = render_from_latent(latents[i], cfg, rng, size=size, aa_factor=aa_factor)
        X[i, :, :, 0] = img
        Y[i] = y
    return X, Y

# ============================================================
# DISPLAY: images + table 
# ============================================================

def show_test_examples_with_tables(X_test, y_test, y_pred_full, img_size=128, k=3):
    """
    Εμφανίζει ΜΟΝΟ τις εικόνες (ελλείψεις + τίτλος TEST idx),
    και ΑΚΡΙΒΩΣ από κάτω το πινακάκι True/Pred/Diff (όπως ήταν).
    """
    names = ["Ilow", "Ihigh", "cx_low", "cy_low", "cx_high", "cy_high"]

    k = min(k, len(X_test))
    idx = np.arange(len(X_test) - k, len(X_test))

    fig = plt.figure(figsize=(5.6 * k, 7.0))
    gs = fig.add_gridspec(2, k, height_ratios=[3.2, 2.0], hspace=0.25, wspace=0.25)

    def fmt_val(name, v):
        if name.startswith("cx") or name.startswith("cy"):
            return f"{v:.4f} ({v*img_size:.1f}px)"
        return f"{v:.4f}"

    for col, j in enumerate(idx):
        y_t = y_test[j]
        y_p = y_pred_full[j]
        diff = y_t - y_p  # True - Pred

        # --- image μόνο ---
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(X_test[j].squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax_img.axis("off")
        ax_img.set_title(f"TEST idx={j}", fontsize=10)

        # --- table (ίδιο όπως ήταν) ---
        ax_tbl = fig.add_subplot(gs[1, col])
        ax_tbl.axis("off")

        cell_text = []
        for i, n in enumerate(names):
            cell_text.append([fmt_val(n, y_t[i]), fmt_val(n, y_p[i]), fmt_val(n, diff[i])])

        table = ax_tbl.table(
            cellText=cell_text,
            rowLabels=names,
            colLabels=["True", "Pred", "Diff (T-P)"],
            loc="center",
            cellLoc="center",
            colLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)

    plt.tight_layout()
    plt.show()

# ============================================================
# COORD CHANNELS INSIDE MODEL
# ============================================================

def add_coord_channels_tf(inp):
    shape = tf.shape(inp)
    batch_size = shape[0]
    h = shape[1]
    w = shape[2]

    xs = tf.linspace(0.0, 1.0, w)
    ys = tf.linspace(0.0, 1.0, h)
    x_grid, y_grid = tf.meshgrid(xs, ys)

    x_grid = tf.cast(x_grid, inp.dtype)
    y_grid = tf.cast(y_grid, inp.dtype)

    x_grid = tf.expand_dims(tf.expand_dims(x_grid, axis=0), axis=-1)
    y_grid = tf.expand_dims(tf.expand_dims(y_grid, axis=0), axis=-1)

    x_grid = tf.tile(x_grid, [batch_size, 1, 1, 1])
    y_grid = tf.tile(y_grid, [batch_size, 1, 1, 1])

    return tf.concat([inp, x_grid, y_grid], axis=-1)

# ============================================================
# MODEL
# ============================================================

def build_model(img_size=128, l2_reg=5e-5):
    inp = layers.Input((img_size, img_size, 1))
    x = layers.Lambda(add_coord_channels_tf, name="coord_concat")(inp)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)

    out_I = layers.Dense(2, activation="sigmoid", name="I_out")(x)
    out_C = layers.Dense(4, activation="sigmoid", name="C_out")(x)

    model = tf.keras.Model(inputs=inp, outputs=[out_I, out_C])

    huber_I = tf.keras.losses.Huber(delta=HUBER_I_DELTA)
    huber_C = tf.keras.losses.Huber(delta=HUBER_C_DELTA)

    model.compile(
        optimizer=Adam(learning_rate=LR_PHASE1, clipnorm=CLIPNORM),
        loss={"I_out": huber_I, "C_out": huber_C},
        loss_weights={"I_out": 1.0, "C_out": 6.0},
        metrics={"I_out": ["mae"], "C_out": ["mae"]}
    )
    return model

# ============================================================
# HISTORY MERGE + PLOTS
# ============================================================

def merge_histories(histories):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, [])
            merged[k].extend(v)
    return merged

def plot_training_history(hist_merged):
    epochs_arr = np.arange(1, len(hist_merged["loss"]) + 1)

    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, hist_merged["loss"], label="Train Loss")
    plt.plot(epochs_arr, hist_merged["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, hist_merged["I_out_mae"], label="Train MAE (I_out)")
    plt.plot(epochs_arr, hist_merged["val_I_out_mae"], label="Val MAE (I_out)")
    plt.plot(epochs_arr, hist_merged["C_out_mae"], label="Train MAE (C_out)")
    plt.plot(epochs_arr, hist_merged["val_C_out_mae"], label="Val MAE (C_out)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE per head")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_validation_loss_only(hist_merged, phase1_epochs=None):
    val_loss = hist_merged["val_loss"]
    epochs_arr = np.arange(1, len(val_loss) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_arr, val_loss, label="Validation Loss")

    if phase1_epochs is not None:
        plt.axvline(phase1_epochs + 0.5, linestyle="--", linewidth=2)
        plt.text(phase1_epochs + 1, max(val_loss) * 0.95, "Phase2 start")

    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

print(f"Generating latent pool ({NUM_SAMPLES} samples)...")
latents = generate_latent_pool(NUM_SAMPLES, size=IMG_SIZE, seed=SEED)

all_idx = np.arange(NUM_SAMPLES)
idx_train, idx_tmp = train_test_split(all_idx, test_size=0.30, random_state=SEED)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)

lat_train = latents[idx_train]
lat_val   = latents[idx_val]
lat_test  = latents[idx_test]

print(f"Train/Val/Test sizes: {len(lat_train)} / {len(lat_val)} / {len(lat_test)}")

print("Building validation dataset...")
X_val, y_val = build_dataset_from_latents(lat_val, EVAL_CFG, size=IMG_SIZE, seed=SEED+101, aa_factor=AA_FACTOR)

print("Building test dataset...")
X_test, y_test = build_dataset_from_latents(lat_test, EVAL_CFG, size=IMG_SIZE, seed=SEED+202, aa_factor=AA_FACTOR)

y_val_I = y_val[:, :2]
y_val_C = y_val[:, 2:]
y_test_I = y_test[:, :2]
y_test_C = y_test[:, 2:]

print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

model = build_model(img_size=IMG_SIZE, l2_reg=5e-5)
model.summary()

# -------------------- PHASE 1 --------------------
print("\n=== PHASE 1: cleaner training ===")
X_train_p1, y_train_p1 = build_dataset_from_latents(lat_train, PHASE1_CFG, size=IMG_SIZE, seed=SEED+303, aa_factor=AA_FACTOR)
y_train_p1_I = y_train_p1[:, :2]
y_train_p1_C = y_train_p1[:, 2:]

cb_p1 = [
    EarlyStopping(monitor="val_loss", patience=4, min_delta=1e-4, restore_best_weights=True, verbose=1)
]

hist1 = model.fit(
    X_train_p1,
    {"I_out": y_train_p1_I, "C_out": y_train_p1_C},
    validation_data=(X_val, {"I_out": y_val_I, "C_out": y_val_C}),
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=cb_p1,
    verbose=1,
)

del X_train_p1, y_train_p1, y_train_p1_I, y_train_p1_C
gc.collect()

# -------------------- PHASE 2 --------------------
print("\n=== PHASE 2: long fine-tuning with global noise ===")
model.optimizer.learning_rate.assign(LR_PHASE2)

X_train_p2, y_train_p2 = build_dataset_from_latents(lat_train, PHASE2_CFG, size=IMG_SIZE, seed=SEED+404, aa_factor=AA_FACTOR)
y_train_p2_I = y_train_p2[:, :2]
y_train_p2_C = y_train_p2[:, 2:]

cb_p2 = [
    EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, restore_best_weights=True, verbose=1)
]

hist2 = model.fit(
    X_train_p2,
    {"I_out": y_train_p2_I, "C_out": y_train_p2_C},
    validation_data=(X_val, {"I_out": y_val_I, "C_out": y_val_C}),
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=cb_p2,
    verbose=1,
)

del X_train_p2, y_train_p2, y_train_p2_I, y_train_p2_C
gc.collect()

# -------------------- PLOTS --------------------
hist_merged = merge_histories([hist1, hist2])
plot_training_history(hist_merged)
plot_validation_loss_only(hist_merged, phase1_epochs=EPOCHS_PHASE1)

# -------------------- TEST EVALUATION --------------------
eval_res = model.evaluate(X_test, {"I_out": y_test_I, "C_out": y_test_C}, verbose=0)
print("\nTest metrics:")
for name, val in zip(model.metrics_names, eval_res):
    print(f"{name:20s}: {val:.6f}")

# ============================================================
# EXTRA EVALUATION: residuals + sigma + plots (TEST SET)
# ============================================================

# Predict on test
pred_I_test, pred_C_test = model.predict(X_test, verbose=0)
y_pred_full = np.concatenate([pred_I_test, pred_C_test], axis=1)

# -------------------- SHOW 3 TEST IMAGES + TABLES --------------------
show_test_examples_with_tables(X_test, y_test, y_pred_full, img_size=IMG_SIZE, k=3)

names = ["Ilow", "Ihigh", "cx_low", "cy_low", "cx_high", "cy_high"]

# Residuals: True - Pred
res = y_test - y_pred_full

# Bias / MAE / Sigma
bias = np.mean(res, axis=0)
mae  = np.mean(np.abs(res), axis=0)
sig  = np.std(res, axis=0, ddof=1)

print("\n--- Residual stats on TEST (True - Pred) ---")
for i, n in enumerate(names):
    if "cx_" in n or "cy_" in n:
        print(
            f"{n:8s}: bias={bias[i]:+.6f} ({bias[i]*IMG_SIZE:+.2f}px) | "
            f"MAE={mae[i]:.6f} ({mae[i]*IMG_SIZE:.2f}px) | "
            f"σ={sig[i]:.6f} ({sig[i]*IMG_SIZE:.2f}px)"
        )
    else:
        print(f"{n:8s}: bias={bias[i]:+.6f} | MAE={mae[i]:.6f} | σ={sig[i]:.6f}")

# ------------------------------------------------------------
# Plots: Residual histograms
# ------------------------------------------------------------

def plot_residual_hist(r, title, bins=30):
    plt.figure(figsize=(6.2, 4.6))
    plt.hist(r, bins=bins)
    s = np.std(r, ddof=1)
    m = np.mean(r)
    plt.title(f"{title}\nmean={m:+.5f}, σ={s:.5f}")
    plt.xlabel("Residual (True - Pred)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Intensities residuals
plot_residual_hist(res[:, 0], "Residuals Ilow")
plot_residual_hist(res[:, 1], "Residuals Ihigh")

# Centers residuals (normalized)
plot_residual_hist(res[:, 2], "Residuals cx_low (norm)")
plot_residual_hist(res[:, 3], "Residuals cy_low (norm)")
plot_residual_hist(res[:, 4], "Residuals cx_high (norm)")
plot_residual_hist(res[:, 5], "Residuals cy_high (norm)")

# ------------------------------------------------------------
# True vs Pred plots
# ------------------------------------------------------------

def true_pred_hexbin(y_true, y_pred, title, gridsize=45):
    plt.figure(figsize=(5.4, 4.8))
    plt.hexbin(y_true, y_pred, gridsize=gridsize, mincnt=1)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")

    mae_ = np.mean(np.abs(y_true - y_pred))
    bias_ = np.mean(y_pred - y_true)          # Pred - True
    sig_ = np.std(y_true - y_pred, ddof=1)    # True - Pred sigma
    plt.title(f"{title}\nMAE={mae_:.4f}, bias={bias_:+.4f}, σ={sig_:.4f}")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def true_pred_scatter(y_true, y_pred, title):
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(y_true, y_pred, alpha=0.35)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")

    mae_ = np.mean(np.abs(y_true - y_pred))
    bias_ = np.mean(y_pred - y_true)
    sig_ = np.std(y_true - y_pred, ddof=1)
    plt.title(f"{title}\nMAE={mae_:.4f}, bias={bias_:+.4f}, σ={sig_:.4f}")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Centers: hexbins
true_pred_hexbin(y_test[:, 2], y_pred_full[:, 2], "cx_low")
true_pred_hexbin(y_test[:, 3], y_pred_full[:, 3], "cy_low")
true_pred_hexbin(y_test[:, 4], y_pred_full[:, 4], "cx_high")
true_pred_hexbin(y_test[:, 5], y_pred_full[:, 5], "cy_high")

# Intensities: scatter
true_pred_scatter(y_test[:, 0], y_pred_full[:, 0], "Ilow")
true_pred_scatter(y_test[:, 1], y_pred_full[:, 1], "Ihigh")

# ------------------------------------------------------------
# Center distance error in pixels 
# ------------------------------------------------------------

dx_low  = (y_test[:, 2] - y_pred_full[:, 2]) * IMG_SIZE
dy_low  = (y_test[:, 3] - y_pred_full[:, 3]) * IMG_SIZE
d_low   = np.sqrt(dx_low**2 + dy_low**2)

dx_high = (y_test[:, 4] - y_pred_full[:, 4]) * IMG_SIZE
dy_high = (y_test[:, 5] - y_pred_full[:, 5]) * IMG_SIZE
d_high  = np.sqrt(dx_high**2 + dy_high**2)

print("\n--- Center distance error (px) ---")
print(f"low center:  mean={np.mean(d_low):.2f}px | median={np.median(d_low):.2f}px | 90%={np.quantile(d_low,0.90):.2f}px")
print(f"high center: mean={np.mean(d_high):.2f}px | median={np.median(d_high):.2f}px | 90%={np.quantile(d_high,0.90):.2f}px")

plt.figure(figsize=(11, 4.6))
plt.subplot(1, 2, 1)
plt.hist(d_low, bins=30, alpha=0.7, label="low center dist (px)")
plt.hist(d_high, bins=30, alpha=0.7, label="high center dist (px)")
plt.xlabel("Distance error (px)")
plt.ylabel("Count")
plt.title("Center distance error histogram (px)")
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for arr, lab in [(d_low, "low"), (d_high, "high")]:
    s = np.sort(arr)
    cdf = np.arange(1, len(s)+1) / len(s)
    plt.plot(s, cdf, label=lab)
plt.xlabel("Distance error (px)")
plt.ylabel("CDF")
plt.title("Center distance error CDF")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
