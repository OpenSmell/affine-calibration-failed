"""
Affine calibration between two virtual devices (UCI batches).
- Takes log of raw readings to linearise MOX response.
- Uses a small set of calibration samples (first 10 per gas).
- Fits diagonal affine transform (gain and offset per sensor) using Ridge regression.
- Applies transform to target test set, evaluates MSE and classification accuracy.
- Produces PCA plots before/after calibration.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from load_uci import load_uci_data

# ------------------------------
# 1. Load data
# ------------------------------
X_source, y_source, X_target, y_target = load_uci_data()

# Add a small epsilon to avoid log(0) or negative numbers
eps = 1e-6
log_source = np.log(np.maximum(X_source, eps))
log_target = np.log(np.maximum(X_target, eps))

# ------------------------------
# 2. Select calibration samples (first 10 per gas from each device)
# ------------------------------
def get_calibration_indices(y, samples_per_gas=10):
    indices = []
    for gas in range(1, 7):
        gas_idx = np.where(y == gas)[0]
        if len(gas_idx) >= samples_per_gas:
            indices.extend(gas_idx[:samples_per_gas])
    return np.array(indices)

calib_idx_source = get_calibration_indices(y_source)
calib_idx_target = get_calibration_indices(y_target)

# Ensure the same gas labels (they are in same order by construction)
# For safety, we will use the actual gas labels to align, but both devices have same gases.
# We'll simply use the first 10 per gas in order.

X_calib_source = log_source[calib_idx_source]
X_calib_target = log_target[calib_idx_target]

print(f"Calibration points: {len(X_calib_source)}")

# ------------------------------
# 3. Fit diagonal affine transform (gain and offset per sensor)
#    Model: X_source = gain * X_target + offset
# ------------------------------
n_sensors = X_calib_source.shape[1]
gains = np.zeros(n_sensors)
offsets = np.zeros(n_sensors)

for s in range(n_sensors):
    reg = Ridge(alpha=0.1, fit_intercept=True)
    reg.fit(X_calib_target[:, s].reshape(-1, 1), X_calib_source[:, s])
    gains[s] = reg.coef_[0]
    offsets[s] = reg.intercept_

def apply_calibration(X, gains, offsets):
    """Apply diagonal affine transform: X_calib = gain * X + offset"""
    return gains * X + offsets

# ------------------------------
# 4. Split source and target into train/test (use all non-calibration as test)
# ------------------------------
mask_source = np.ones(len(log_source), dtype=bool)
mask_source[calib_idx_source] = False
X_source_test = log_source[mask_source]
y_source_test = y_source[mask_source]

mask_target = np.ones(len(log_target), dtype=bool)
mask_target[calib_idx_target] = False
X_target_test = log_target[mask_target]
y_target_test = y_target[mask_target]

# Apply calibration to target test set
X_target_calib = apply_calibration(X_target_test, gains, offsets)

# ------------------------------
# 5. Evaluation: MSE between calibrated target and source readings (same gas)
#    For each gas, we compute MSE only between samples of same gas.
#    Since we have paired readings? No, we have independent samples.
#    We'll compute overall MSE by matching gas labels? Actually we can't pair samples.
#    Instead, we compute the mean squared error between the transformed target
#    and the source test set for the same gas, but only the means? Simpler:
#    Train a classifier on source, test on calibrated target.
# ------------------------------
# Train SVM classifier on source test set
clf = SVC(kernel='rbf', gamma='scale', random_state=42)
clf.fit(X_source_test, y_source_test)
pred_before = clf.predict(X_target_test)
acc_before = accuracy_score(y_target_test, pred_before)

# After calibration
pred_after = clf.predict(X_target_calib)
acc_after = accuracy_score(y_target_test, pred_after)
print(f"Classification accuracy on target device:")
print(f"  Before calibration: {acc_before:.4f}")
print(f"  After calibration:  {acc_after:.4f}")

# Also compute MSE between transformed target and source (but need paired data? Not available)
# We'll compute the difference between class centroids as a rough measure.
def class_centroids(X, y):
    centroids = []
    for gas in range(1, 7):
        idx = np.where(y == gas)[0]
        centroids.append(np.mean(X[idx], axis=0))
    return np.array(centroids)

cent_source = class_centroids(X_source_test, y_source_test)
cent_target_before = class_centroids(X_target_test, y_target_test)
cent_target_after = class_centroids(X_target_calib, y_target_test)

mse_before = np.mean((cent_source - cent_target_before)**2)
mse_after = np.mean((cent_source - cent_target_after)**2)
print(f"Mean squared error between class centroids (source vs target):")
print(f"  Before calibration: {mse_before:.4f}")
print(f"  After calibration:  {mse_after:.4f}")

# ------------------------------
# 6. Visualisation: PCA of source, target before/after
# ------------------------------
pca = PCA(n_components=2)
all_data = np.vstack([X_source_test, X_target_test, X_target_calib])
pca.fit(all_data)

source_2d = pca.transform(X_source_test)
target_before_2d = pca.transform(X_target_test)
target_after_2d = pca.transform(X_target_calib)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for gas in range(1, 7):
    idx = np.where(y_source_test == gas)[0]
    plt.scatter(source_2d[idx, 0], source_2d[idx, 1], label=f"Gas {gas}", alpha=0.6)
plt.title("Source device (batches 1-5)")
plt.legend()

plt.subplot(1, 3, 2)
for gas in range(1, 7):
    idx = np.where(y_target_test == gas)[0]
    plt.scatter(target_before_2d[idx, 0], target_before_2d[idx, 1], label=f"Gas {gas}", alpha=0.6)
plt.title("Target device BEFORE calibration")
plt.legend()

plt.subplot(1, 3, 3)
for gas in range(1, 7):
    idx = np.where(y_target_test == gas)[0]
    plt.scatter(target_after_2d[idx, 0], target_after_2d[idx, 1], label=f"Gas {gas}", alpha=0.6)
plt.title("Target device AFTER calibration")
plt.legend()

plt.tight_layout()
plt.savefig("calibration_pca.png", dpi=150)
plt.show()
print("PCA plot saved as calibration_pca.png")