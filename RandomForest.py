import time
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA

# Paths & classes
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR  = SCRIPT_DIR / 'data' / 'ProcessedResizedNorm' / 'train'
categories = ['Cats', 'Dogs']

# Gather .npy filepaths & labels
filepaths, labels = [], []
for idx, cls in enumerate(categories):
    cls_folder = TRAIN_DIR / cls
    for p in cls_folder.glob('*.npy'):
        filepaths.append(p)
        labels.append(idx)

print(f"Found {len(filepaths)} samples: {labels.count(0)} Cats, {labels.count(1)} Dogs")

# Load and flatten into X
# Each arr might be shape (224,224,3); flatten → (224*224*3,)
X = np.stack([np.load(p).astype(np.float32).flatten() for p in filepaths])
Y = np.array(labels, dtype=np.int64)

# 5-fold CV with progress
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, f1s = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y), 1):
    print(f"\n=== Fold {fold}/5 ===")
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # PCA
    print(" - Fitting PCA ...", end="", flush=True)
    t0 = time.time()
    pca = PCA(n_components=500).fit(X_train)
    X_train = pca.transform(X_train)
    X_val   = pca.transform(X_val)
    print(f" done in {time.time()-t0:.1f}s")

    # Random Forest
    print(" - Training Random Forest ...", end="", flush=True)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1, random_state=42)
    t1 = time.time()
    rf.fit(X_train, Y_train)
    print(f" done in {time.time()-t1:.1f}s")

    # Evaluate
    preds = rf.predict(X_val)
    acc   = accuracy_score(Y_val, preds)
    f1    = f1_score(Y_val, preds)
    print(f" - Fold {fold} Accuracy: {acc:.3f}, F1: {f1:.3f}")

    accuracies.append(acc)
    f1s.append(f1)

# 5) Final summary
print(f"\nOverall Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"Overall F1-score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}\n")
print("Classification report on last fold:")
print(classification_report(Y_val, preds, target_names=categories))

