import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Configuration
SCRIPT_DIR     = Path(__file__).resolve().parent
TRAIN_DIR      = SCRIPT_DIR / 'data' / 'ProcessedResizedNorm' / 'train'
CATEGORIES     = ['Cats', 'Dogs']
APPLY_PCA      = True
PCA_COMPONENTS = 100      
RF_TREES       = 30
KFOLD_SPLITS   = 5

# Load file paths & labels
filepaths, labels = [], []
for idx, cls in enumerate(CATEGORIES):
    folder = TRAIN_DIR / cls
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    for p in folder.glob('*.npy'):
        filepaths.append(p)
        labels.append(idx)

print(f"Found {len(filepaths)} samples: {labels.count(0)} Cats, {labels.count(1)} Dogs")

# Build feature matrix
X = np.stack([np.load(p).astype(np.float32).flatten() for p in filepaths])
Y = np.array(labels, dtype=np.int64)

# Cross‐validation
skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
accs, f1s = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y), 1):
    X_tr, X_va = X[train_idx], X[val_idx]
    Y_tr, Y_va = Y[train_idx], Y[val_idx]

    if APPLY_PCA:
        pca = PCA(n_components=PCA_COMPONENTS).fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_va = pca.transform(X_va)

    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_tr, Y_tr)

    preds = rf.predict(X_va)
    acc   = accuracy_score(Y_va, preds)
    f1    = f1_score(Y_va, preds)
    print(f"Fold {fold}: Accuracy={acc:.3f}, F1={f1:.3f}")
    accs.append(acc); f1s.append(f1)

print(f"Overall: Accuracy={np.mean(accs):.3f}±{np.std(accs):.3f}, "
      f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

# Retrain on full data and save the pipeline
pipeline = Pipeline([
    ('pca', PCA(n_components=PCA_COMPONENTS)),
    ('rf', RandomForestClassifier(
        n_estimators=RF_TREES,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    ))
])
pipeline.fit(X, Y)

model_dir = SCRIPT_DIR / 'models'
model_dir.mkdir(exist_ok=True)
out_path = model_dir / 'rf_pipeline.joblib'
joblib.dump(pipeline, out_path)
print(f"Saved trained pipeline to {out_path}")


