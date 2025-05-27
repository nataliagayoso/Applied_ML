import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report  # Optional


data_dir = 'PetImages'
categories = ['Cats', 'Dogs']
image_size = (64, 64)
apply_pca = True
# Confirm with curve; we need to give a reasoning for this
pca_components = 500

filepaths = []
labels = []

# Label the data
for label in categories:
    folder = os.path.join(data_dir, label)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg')):
            filepaths.append(os.path.join(folder, fname))
            labels.append(0 if label == 'Cats' else 1)

X = []
Y = []

corrupt_files = []

# Pre-process if not corrupt, othwerwise ignore
for i, path in enumerate(filepaths):
    try:
        img = Image.open(path)
        resized_img = img.resize((64, 64))
        img_array = np.array(resized_img) / 255.0
        flattened_img = img_array.flatten()
        X.append(flattened_img)
        Y.append(labels[i])
    except IOError:
        print(f"Skip corrupt file: {path}")
        corrupt_files.append(path)
        continue

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

accuracies = []
f1_scores = []

PCA(n_components=pca_components)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X, Y)

for train_idx, validation_idx in skf.split(X, Y):
    X_train, X_val = X[train_idx], X[validation_idx]
    Y_train, Y_val = Y[train_idx], Y[validation_idx]

    if apply_pca:
        pca = PCA(n_components=pca_components)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, Y_train)

    predictions = random_forest.predict(X_val)
    accuracy = accuracy_score(Y_val, predictions)
    f1 = f1_score(Y_val, predictions)

    accuracies.append(accuracy)
    f1_scores.append(f1)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

# Save the baseline model results into a dictionary
baseline_results = {
    'model': 'Random Forest Classifier',
    'mean_accuracy': mean_accuracy,
    'std_accuracy': std_accuracy,
    'mean_f1': mean_f1,
    'std_f1': std_f1
}

# Convert it to a Data Frame (optional, maybe not?)
baseline_results_df = pd.DataFrame([baseline_results])

# Classification report, also optional and maybe too much for a baseline
report = classification_report(Y_val, predictions, target_names=categories)
