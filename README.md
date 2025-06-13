# Animal Binary Classifier - Applied ML

**Authors:**  
- Natalie Mladenova – s5161517 
- Natalia Gayoso – s5221218   
- Raul Ardelean – s5237688  

## 1. Overview
We build a machine-learning pipeline to distinguish cats from dogs in images. Using a real-world 
Kaggle dataset of 25 000 JPG/PNG images (12 500 cats, 12 500 dogs), we’ll clean, preprocess, train 
both a Random Forest baseline and a fine-tuned ResNet CNN, and evaluate with a suite of metrics to measure performance and robustness.

## 2. Preprocessing 

1. **Loading & Cleaning** (`remove.py`)  
   - Scan `PetImages/{Cat,Dog}` with PIL `verify()`, move unreadable files to `Corrupt/` and log them.  
2. **Train/Val/Test Split** (`split.py`)  
   - Stratified 80 % train / 10 % val / 10 % test with `random_state=42`.  
3. **Resizing & Normalization** (`resize.py`)  
   - Resize all images to 224×224 via bilinear interpolation.  
   - Convert to RGB and save as `.npy` arrays of `float32` in [0,1].  
4. **Augmentation** (during CNN training)  
   - Random flips, rotations, crops, color jitter, Gaussian noise.  
5. **Class-Imbalance Handling**  
   - Monitor post-cleaning counts; apply oversampling or class-weighted loss if needed.

## 3. Baseline Model (`RandomForest.py`)
- **Preprocessing:**  
  - Resize to 224x224, flatten to 1D vectors, normalize [0,1].  
  - Optional PCA (500 components) to reduce from ~12 288 dims → 500 dims.  
- **Model:**  
  - 100-tree Random Forest, 5-fold stratified CV.  
- **Results:**  
  | Fold | Accuracy | F1-score |  
  |:----:|:--------:|:--------:|  
  | 1    | 0.610    | 0.601    |  
  | 2    | 0.606    | 0.598    |  
  | 3    | 0.614    | 0.606    |  
  | 4    | 0.611    | 0.609    |  
  | 5    | 0.615    | 0.612    |  
  | **Mean ± Std** | **0.611 ± 0.003** | **0.605 ± 0.005** |  
- **Baseline Insight:** ≈61 % accuracy, consistent across folds; serves as a reference.

## 4. Proposed CNN Model (cnn1.py)
- **Architecture:**  
  - Basic CNN with 3 convolutional layers, each followed by ReLU activation and max pooling.  
  - Two fully connected layers for classification.

- **Preprocessing:**  
  - Images resized to 224×224, normalized to [0, 1].  
  - Augmentation (only during training): horizontal flip, small rotations.

- **Training:**  
  - Adam optimizer, CrossEntropyLoss, 10 epochs, batch size of 32.  
  - Early stopping based on validation F1-score (patience = 3 epochs).  
  - Metrics logged per epoch: loss, accuracy, F1-score.

- **Explainability:**  
  - Grad-CAM visualizations added for 3 random validation samples, can be modified to use more, using the final convolutional layer.  
  - Helps identify what parts of the image influenced the model's predictions.
 
- **Results:**  
  | Epoch | Accuracy | F1-score | Loss |
  |:-----:|:--------:|:--------:|:----:|
  | 1   |   0.726    |   0.727    | 0.6136  |
  | 2   |   0.754    |   0.748    | 0.5373  |
  | 3   |   0.787    |   0.794    | 0.4837  |
  | 4   |   0.805    |   0.800    | 0.4483  |
  | 5   |   0.816    |   0.805    | 0.4215  |
  | 6   |   0.821    |   0.828    | 0.4005  |
  | 7   |   0.830    |   0.821    | 0.3809  |
  | 8   |   0.838    |   0.834    | 0.3663  |
  | 9   |   0.833    |   0.843    | 0.3497  |
  | 10   |   0.842    |   0.852    | 0.3363  |
  | **Mean ± Std** | **0.805 ± 0.037** | **0.805 ± 0.039** | **0.434 ± 0.084** |

  > *Model consistently outperforms the Random Forest baseline by ~19% in accuracy.*

## 5. Comparison & Evaluation

| Model           | Accuracy (± std) | Confidence Interval | Inference | Explainability |
|----------------|------------------|------------------|-----------|----------------|
| Random Forest  | 0.611 ± 0.003    | [0.608, 0.614]    | Fast      | ✗              |
| CNN + Grad-CAM | 0.805 ± 0.037    | [0.782, 0.828]    | Slower    | ✓ (Grad-CAM)   |

- CNN shows a clear performance boost.  
- Grad-CAM provides visual interpretability.  
- RF remains faster but weaker in performance.

## API Usage
FastAPI-powered REST API for classifying images of cats and dogs using two models:

- Random Forest (baseline)

- CNN (Convolutional Neural Network) with optional Grad-CAM visualization

### Running the API
uvicorn api:app --reload

Should see message like:

Uvicorn running on http://127.0.0.1:8000 

Then:

Swagger UI: http://127.0.0.1:8000/docs – for interactive testing


### Endpoints
- GET /
     {"status": "running"}
- POST /predict_rf
  {
  "label": "Cat",
  "probability": 0.87
   }
- POST /predict_cnn
  {
  "label": "Dog",
  "probability": 0.91
  }
- POST /predict_cnn_cam
  {
  "label": "Dog",
  "probability": 0.91,
  "gradcam": "<base64_encoded_image>"
   }

### Error Handling 
Invalid or missing files return 400 Bad Request

All errors include a clear JSON message, e.g.: {"detail": "Only image files accepted."}





  
## Installation
Install the required packages: 
pip install -r requirements.txt


