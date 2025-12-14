# Weather-Classification
# Image Classification via Transfer Learning with ResNet-50 and Classical ML

This repository contains a three-part Python project implementing an image classification pipeline using **Transfer Learning**. It leverages a pre-trained **ResNet-50** model to extract high-level features (embeddings) from images and then trains traditional machine learning classifiers (**Logistic Regression** and **XGBoost**) on those features.

The project is structured into three files: `Preprocessing.py` (data handling), `main.py` (feature extraction), and `model.py` (training and evaluation).

## Project Architecture

The pipeline follows a two-stage approach for efficient classification:

1.  **Feature Extraction (Transfer Learning):** A pre-trained ResNet-50 model (with `IMAGENET1K_V2` weights) is used as a fixed feature extractor.
    * The model's parameters are frozen (`requires_grad = False`).
    * The output of the model is saved as a numerical array of features (`.npy` files) for training and testing.

2.  **Classification:** Two classical ML models are trained on the extracted features: **XGBoost** and **Logistic Regression**.
    * The models are evaluated using a `classification_report`.
    * Both trained models are saved for later inference.

## File Descriptions

| File Name | Role | Description |
| :--- | :--- | :--- |
| `Preprocessing.py` | Data & Utilities | Contains functions for loading image file paths (`load_all`, `data_loading`), defining the PyTorch `CustomDataset` (resizing images to 300x300 and converting to tensors), creating the `DataLoader`, and the core `feature_extraction` function. |
| `main.py` | Feature Extraction | Loads the ResNet-50 model, sets the execution device (`cuda` or `cpu`), loads data from hardcoded training and testing paths, runs the feature extraction pipeline, and saves the resulting embeddings and labels into the `Data/` folder as `.npy` files. |
| `model.py` | Training & Evaluation | Loads the `.npy` feature files, initializes and trains both the Logistic Regression and XGBoost classifiers, evaluates their performance using `classification_report`, and serializes the trained models as `model.pkl` and `xgb_model.json`. |

## Dependencies

You need to install the following Python packages:

* `numpy`
* `scikit-learn` (`sklearn`)
* `xgboost`
* `torch`
* `torchvision`
* `torchsummary`
* `tqdm`
* `Pillow` (`PIL`)

## Setup and Usage

### 1. Data Structure

The project expects the images to be organized into class-specific folders inside `train` and `test` directories. The hardcoded data path suggests a structure like this:
nature-dataset/ ├── train/ │ ├── class_0/ │ │ ├── image1.jpg │ │ └── ... │ ├── class_1/ │ │ ├── imageA.jpg │ │ └── ... └── test/ ├── class_0/ │ └── ... └── class_1/ └── ...
### 2. Configuration

You must update the hardcoded paths in `main.py` to point to your actual training and testing data directories:

```python
# In main.py
X_train, Y_train = load_all("C:\\Users\\Kagero\\PycharmProjects\\weather_classification\\nature-dataset\\train") # <--- UPDATE THIS PATH
# ...
X_test, Y_test = load_all("C:\\Users\\Kagero\\PycharmProjects\\weather_classification\\nature-dataset\\test")   # <--- UPDATE THIS PATH

Logistic Regression
Parameter	Value
max_iter	1000
solver	lbfgs
multi_class	multinomial
penalty	l2
C	0.1

XGBoost Classifier
Parameter	Value
n_estimators	200
max_depth	5
learning_rate	0.5
reg_lambda	0.5
reg_alpha	0.1
