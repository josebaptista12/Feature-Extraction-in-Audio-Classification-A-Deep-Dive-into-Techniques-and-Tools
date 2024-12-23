# Audio Classification Project

## Overview
This repository contains code for audio feature extraction and classification using various machine learning models. The code leverages audio datasets to build classifiers capable of identifying audio samples.

## Project Structure
- **extract_features**: Extracts features from audio files using Librosa.
- **MyGridSearchCV**: Implements a custom GridSearchCV for hyperparameter tuning and validation.

## Dependencies
Ensure the following packages are installed:
- Python 3.6+
- numpy
- pandas
- matplotlib
- seaborn
- librosa
- soundfile
- scipy
- IPython
- sklearn
- catboost
- tqdm

You can install the dependencies with:
```bash
pip install numpy pandas matplotlib seaborn librosa soundfile scipy IPython scikit-learn catboost tqdm
```

## Setup and Usage
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Prepare the dataset:
- Download the ESC-50 dataset (or any other audio dataset).
- Place the audio files in the directory `./ESC-50/input/audio/`.

3. Feature extraction:
```bash
python feature_extraction.py
```
Features will be extracted and saved as CSV files.

4. Model training and evaluation:
```bash
python model_training.py
```
This will train models, perform hyperparameter tuning, and evaluate their performance.

## Feature Extraction
Features extracted include:
- Zero Crossing Rate
- Chroma STFT
- Spectral Centroid
- MFCCs
- RMS Energy
- Spectral Bandwidth
- Spectral Contrast
- Spectral Rolloff

## Classification Models
The following classifiers are implemented:
- Support Vector Classifier (SVC)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (LinearSVC)

## Replicating Results
To replicate results:
1. Set the random seed to ensure reproducibility:
```python
seed_everything(42)
```
2. Run feature extraction and model training scripts sequentially.
3. Evaluate model performance using the provided validation set.

## Results
Confusion matrices and classification reports are generated for model evaluation. These can be plotted using:
```python
my_grid_search.plot_confusion_matrix(X_validate, y_validate)
```

## Notes
- Ensure the audio files are properly formatted (wav files).
- Adjust model parameters for better performance.
- This code is optimized for ESC-50 but can be adapted for other audio datasets.

