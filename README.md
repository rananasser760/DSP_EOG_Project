# 🚀 EOG Project
This project is a Digital Signal Processing-based Electrooculography (EOG) Signal Classifier. It provides a graphical interface to load EOG signals, preprocess them, extract features, train machine learning models, and classify eye movements (Up vs Down). with *best Accuracy = 96%* on Random Forest

## 📌 Features

- 📁 Load and preprocess EOG data (remove DC, Smoothing via moving average, bandpass filter, Outlier removal, Signal normalization, resample).
- 📊 Extract both Time-domain statistical and wavelet features.
- 🧠 Train and test classifiers: K-Nearest Neighbors, Logistic Regression, Support Vector Machine, Random Forest.
- 🧪 GridSearch hyperparameter tuning for SVM.
- 🖥️ GUI using Tkinter to load files, train models, and visualize predictions.

## 🛠️ Requirements

- Python 3.x
- numpy
- matplotlib
- scipy
- scikit-learn
- pywavelets (`pywt`)
- tkinter (standard in Python installations)

## How to run:
  - ### Install the required libraries:
      - pip install numpy scipy matplotlib scikit-learn pywt
  - ### Run the application:
      - python project.py

## Use the GUI to:

- Load training/testing .txt files

- Choose the model

- Train and predict

- View results in the output box

## 🙋‍♀️ Created by:
### Rana Nasser
### Esraa Taha
### AbdElRahman Ahmed 
