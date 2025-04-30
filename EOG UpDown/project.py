from matplotlib import pyplot as plt
import numpy as np
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import butter, filtfilt,resample
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import skew, kurtosis

def read_file(file_path):
    try:
        array_of_arrays = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    values = [float(value) for value in line.split() if value]
                    array_of_arrays.append(values)
        return array_of_arrays
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def remove_dc_time(y_values):
    sum_signal = 0
    for value in y_values:
        sum_signal += value
    mean_value = sum_signal / len(y_values)
    dc_removed_signal = [value - mean_value for value in y_values]

    #print(f"DC component (Time Domain) removed successfully. Mean value: {mean_value}")
    return dc_removed_signal

# Normalize signal
def normalize_signal(amplitude, norm_type):
    if norm_type == 0:
        return (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
    elif norm_type == 1:
        return (2 * (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))) - 1

def apply_convolution(signal, kernel):
    """Apply convolution to the signal using the provided kernel."""
    return np.convolve(signal, kernel, mode='same')

def preprocess_signal(signal, fs=176, target_fs=50):
    # Mean removal
    removed_signal = remove_dc_time(signal)
    
    # Apply moving average filter for noise reduction
    window_size = 5
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(removed_signal, kernel, mode='same')
    
    # Bandpass Filter with optimized parameters
    filtered_signal = bandpass_filter(smoothed_signal, 0.5, 20, fs, order=5)
    
    # Remove outliers
    z_scores = np.abs((filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal))
    filtered_signal[z_scores > 3] = np.mean(filtered_signal)
    
    # Normalization
    normalized_signal = normalize_signal(filtered_signal, norm_type=1)  # Changed to [-1,1] normalization
    
    # Resampling
    resampled_signal = resample(normalized_signal, int(len(normalized_signal) * target_fs / fs))
    
    return resampled_signal




def extract_additional_features(signal):
    """Extract additional statistical and time-domain features"""
    features = []
    
    # Statistical features
    features.extend([
        np.mean(signal),
        np.std(signal),
        skew(signal),
        kurtosis(signal),
        np.max(signal),
        np.min(signal),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75)
    ])
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features.append(len(zero_crossings))
    
    # Energy
    features.append(np.sum(np.square(signal)))
    
    return np.array(features)

def extract_wavelet_features(signal, wavelet_name='db4', max_level=4):
    features = []
    
    # Decompose signal using wavelet transform
    coeffs = pywt.wavedec(signal, wavelet_name, level=max_level)
    
    # Extract features from each decomposition level
    for i, coef in enumerate(coeffs):
        features.extend([
            np.mean(coef),
            np.std(coef),
            np.max(coef),
            np.min(coef),
            skew(coef),
            kurtosis(coef),
            np.sum(np.square(coef))  # Energy
        ])
    
    return np.array(features)

def tune_svm(features, labels):
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }
            svm = SVC()
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(features, labels)
            print(f"Best SVM Parameters: {grid_search.best_params_}")
            return grid_search.best_score_
class EOGClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EOG Signal Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="black") 

        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.logreg = LogisticRegression(C=0.1)
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=35, min_samples_split=2, min_samples_leaf=5, max_features="sqrt", bootstrap=True)  

        self.svm = SVC(C=1, kernel='linear', gamma='scale' , probability=True) 

        self.model = None
        self.is_trained = False
        self.create_gui()

    def create_gui(self):
        # File Loading Frame
        load_frame = tk.LabelFrame(
            self.root, text="Data Loading", padx=10, pady=5,
            bg="black", fg="white", relief="groove"
        )
        load_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            load_frame, text="Load Up Training", command=lambda: self.load_file("up_train"),
            bg="#2980b9", fg="white"
        ).pack(side="left", padx=5)

        tk.Button(
            load_frame, text="Load Down Training", command=lambda: self.load_file("down_train"),
            bg="#2980b9", fg="white"
        ).pack(side="left", padx=5)

        tk.Button(
            load_frame, text="Load Test Data", command=lambda: self.load_file("test"),
            bg="#2980b9", fg="white"
        ).pack(side="left", padx=5)

        # Model Control Frame
        control_frame = tk.LabelFrame(
            self.root, text="Model Controls", padx=10, pady=5,
            bg="black", fg="white", relief="groove"
        )
        control_frame.pack(fill="x", padx=10, pady=5)

        self.model_choice = tk.StringVar(value="KNN")
        tk.Radiobutton(control_frame, text="KNN", variable=self.model_choice, value="KNN", bg="black", fg="grey").pack(side="left", padx=5)
        tk.Radiobutton(control_frame, text="Logistic Regression", variable=self.model_choice, value="Logistic Regression", bg="black", fg="grey").pack(side="left", padx=5)
        tk.Radiobutton(control_frame, text="SVM", variable=self.model_choice, value="SVM", bg="black", fg="grey").pack(side="left", padx=5)
        tk.Radiobutton(control_frame, text="Random Forest", variable=self.model_choice, value="Random Forest", bg="black", fg="grey").pack(side="left", padx=5)


        self.k_var = tk.IntVar(value=5)
        self.k_scale = tk.Scale(
            control_frame, from_=1, to=20, orient="horizontal",
            variable=self.k_var, command=self.update_k, bg="black", fg="white",
            troughcolor="#2980b9"
        )
        self.k_scale.pack(side="left", padx=5)

        tk.Button(
            control_frame, text="Train Model", command=self.train_model,
            bg="#2980b9", fg="white"
        ).pack(side="left", padx=5)

        tk.Button(
            control_frame, text="Predict", command=self.predict,
            bg="#2980b9", fg="white"
        ).pack(side="left", padx=5)

        tk.Button(
            control_frame, text="Clear Results", command=self.clear_results,
            bg="#2980b9", fg="white"
        ).pack(side="left", padx=5)

        # Results Frame
        results_frame = tk.LabelFrame(
            self.root, text="Results", padx=10, pady=5,
            bg="black", fg="white", relief="groove"
        )
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Add scrollbar to results
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side="right", fill="y")

        self.results_text = tk.Text(
            results_frame, height=20, yscrollcommand=scrollbar.set,
            bg="black", fg="white", insertbackground="white"
        )
        self.results_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.results_text.yview)

    def update_k(self, value):
        self.knn.n_neighbors = int(value)
        if self.is_trained:
            self.is_trained = False
            self.results_text.insert(tk.END, "Please retrain the model with new K value\n")

    def load_file(self, file_type):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return

        try:
            data = read_file(file_path)
            if file_type == "up_train":
                self.Up_Train = data
                self.results_text.insert(tk.END, f"Loaded Up Training: {len(data)} samples\n")
            elif file_type == "down_train":
                self.Down_Train = data
                self.results_text.insert(tk.END, f"Loaded Down Training: {len(data)} samples\n")
            elif file_type == "test":
                self.test_data = data
                self.results_text.insert(tk.END, f"Loaded Test Data: {len(data)} samples\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def train_model(self):
        if not hasattr(self, 'Up_Train') or not hasattr(self, 'Down_Train'):
            messagebox.showerror("Error", "Please load both Up and Down training data first")
            return

        try:
            up_train_processed = [preprocess_signal(signal) for signal in self.Up_Train]
            down_train_processed = [preprocess_signal(signal) for signal in self.Down_Train]
            self.train_signals = up_train_processed + down_train_processed
            self.train_labels = [1] * len(up_train_processed) + [0] * len(down_train_processed)

            self.train_features = np.array([extract_wavelet_features(signal) for signal in self.train_signals])
            #self.train_features = np.array([extract_additional_features(signal) for signal in self.train_signals])

            if self.model_choice.get() == "KNN":
                self.model = self.knn
            elif self.model_choice.get() == "Logistic Regression":
                self.model = self.logreg
            elif self.model_choice.get() == "Random Forest":
                self.model = self.rf  # Set Random Forest model
            elif self.model_choice.get() == "SVM":
                self.model = self.svm  # Set SVM model with parameters

            self.model.fit(self.train_features, self.train_labels)
            self.is_trained = True

            cv_scores = cross_val_score(self.model, self.train_features, self.train_labels, cv=5)
            train_accuracy = self.model.score(self.train_features, self.train_labels)

            self.results_text.insert(tk.END, f"\nTraining completed!\n")
            #self.results_text.insert(tk.END, f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})\n")
            self.results_text.insert(tk.END, f" Accuracy: {train_accuracy:.3f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def predict(self):
        if not self.is_trained:
            messagebox.showerror("Error", "Please train the model first")
            return

        if not hasattr(self, 'test_data'):
            messagebox.showerror("Error", "Please load test data first")
            return

        try:
            test_processed = [preprocess_signal(signal) for signal in self.test_data]
            test_features = np.array([extract_wavelet_features(signal) for signal in test_processed])

            predictions = self.model.predict(test_features)
            probabilities = self.model.predict_proba(test_features)

            self.results_text.insert(tk.END, "\nPrediction Results:\n")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                direction = "Up" if pred == 1 else "Down"
                confidence = prob[1] if pred == 1 else prob[0]
                self.results_text.insert(tk.END, f"Signal {i + 1}: {direction} (Confidence: {confidence:.2f})\n")

                #self.plot_signal(test_processed[i], i + 1)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")


    def plot_signal(self, signal, signal_number):
        plt.figure(figsize=(10, 6))
        plt.plot(signal, label=f'Signal {signal_number}')
        plt.title(f'Signal {signal_number}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
def main():
    root = tk.Tk()
    app = EOGClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

