#!/usr/bin/env python
# coding: utf-8
import zipfile
import numpy as np
import io
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
from scipy.signal import welch
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, f1_score
from tqdm import tqdm
import time
import warnings
import random
warnings.filterwarnings('ignore')

# seeds for same results each time, seed=21! 
seed = 21
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# defs 
def extract(zips):
    signals = []
    with zipfile.ZipFile(zips, 'r') as z:
        for fname in z.namelist():
            if fname.endswith(".TXT") or fname.endswith(".txt"):
                with z.open(fname) as f:
                    data = np.loadtxt(io.TextIOWrapper(f, encoding='utf-8'))
                    signals.append(data)

    return np.array(signals)
    
# noise filtering
def butterworth(lowfreq, highfreq, fs, order=4):
    nyq = 0.5 * fs
    low = lowfreq / nyq
    high = highfreq / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return b, a

def apply_filter(data, lowfreq, highfreq, fs, order=4):
    b, a = butterworth(lowfreq, highfreq, fs, order=order)
    return signal.filtfilt(b, a, data)

def applyFilter(dataset, lowfreq, highfreq, fs, order=4):
    filtered_set = []
    for sig in dataset:
        filtered_set.append(apply_filter(sig, lowfreq, highfreq, fs, order=order))
    return np.array(filtered_set)

# segmentation using sliding window
def segment(signals, window_size=256, overlap=0.5):
    step_size = int(window_size * (1 - overlap))
    all_segments = []
    for sig in signals:
        num_windows = (len(sig) - window_size) // step_size + 1
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            all_segments.append(sig[start:end])
    return np.array(all_segments)

# normalization 
def normalize_z(data):
    normalized_data = []
    for segment in data:
        mean = np.mean(segment)
        std = np.std(segment)
        if std > 0:
            normalized_segment = (segment - mean) / std
        else:
            # handles NaN makes list of 0's
            normalized_segment = segment - mean
        normalized_data.append(normalized_segment)
    return np.array(normalized_data)
    
# global init variables
fs = 173.61
low = 0.5
high = 40.0
order = 4
window_size = 256

os.makedirs('data/processed_data', exist_ok=True)

zip_paths = {
    'A': 'data/zips/set_A.zip',
    'B': 'data/zips/set_B.zip',
    'C': 'data/zips/set_C.zip',
    'D': 'data/zips/set_D.zip',
    'E': 'data/zips/set_E.zip'
}

# pipeline
for set_name, path in zip_paths.items():
    print(f"Processing Set {set_name}...")

    # extract
    eeg_raw = extract(path)
    print(f"Extracted {len(eeg_raw)} signals.")

    # butterworth in loop
    eeg_filtered = applyFilter(eeg_raw, low, high, fs, order)
    print("Applied Butterworth filter.")

    # segment
    eeg_segmented = segment(eeg_filtered, window_size=window_size)
    print(f"Segmented into {len(eeg_segmented)} windows.")

    # norm
    eeg_normalized = normalize_z(eeg_segmented)
    print("Applied Z-score normalization")
    
    eeg_processed = eeg_normalized
    # save for classification
    
    save_path = f'data/processed_data/processed_{set_name}.npy'
    np.save(save_path, eeg_processed)
    print("Done.\n")

print("Pre-processing complete!")
print("Saved in 'data/processed_data'\n")

class cnn_model(nn.Module):
    def __init__(self, num_classes=2, input_length=256):
        super(cnn_model, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # calculate flatten side
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            flatten_dim = x.numel()

        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_evaluate(X1, X2, c1_name, c2_name, model_def, input_len=256):
    lr_grid = [0.1, 0.01, 0.001]
    winning_auc = 0
    winning_lr = None
    results = {}
    
    os.makedirs("plots/roc", exist_ok=True)
    
    total_start = time.time()

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([np.zeros(X1.shape[0]), np.ones(X2.shape[0])])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    for lr in lr_grid:
        print(f"\nTraining Learning Rate: {lr}")
        
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        
        fold_aucs, fold_accuracies, fold_sensitivities, fold_specificities, fold_f1s = [], [], [], [], []
        
        # declare roc auc plot size
        plt.figure(figsize=(8, 6))

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

            smote = SMOTE(random_state=seed)
            X_train_res, y_train_res = smote.fit_resample(X_train.numpy(), y_train.numpy())

            X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).unsqueeze(1)
            y_train_tensor = torch.tensor(y_train_res, dtype=torch.long)
            X_test_tensor = X_test.unsqueeze(1)
            y_test_tensor = y_test

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
            
            model = model_def(num_classes=2, input_length=input_len)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

            # trainin 
            model.train()
            for epoch in range(10):
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # eval
            model.eval()
            y_scores, y_true, y_pred = [], [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    # Store prob of class 1 for ROC
                    y_scores.extend(probs[:, 1].cpu().numpy())
                    _, preds = torch.max(outputs, 1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(labels.cpu().numpy())

            acc = np.mean(np.array(y_pred) == np.array(y_true)) * 100
            auc_score = roc_auc_score(y_true, y_scores)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            fold_accuracies.append(acc)
            fold_aucs.append(auc_score)
            fold_sensitivities.append(sensitivity)
            fold_specificities.append(specificity)
            fold_f1s.append(f1)
            
            # each fold roc
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.9, label=f'Fold {fold+1} (AUC = {auc_score:.3f})')

            print(f"Fold {fold+1}/10, Acc: {acc:.2f}%, AUC: {auc_score:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}, F1: {f1:.4f}")
            
        mean_auc = np.mean(fold_aucs)
        mean_acc = np.mean(fold_accuracies)
        mean_sensitivity = np.mean(fold_sensitivities)
        mean_specificity = np.mean(fold_specificities)
        mean_f1 = np.mean(fold_f1s)
        
        # 50 line and plot by 0.2 and customizations
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (10-Fold CV): {c1_name} vs {c2_name}')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True, alpha=0.3)
        
        # save plot
        clean_name = f"{c1_name}_vs_{c2_name}_lr{lr}".replace("+", "").replace(" ", "_").replace("(", "").replace(")", "")
        save_path = f"plots/roc/roc_folds_{clean_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"ROC Plot with 10 folds saved to {save_path}")
        plt.show()

        print(f"Averages for LR {lr}")
        print(f"Mean Acc: {mean_acc:.2f}%, Mean AUC: {mean_auc:.4f}, Mean Sens: {mean_sensitivity:.4f}, Mean Spec: {mean_specificity:.4f}, Mean F1: {mean_f1:.4f}")
        
        results[lr] = {'mean_acc': np.mean(fold_accuracies), 'std_acc': np.std(fold_accuracies)}

        if mean_auc > winning_auc:
            winning_auc = mean_auc
            winning_lr = lr

    total_end = time.time()

    print(f"\nBest LR = {winning_lr}, AUC = {winning_auc:.4f}")
    print(f"Total CV Time: {total_end - total_start:.2f}s")

    return winning_lr

def conv1d_filters_to_numpy(conv_layer):
    return conv_layer.weight.detach().cpu().numpy()

def compute_dft_magnitude(filters, nfft=1024):
    mags = np.abs(np.fft.rfft(filters, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0)
    return freqs, mags

# dft spectral plot:
def plot_dft(freqs, mags, fs, title="", max_filters=8, save_path=None):
    plt.figure(figsize=(10, 6))
    scaled_freqs = freqs * (fs / 2) 
    for i in range(min(mags.shape[0], max_filters)):
        plt.plot(scaled_freqs, mags[i].mean(axis=0), label=f'Filter {i+1}')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"{title} Filter DFT Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

# lipschitz defined as 
def estimate_lipschitz(model):
    norms = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            w = layer.weight.detach().cpu().numpy().reshape(layer.weight.shape[0], -1)
            s = np.linalg.svd(w, compute_uv=False)
            norms.append(np.max(s))
    L_hat = np.prod(norms)
    return L_hat, norms

print('ready to train...')


# #### The following equation used in **`estimate_lipschitz(model)`**
# $$
# \sum_{N \in V} \| f_N - \tilde{f}_N \|_2^2 \le L \| f - \tilde{f} \|_2^2, \\
# \text{where } V = \bigcup_{m=1}^{M} V_m \text{ is the collection of all output-generating nodes}
# $$

# load fully Preprocessed npy
processed_A = np.load("data/processed_data/processed_A.npy")
processed_B = np.load("data/processed_data/processed_B.npy")
processed_C = np.load("data/processed_data/processed_C.npy")
processed_D = np.load("data/processed_data/processed_D.npy")
processed_E = np.load("data/processed_data/processed_E.npy")


# # A vs E Train & Interpretability Visuals
# A vs E 
best_lr = train_and_evaluate(processed_A, processed_E, "A", "E", cnn_model, input_len=256)

model_A_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_A_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_A, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_A.shape[0]), np.ones(processed_E.shape[0])])

# train and eval with best LR
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_A_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_A_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal model trained — running interpretability...")

# dft learned features
filters = conv1d_filters_to_numpy(model_A_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (A vs E)", save_path="plots/dft_A.png")

# lipschitz continuity 
L_hat, norms = estimate_lipschitz(model_A_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_A_E.state_dict(), "models/final_model_A_vs_E.pth")


# # B vs E Train & Interpretability Visuals
# B vs E 
best_lr = train_and_evaluate(processed_B, processed_E, "B", "E", cnn_model, input_len=256)

model_B_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_B_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_B, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_B.shape[0]), np.ones(processed_E.shape[0])])

# train and eval with best LR
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_B_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_B_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal model trained — running interpretability...")

# dft learned features
filters = conv1d_filters_to_numpy(model_B_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (B vs E)", save_path="plots/dft_B.png")

# lipschitz continuity 
L_hat, norms = estimate_lipschitz(model_B_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_B_E.state_dict(), "models/final_model_B_vs_E.pth")


# # C vs E Train & Interpretability Visuals
# C vs E 
best_lr = train_and_evaluate(processed_C, processed_E, "C", "E", cnn_model, input_len=256)

model_C_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_C_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_C, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_C.shape[0]), np.ones(processed_E.shape[0])])

# train and eval with best LR
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_C_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_C_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal model trained — running interpretability...")

# dft learned features
filters = conv1d_filters_to_numpy(model_C_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (C vs E)", save_path="plots/dft_C.png")

# lipschitz continuity 
L_hat, norms = estimate_lipschitz(model_C_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_C_E.state_dict(), "models/final_model_C_vs_E.pth")


# # D vs E Train & Interpretability Visuals
# D vs E 
best_lr = train_and_evaluate(processed_D, processed_E, "D", "E", cnn_model, input_len=256)

model_D_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_D_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_D, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_D.shape[0]), np.ones(processed_E.shape[0])])

# train and eval with best LR
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_D_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_D_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal model trained — running interpretability...")

# dft learned features
filters = conv1d_filters_to_numpy(model_D_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (D vs E)", save_path="plots/dft_D.png")

# lipschitz continuity 
L_hat, norms = estimate_lipschitz(model_D_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_D_E.state_dict(), "models/final_model_D_vs_E.pth")


# # (A,B) vs E Train & Interpretability Visuals

processed_AB = np.concatenate([processed_A, processed_B], axis=0)

best_lr = train_and_evaluate(processed_AB, processed_E, "A+B", "E", cnn_model, input_len=256)

model_AB_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_AB_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_AB, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_AB.shape[0]), np.ones(processed_E.shape[0])])

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_AB_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_AB_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal (A+B) vs E model trained — running interpretability...")

filters = conv1d_filters_to_numpy(model_AB_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (A+B vs E)", save_path="plots/dft_AB.png")

L_hat, norms = estimate_lipschitz(model_AB_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_AB_E.state_dict(), "models/final_model_AB_vs_E.pth")


# # (A,B,C) vs E Train & Interpretability Visuals

processed_ABC = np.concatenate([processed_A, processed_B, processed_C], axis=0)

best_lr = train_and_evaluate(processed_ABC, processed_E, "A+B+C", "E", cnn_model, input_len=256)

model_ABC_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ABC_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_ABC, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_ABC.shape[0]), np.ones(processed_E.shape[0])])

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_ABC_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_ABC_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal (A+B+C) vs E model trained — running interpretability...")

filters = conv1d_filters_to_numpy(model_ABC_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (A+B+C vs E)", save_path="plots/dft_ABC.png")

L_hat, norms = estimate_lipschitz(model_ABC_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_ABC_E.state_dict(), "models/final_model_ABC_vs_E.pth")


# # (A,B,C,D) vs E Train & Interpretability Visuals

processed_ABCD = np.concatenate([processed_A, processed_B, processed_C, processed_D], axis=0)

best_lr = train_and_evaluate(processed_ABCD, processed_E, "A+B+C+D", "E", cnn_model, input_len=256)

model_ABCD_E = cnn_model(num_classes=2, input_length=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ABCD_E.parameters(), lr=best_lr, momentum=0.9, weight_decay=1e-4)

X = np.concatenate([processed_ABCD, processed_E], axis=0)
y = np.concatenate([np.zeros(processed_ABCD.shape[0]), np.ones(processed_E.shape[0])])

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(10):
    model_ABCD_E.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_ABCD_E(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("\nFinal (A+B+C+D) vs E model trained — running interpretability...")

filters = conv1d_filters_to_numpy(model_ABCD_E.conv1)
freqs, mags = compute_dft_magnitude(filters, nfft=1024)
plot_dft(freqs, mags, fs=173.61, title="Conv1 Layer (A+B+C+D vs E)", save_path="plots/dft_ABCD.png")

L_hat, norms = estimate_lipschitz(model_ABCD_E)
print("Estimated Lipschitz bound:", L_hat)
print("Per-layer spectral norms:", norms)

# save model
torch.save(model_ABCD_E.state_dict(), "models/final_model_ABCD_vs_E.pth")

