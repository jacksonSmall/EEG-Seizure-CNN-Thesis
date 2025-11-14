# defs
class cnn_model(nn.Module):
    def __init__(self, num_classes=2):
        super(cnn_model, self).__init__() # init model
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1) # convolutional layer 1
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1) # convolutional layer 2
        self.pool = nn.MaxPool1d(2) # pooling

        self.fc1 = nn.Linear(32 * 64, 128) # fully connected 1
        self.fc2 = nn.Linear(128, num_classes) #fully connceted 2
        self.relu = nn.ReLU() # relu activation 
        
    # forward pass
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_evaluate(X1, X2, c1_name, c2_name, model_def):
    
    print(f"\n{'='*60}\nSGD Hyperparameter Tuning: {c1_name} vs {c2_name}\n{'='*60}")

    # --- 1. Define Hyperparameter Grid for SGD ---
    learning_rates_to_test = [0.1, 0.01, 0.001]
    best_avg_auc = 0
    best_lr = None
    results = {}

    # --- 2. Prepare Data (Correctly, without flattening) ---
    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([np.zeros(X1.shape[0]), np.ones(X2.shape[0])])
    
    # Create tensors from the original data, preserving the segment structure
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # --- 3. Loop Over Hyperparameters ---
    for lr in learning_rates_to_test:
        print(f"\n----- Testing SGD Learning Rate: {lr} -----")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=4000)
        fold_aucs = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

            # Apply SMOTE only to the training data of this fold
            smote = SMOTE(random_state=21)
            X_train_res, y_train_res = smote.fit_resample(X_train.numpy(), y_train.numpy())

            X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).unsqueeze(1)
            y_train_tensor = torch.tensor(y_train_res, dtype=torch.long)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test.unsqueeze(1), y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # --- Model Initialization & Training ---
            model = model_def(num_classes=2)
            criterion = nn.CrossEntropyLoss()
            # Use SGD optimizer with the current learning rate from the loop
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            
            model.train()
            for epoch in range(10):
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # --- Model Evaluation ---
            model.eval()
            y_scores, y_true = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    y_scores.extend(probabilities[:, 1].cpu().numpy())
                    y_true.extend(labels.cpu().numpy())
            
            fold_aucs.append(roc_auc_score(y_true, y_scores))
        
        # --- Aggregate and Store Results for the Current LR ---
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        results[lr] = {'mean_auc': mean_auc, 'std_auc': std_auc}
        print(f"  > Average 5-Fold AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        if mean_auc > best_avg_auc:
            best_avg_auc = mean_auc
            best_lr = lr

    # --- 4. Final Summary ---
    print(f"\n--- Hyperparameter Tuning Summary for {c1_name} vs {c2_name} ---")
    print(f"  🏆 Best SGD Learning Rate Found: {best_lr}")
    print(f"  🚀 Best 5-Fold Average AUC: {results[best_lr]['mean_auc']:.4f} ± {results[best_lr]['std_auc']:.4f}")