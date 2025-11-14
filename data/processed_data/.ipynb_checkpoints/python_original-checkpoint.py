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
    
    print(f"Classification: {c1_name} vs {c2_name}")
    
    # prep data
    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([np.zeros(X1.shape[0]), np.ones(X2.shape[0])])
    X_flat = X.reshape(X.shape[0], -1)
    
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # cv kfold 10 / currently 5
    kf = KFold(n_splits=5, shuffle=True, random_state=4000)
    fold_accuracies = []
    fold_aucs = []
    total_start = time.time()

    plt.figure(figsize=(7,6))


    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        print(f"\n--- Fold {fold+1}/5(10) ---")
        fold_start = time.time()

        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        # smote for unbalanced data 
        smote = SMOTE(random_state=21)
        X_train_res, y_train_res = smote.fit_resample(X_train.numpy(), y_train.numpy())

        # pyTorch tensors
        X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train_res, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        test_dataset = TensorDataset(X_test.unsqueeze(1), y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # init
        model = model_def(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9,  weight_decay=1e-4)

        # train loop
        model.train()
        for epoch in range(10):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # eval loop
        model.eval()
        y_pred = []
        y_true = []
        y_scores = [] # create y-pred, y actual, y_score (for roc-auc)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)

                probabilities = torch.softmax(outputs, dim=1)
                y_scores.extend(probabilities[:, 1].numpy())
                
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.numpy())
                y_true.extend(labels.numpy())
        
        acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
        auc_score = roc_auc_score(y_true, y_scores)
        fold_accuracies.append(acc)
        fold_aucs.append(auc_score)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, lw=1.5, label=f'Fold {fold+1} (AUC={auc_score:.3f})')

        print(f"Fold {fold+1} Accuracy: {acc:.2f}%")
        print(classification_report(y_true, y_pred, target_names=[c1_name, c2_name], digits=4))

    # ROC summary plot
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {c1_name} vs {c2_name}')
    plt.legend()
    plt.show()

    total_end = time.time()
    print(f"\nSummary of {c1_name} vs {c2_name}")
    print(f"Average 5-Fold Accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Average 5-Fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Total CV Time: {total_end - total_start:.2f} seconds\n")

