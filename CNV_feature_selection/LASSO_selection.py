import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

data_file = "path/to/json"
with open(data_file, 'r') as f:
    data_list = json.load(f)

def is_invalid_subtype(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and (x.strip() == "" or x.strip().lower() == "nan"):
        return True
    return False

filtered_data = [d for d in data_list if not is_invalid_subtype(d.get("subtype"))]

train_data = [d for d in filtered_data if d.get("split") == "train"]
val_data   = [d for d in filtered_data if d.get("split") == "val"]
test_data  = [d for d in filtered_data if d.get("split") == "test"]

def convert_label(label):
    return str(label)

class TCGA_Dataset(Dataset):
    def __init__(self, data, label_to_index):
        self.data = data
        self.label_to_index = label_to_index
        self.input_dim = len(data[0]["CNV"])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = torch.tensor(sample["CNV"], dtype=torch.float32)
        y = self.label_to_index[convert_label(sample["subtype"])]
        return x, y

unique_labels = sorted(list({convert_label(d["subtype"]) for d in train_data}))
num_classes = len(unique_labels)
print("Name of unique label:", unique_labels)
print("Num of class:", num_classes)

label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

batch_size = 32
train_dataset = TCGA_Dataset(train_data, label_to_index)
val_dataset   = TCGA_Dataset(val_data, label_to_index)
test_dataset  = TCGA_Dataset(test_data, label_to_index)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = train_dataset.input_dim
print("CNV length:", input_dim)

class FeatureSelector(nn.Module):
    def __init__(self, num_features, l1_reg=1e-3):
        super(FeatureSelector, self).__init__()
        self.mask = nn.Parameter(torch.ones(num_features))
        self.l1_reg = l1_reg

    def forward(self, x):
        return x * self.mask

class LassoClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, l1_reg=1e-4):
        super(LassoClassifier, self).__init__()
        self.feature_selector = FeatureSelector(input_dim, l1_reg)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.feature_selector(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

l1_reg = 1e-3
model = LassoClassifier(input_dim, num_classes, l1_reg).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
best_test_acc = 0.0

def evaluate(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        l1_penalty = l1_reg * torch.sum(torch.abs(model.feature_selector.mask))
        loss = loss + l1_penalty
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
    
    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    val_loss, val_acc = evaluate(val_loader, model, criterion)
    test_loss, test_acc = evaluate(test_loader, model, criterion)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "lasso_model.pth")
        
        mask_np = model.feature_selector.mask.detach().cpu().numpy()
        selected_indices = np.argsort(-np.abs(mask_np))[:173]

        np.save("lasso_weights.npy", mask_np)
        np.save("selected_indices.npy", selected_indices)

        print(f"Best Acc: {best_test_acc:.4f}, saved!")
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

model.cpu()
mask_np = model.feature_selector.mask.detach().numpy()  # shape: (input_dim,)
top_k = 173
selected_indices = np.argsort(-np.abs(mask_np))[:top_k]
print("The top 173 selected feature indices:", selected_indices)

np.save("lasso_weights.npy", mask_np)
np.save("selected_indices.npy", selected_indices)
print("LASSO weights and selected indices have been saved.")

torch.save(model.state_dict(), "lasso_model.pth")
print("The model state_dict has been saved as 'lasso_model.pth'.")
