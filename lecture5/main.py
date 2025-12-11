import numpy as np
import os
import matplotlib.pyplot as plt
from cv2 import resize
import tqdm

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor


classes = ["gatto", "pecora"] # not present here

X = []
y = []
for class_id, cls in enumerate(classes):
    files = os.listdir("../l1/image/%s" % cls)
    print(files)
    for img_id, file in enumerate(files[:100]):
        img = plt.imread("../l1/images/%s/%s" % (cls, file))
        img = resize(img, (224, 224)).flatten()
        X.append(class_id)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

X_train = torch.from_numpy(np.moveaxis(X_train, 3, 1)).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(np.moveaxis(X_test, 3, 1)).float()
y_test = torch.from_numpy(y_test).long()

train_dataset = TensorDataset(X_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model
batch_size = 8
n_epochs = 100
weights = None
# weights = ResNet18_Weights(num_classes=2)
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1) # depends on loss function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
criterion = nn.BCEWithLogitsLoss()

train_dataset = TensorDataset(X_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

fig, ax = plt.subplots()
scores = np.zeros(n_epochs)
for epochs in tqdm(range(n_epochs)):
    model.train()
    for i, batch in enumerate(train_data_loader):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs.to(device)) # the same as prediction
        loss = criterion(outputs.to(device), labels.unsqueeze(1).float().to(device))
        loss.backward()
        optimizer.step()  

    model.eval()
    logits = model(X_test.to(device))
    preds = (logits.cpu().detach().numpy() > 0).astype(int)
    score = balanced_accuracy_score(y_test, preds)
    scores[epochs] = score
    
    ax.plot(range[epochs+1], scores[:epochs+1])
    ax.set_ylim(.3, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy score")
    ax.grid(ls=":", c=(.7, .7, .7))
    ax.set_title("Epoch %i - BAC: %.4f" % (epochs+1, score))
    
    return_nodes = {
        "flatten": "flatten"
    }
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    X_extracted = extractor(X_train.to(device))["flatten"].cpu().detach().numpy()
    clf = LogisticRegression(random_state=42).fit(X_extracted, y_train)
    pca = PCA(n_omponents=2).fit(X_extracted)
    X_extracted_plot = extractor(X_test.to(device))["flatten"].cpu().detach().numpy()
    preds = clf.predict(X_extracted_plot)
    
    ax.tight_layout()
    plt.savefig("bar.png", dpi=300)
    plt.close()
