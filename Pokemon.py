import torch
import pandas as pd
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

class PokemonDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'{self.labels.iloc[idx,0]}.png')
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor()
])

dataset = PokemonDataset(csv_file=r"C:\Users\herop\OneDrive\Documents\GitHub\Pokemon_Classifier\archive (1)\pokemon.csv", 
                         root_dir=r"C:\Users\herop\OneDrive\Documents\GitHub\Pokemon_Classifier\archive (1)\images", 
                         transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(dataset.labels['Type'].unique())}

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(120 * 120 * 3, 1024)  
        self.fc2 = nn.Linear(1024, 784)
        self.fc3 = nn.Linear(784, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 18)  

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = F.log_softmax(self.fc6(x), dim=1)
        return x

model = MLP().to(device) 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 25  

for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_dataloader:
        labels = torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long).to(device)
        images = images.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch: {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader)}')

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        labels = torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long).to(device)
        images = images.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')

import random

image_files = os.listdir(dataset.root_dir)

random_image_file = random.choice(image_files)
random_image_path = os.path.join(dataset.root_dir, random_image_file)

image = Image.open(random_image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    logits = model(image)
    ps = F.softmax(logits, dim=1)
    probab = ps.cpu().numpy()[0]
    pred_label_idx = probab.argmax()
    pred_label = list(class_to_idx.keys())[list(class_to_idx.values()).index(pred_label_idx)]

img_np = image.cpu().numpy().squeeze().transpose(1, 2, 0)

plt.imshow(img_np)
plt.title(f'Predicted: {pred_label}')
plt.axis('off')
plt.show()
