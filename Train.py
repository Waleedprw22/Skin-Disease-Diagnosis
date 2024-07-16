import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, random_split
from Environment import skin_dataset


# Don't have gpus so this will by default be 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
train_size = int(0.8 * skin_dataset.__len__())
val_size = skin_dataset.__len__() - train_size
train_dataset, val_dataset = random_split(skin_dataset, [train_size, val_size])

# Set up the Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True) # Drop last batch if less than batch size.


# Setup model for training

model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 7) # This is to account for the 7 different classes of skin diseases.
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 10

for epoch in range(num_epochs):
    
    # Training Process
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        prediction = model(imgs)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        

    # Validation Process
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            
            outputs = model(imgs)

            loss = criterion(outputs, labels)  # Corrected this line
            val_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


# Save the trained model
torch.save(model.state_dict(), 'efficientnet_skin_lesion_classifier.pth')
print('saved')