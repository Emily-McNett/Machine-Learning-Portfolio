import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

class LeafData(Dataset):
    def __init__(self):

        # https://www.kaggle.com/datasets/ashishmotwani/tomato/data
        # Image dataset with 10 disease classes and 1 healthy class
        # Images are a collection of lab and wild photos
        # Only used images of healthy, early blight, and late blight

        # Transform the data
        transform = transforms.Compose([
            transforms.Resize((96, 96)), # Resize the images to 96x96
            transforms.ToTensor(), # Convert the PIL image to a Tensor
        ])

        # Gather train and test data
        self.leaf_train_data = datasets.ImageFolder(root='train', transform=transform)
        self.leaf_test_data = datasets.ImageFolder(root='test', transform=transform)

        # Gather class names from folder names
        self.leaf_class_names = self.leaf_train_data.classes
        print("Classes:", self.leaf_class_names)

        # Set the size of the data
        self.len = len(self.leaf_train_data)

    def __getitem__(self, item):
        return self.leaf_train_data[item]

    def __len__(self):
        return self.len


class LeafClassify(nn.Module):
    def __init__(self):
        super(LeafClassify, self).__init__()

        self.in_to_h1 = nn.Conv2d(3, 16, (3,3), padding = (1,1))
        self.bn1 = nn.BatchNorm2d(16) # Normalize
        self.h1_to_h2 = nn.Conv2d(16, 9, (3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(9) # Normalize
        self.h2_to_out = nn.Linear(9*24*24, 11)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x)) # (3, 96, 96)
        x = F.max_pool2d(x, (2,2)) # (16, 48, 48)
        x = F.relu(self.h1_to_h2(x)) # (9, 48, 48)
        x = F.max_pool2d(x, (2,2)) # (9, 24, 24)
        x = torch.flatten(x, 1)
        return self.h2_to_out(x)

def train_neural_network(epochs=10, batch_size=16, lr=0.001):

    # determine which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu or cpu

    leaves = LeafData()

    # Build data loaders from train and test data
    leaf_train_loader = DataLoader(leaves.leaf_train_data, batch_size=batch_size, shuffle=True)
    leaf_test_loader = DataLoader(leaves.leaf_test_data, batch_size=16, shuffle=False)

    # Create CNN
    leaf_classify = LeafClassify().to(device)
    print(f"Total parameters: {sum(param.numel() for param in leaf_classify.parameters())}") # add up entries for every tensor

    leaf_classify.train()

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(leaf_classify.parameters(), lr=lr)

    # Train in Batches
    for epoch in range(epochs):
        running_loss = 0.0 # reset running loss
        print(f"Epoch {epoch + 1} of {epochs}")

        # Training Run
        for _, data in enumerate(tqdm(leaf_train_loader)):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = leaf_classify(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        tqdm.write(f"Running loss for {epoch + 1}: {running_loss:.4f}")

        correct = 0
        total = 0
        leaf_classify.eval()

        # Testing Run
        with torch.no_grad():
            for x, y in leaf_test_loader:
                x, y = x.to(device), y.to(device)
                output = leaf_classify(x)
                preds = torch.argmax(output, dim=1)
                # all_preds.extend(preds.cpu().numpy())
                # all_labels.extend(y.cpu().numpy())
                correct += (preds == y).sum().item()
                total += y.size(0)
        tqdm.write(f"Accuracy on test set: {correct / total:.4f}")

    return leaf_classify, leaf_test_loader, leaves.leaf_class_names, device

# Gather Neural Network
leaf_classify, leaf_test_loader, leaf_class_names, device = train_neural_network(epochs=5, batch_size=16) #(.8192 on 5 .8741 on 50 .8897 (normalized) on 50) on batch size 32 - .8629 on 5 .8839 on 50 .8924 on 10

leaf_classify.eval()

# For confusion matrix
all_preds = []
all_labels = []

# Run test for accuracy and confusion matrix
with torch.no_grad():
     for x, y in leaf_test_loader:
        x, y = x.to(device), y.to(device)
        output = leaf_classify(x)
        preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Accuracy
correct = sum(p == l for p, l in zip(all_preds, all_labels))
total = len(all_labels)
print(f"\n Final Accuracy on test set: {correct / total:.4f}")

#Confusion matrix
cm = confusion_matrix(all_labels, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=leaf_class_names)
disp.plot()
plt.title("Final Confusion Matrix - Leaf CNN")
plt.show()