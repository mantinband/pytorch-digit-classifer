import torch
import sys
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

MODEL_FILE_NAME = "digit-model.pt"


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        :param x: batch of input data in the shape: (-1, input_size)
        :return: model prediction
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)


def fit(model, trainset, image_size, epochs=3, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in tqdm(trainset, file=sys.stdout):
            model.zero_grad()
            model_outputs = model(X.view(-1, image_size[0] * image_size[1]))
            loss = F.nll_loss(model_outputs, y)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss})")


def test_accuracy(model, testset, image_size):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in tqdm(testset, file=sys.stdout):
            model_outputs = model(X.view(-1, image_size[0] * image_size[1]))
            _, model_predictions = torch.max(model_outputs, dim=1)
            correct += torch.sum(model_predictions == y).item()
            total += len(y)

    return correct / total


def main():
    # download datasets as tensors
    train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    trainset = DataLoader(train, batch_size=10, shuffle=True)
    testset = DataLoader(test, batch_size=10, shuffle=False)

    image_width = train[0][0].shape[1]
    image_height = train[0][0].shape[2]

    try:
        model = torch.load("digit-model.pt")
        print(f"loaded model '{MODEL_FILE_NAME}'!")
    except:
        print("training model")
        model = Net(image_width * image_height, 10)
        fit(model, trainset, (image_width, image_height))

        print(f"model trained! saving as '{MODEL_FILE_NAME}'..")
        torch.save(model, MODEL_FILE_NAME)
        print("saved!")

    for dataset_type, dataset in {"test data set": testset, "train data set": trainset}.items():
        print(f"testing {dataset_type} accuracy")
        model_accuracy = test_accuracy(model, dataset, (image_width, image_height))
        print(f"model accuracy is: {model_accuracy * 100}\n")


if __name__ == '__main__':
    main()
