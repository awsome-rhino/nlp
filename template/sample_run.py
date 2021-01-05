import sys, os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# 입력받을 인자값 등록
parser.add_argument('--path', type=str, default=".")
parser.add_argument('--data', type=str, default="example")
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--wdecay', type=float, default=0.002)
parser.add_argument('--lr', type=float, default=1e-4)

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()
device = torch.device("cpu")


"""
# Classifier Dataset
class ClassifierDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return {
            'inputs' : torch.tensor(self.inputs[i]),
            'labels' : torch.tensor(self.labels[i])
        }


# Classifier Neural Network Model
class ClassifierDefault(nn.Module):
    def __init__(self, x_dim, total_labels, y_dim=128, drop_rate=0.2):
        super().__init__()

        self.FEATURE = nn.Linear(x_dim,256)
        # data embedding
        self.a_layer = nn.Sequential(
            self.FEATURE,
            nn.Dropout(p=drop_rate),
            nn.Linear(256, 128),
            nn.Dropout(p=drop_rate),
            nn.Linear(128,20)
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.INTENT = nn.Softmax(dim=-1)
        self.softmax = self.INTENT

    def forward(self, x, y):

        x_a = self.a_layer(x)

        # Simple NN Model
        loss = self.ce_loss(x_a, y)
        output = torch.argmax(self.softmax(x_a), dim=-1)

        return [loss, output]
"""

# CLASSIFIER
def run(path):
    # 학습 데이터 불러오기
    # TODO: Need to Implement
    data = ClassifierDataset(path)
    data_loader = DataLoader(data, batch_size=args.batch, shuffle=True, pin_memory=True)

    # 모델 선언
    # TODO: Need to implement
    model = Model()

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    # Training Itervatively
    for epoch in tqdm(range(args.epochs), desc='Processing Classifier'):
        gt, pred = [], []
        avg_loss, nbatch = 0., 0.
        for idx, batch in enumerate(data_loader):
            # Forward-Propagation
            opt.zero_grad()
            inputs = batch['inputs'].float().to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs, labels)

            outputs = [outputs[0].cpu(), outputs[1].cpu()]

            loss = outputs[0]

            # Record
            avg_loss += loss.data.numpy()
            nbatch = idx
            gt.extend(labels)
            pred.extend(outputs[1])
            
            # Backward Propagation
            loss.backward()
            opt.step()

        acc = np.mean(np.equal(gt, pred))
        print("Acc. ", acc, ' Loss. ', avg_loss/nbatch)

    torch.save(model, 'model.pt')


if __name__ == "__main__":
    run(args.path+'/'+args.data)
