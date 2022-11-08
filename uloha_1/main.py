import torch
import torchvision
from torchvision import transforms, models
from torch.nn import Sequential, Linear, ReLU, Softmax, Conv2d, MaxPool2d, Flatten, BatchNorm2d, BatchNorm1d, Dropout1d, Dropout2d, Tanh, Sigmoid, ELU, Module
from torch.nn.functional import relu, dropout2d

import numpy as np
from matplotlib import pyplot as plt

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

trainset = torchvision.datasets.ImageFolder(root='dogscats/train', transform=transform)
valset = torchvision.datasets.ImageFolder(root='dogscats/valid', transform=transform)

model = models.resnet18(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, 2)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

def one_epoch(model, optimizer, dataloader_train, dataloader_val, verbose=True, max_training_samples=None, batch_size=None):
  train_losses = []
  val_losses = []

  model.train()
  for i, batch in enumerate(dataloader_train):  
    if max_training_samples is not None and batch_size is not None and i * batch_size >= max_training_samples:
      break
    x, y = batch[0].to(device), batch[1].to(device) 
    optimizer.zero_grad()

    out = model(x)
    loss = ce_loss(out, y)
    loss.backward()
    train_losses.append(loss.item())
    optimizer.step()
    if i % 100 == 0 and verbose:
      print("Training loss at step {}: {}".format(i, loss.item()))

  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for i, batch in enumerate(dataloader_val):  
      x, y = batch[0].to(device), batch[1].to(device)  

      out = model(x)
      loss = ce_loss(out, y)
      acc = torch.sum(torch.argmax(out, dim=-1) == y)
      correct += acc.item()
      total += len(batch[1])
      val_losses.append(loss.item())

  val_acc = correct / total

  return np.mean(train_losses), np.mean(val_losses), val_acc

batch_size = 128

dataloader_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, workers=4)
dataloader_val = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ce_loss = torch.nn.CrossEntropyLoss().to(device)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epoch_val_losses_base = []
epoch_val_accs_base = []

for e in range(20):
  train_loss, val_loss, val_acc = one_epoch(model, optimizer, dataloader_train, dataloader_val, False)

  print("Val loss at epoch {}: {}".format(e, val_loss))
  print("Val acc at epoch {}: {}".format(e, val_acc))

  epoch_val_losses_base.append(val_loss)
  epoch_val_accs_base.append(val_acc)