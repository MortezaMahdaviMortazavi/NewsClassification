import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, random_split
from evaluate import validate

def train_batch(model, optimizer, criterion, data, targets):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer):
    logging.basicConfig(level=logging.INFO)
    model.train()
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        loss = train_batch(model, optimizer, criterion, data, targets)
        total_loss += loss
        if batch_idx % 100 == 0:
            logging.info(f'Train Epoch: {epoch}\tAverage Loss: {train_loss:.6f}')

    average_loss = total_loss / len(train_loader)
    return average_loss

def train(model, train_loader, valid_loader, num_epochs, lr, device):
    logging.basicConfig(level=logging.INFO)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        logging.info(f'Train Epoch: {epoch}\tAverage Loss: {train_loss:.6f}')
        validate(model,eval_loader,device)


