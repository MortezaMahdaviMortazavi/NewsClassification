import torch
import torch.nn as nn
import torch.optim as optim
import logging
import config
from tqdm import tqdm
from evaluate import validate

def train_batch(model, optimizer, criterion, data, targets):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Set the filename for the log file
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        data = batch['input']
        targets = batch['target']
        data, targets = data.to(device), targets.to(device)
        loss = train_batch(model, optimizer, criterion, data, targets)
        total_loss += loss
        if batch_idx % 100 == 0:
            logging.info(f'Train Epoch: {epoch}\tBatch Index: {batch_idx}\tAverage Loss: {loss:.6f}')

    average_loss = total_loss / len(train_loader)
    return average_loss

def train(model=None, train_loader=None, valid_loader=None, num_epochs=None, lr=None, device=None, log_file=config.logfile):
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Set the filename for the log file
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, log_file)
        logging.info(f'Train Epoch: {epoch}\tAverage Loss: {train_loss:.6f}')
        torch.save(model.state_dict(), f'model_checkpoint_{config.cross_val}.pth')
        validate(model, valid_loader, device)
