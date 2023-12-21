import torch

from models import RNN

if __name__ == "__main__":
    model = RNN(vocab_size=1000, embed_size=50, embed_proj_size=50, rnn_hidden_size=64,
                output_size=10, is_bidirectional=True, n_layer=2, num_classes=5, dropout_rate=0.2)
    
    # Replace MyDataset and DataLoader with your actual dataset and DataLoader
    dataset = MyDataset(data=torch.randn((1000, 10)), targets=torch.randint(0, 5, (1000,)))
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Set your device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training
    train(model, train_loader, None, num_epochs=5, lr=0.001, device=device)
