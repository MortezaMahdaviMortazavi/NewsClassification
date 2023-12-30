import torch
import pandas as pd
import logging
import config
from models import RNN
from preprocess import Vocabulary
from dataloader import get_dataloader
from train import train

# Configure logging
logging.basicConfig(filename=config.logfile, level=logging.INFO)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Found {torch.cuda.device_count()} GPUs. Using GPU for computations.")
    else:
        device = torch.device('cpu')
        logging.info("GPU is not available. Computations will be on CPU.")

    train_df = pd.read_csv(config.train_path, encoding='utf-8')
    train_texts = train_df['Text'].to_list()
    train_targets = train_df['Category'].tolist()
    num_classes = len(train_df['Category'].value_counts())
    vocab = Vocabulary(train_texts)
    vocab_size = len(vocab.word2idx)
    train_dataloader = get_dataloader(
        train_texts, train_targets, vocab, batch_size=config.batch_size, shuffle=True
    )

    # Load and preprocess validation data
    val_df = pd.read_csv(config.val_path, encoding='utf-8')
    val_texts = val_df['Text'].to_list()
    val_targets = val_df['Category'].tolist()
    val_dataloader = get_dataloader(val_texts, val_targets, vocab, batch_size=config.batch_size, shuffle=False)

    model = RNN(
        vocab_size,
        config.embed_size,
        config.embed_proj_size,
        config.hidden_size,
        config.is_bidirectional,
        config.num_layers,
        num_classes,
        config.dropout_rate
    ).to(device)

    if config.checkpoint_path:
        logging.info("Checkpoint is loaded and model if finetuning")
        # model.load_state_dict(torch.load(config.checkpoint_path))

    # Train the model
    logging.info("Training started.")
    train(model, train_dataloader, val_dataloader, config.num_epochs, config.lr, device)
    logging.info("Training completed.")


