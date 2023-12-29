import torch
from torch.utils.data import Dataset , DataLoader
from collections import Counter
from preprocess import Vocabulary
from torch.nn.utils.rnn import pad_sequence

class NewsDataset(Dataset):
    def __init__(self, texts,targets,vocab):
        self.texts = texts
        self.targets = targets
        self.vocab = vocab
        class_freq = Counter(self.targets)
        sorted_class_freq = dict(sorted(class_freq.items(), key=lambda item: item[1], reverse=True))
        self.class_labels = {label: idx for idx, (label, _) in enumerate(sorted_class_freq.items())}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indexes = self.vocab(text)
        target = self.class_labels[self.targets[idx]]
        return {'input': torch.tensor(indexes) , 'target': torch.tensor(target)}


def my_collate(batch):
    texts = [item['input'] for item in batch]
    targets = torch.LongTensor([item['target'] for item in batch])
    tokenized_texts = [torch.tensor(text) for text in texts]
    padded_texts = pad_sequence([seq[:128] for seq in tokenized_texts], batch_first=True, padding_value=0)
    return {'input': padded_texts, 'target': targets}


def get_dataloader(texts,targets,vocab,batch_size,shuffle=True):
    dataset = NewsDataset(texts,targets,vocab)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=my_collate)
    return dataloader
