import torch
import torch.nn as nn
from keras.datasets import imdb
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from ds import ListDataset




class IMDB():
    def __init__(self, device = 'cpu'):
        self.device = device
        self.vocab = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.train_ds = None
        self.val_ds = None

    def load(self, path):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(path=path)
        word_index = imdb.get_word_index()
        ds = ListDataset (self.x_train, self.y_train)
        self.train_ds, self.val_ds = random_split(list(ds), [20000, 5000])
        self.vocab = dict((i + 3, word) for (word, i) in word_index.items())
        self.vocab[1] = "[START]"
        self.vocab[2] = "[OOV]"

    def collate_batch(self, batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(_label)
            processed_text = torch.tensor(_text, dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(processed_text.size(0))
        label_list = torch.tensor(label_list).float()
        lengths = torch.tensor(lengths)
        padded_text_list = nn.utils.rnn.pad_sequence(
            text_list, batch_first=True)
        return padded_text_list.to(self.device), label_list.to(self.device), lengths.to(self.device)

    def loader(self, batch_size = 4):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_batch)


    def val_loader(self, batch_size = 4):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_batch)
