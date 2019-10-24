import torch
import numpy as np
from torch.utils.data import Dataset
import csv
import pickle

# custom dataloader
class MyDataset(Dataset):
    def __init__(self, data_path, class_path=None, max_length=200):
        self.data_path = data_path
        self.vocabulary = []
        with open("vocab_list.pkl", 'rb') as vocab_file:
            self.vocabulary = pickle.load(vocab_file)

        #self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                # if len(line) == 3:
                #     text = "{} {}".format(line[1].lower(), line[2].lower())
                # else:
                #     text = "{}".format(line[1].lower())
                label = int(line[0])
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        if class_path:
            self.num_classes = sum(1 for _ in open(class_path))

    # gets the length
    def __len__(self):
        return self.length

    # gets data based on given index
    # done the encoding here itself
    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.vocabulary.index(i) if i in self.vocabulary else 0 for i in raw_text],
                        dtype=np.long)
        #print(len(data))

        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.pad(data,(0,(self.max_length - len(data))),'constant')
        elif len(data) == 0:
            data = np.pad(data,(0,(self.max_length)),'constant')
        label = self.labels[index]
        #print(data)
        return data, label
