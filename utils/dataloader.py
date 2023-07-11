import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence

from constant import *

class MyDataset(Dataset):
    def __init__(self, Test=False):
        '''
        RAWDATA_LINES = open(DATA_FILENAME).readlines()

        #   Remove csv header line
        RAWDATA_LINES = RAWDATA_LINES[1:]

        random.Random(SEED).shuffle(RAWDATA_LINES)
        
        if Test:
            RAWDATA_LINES = RAWDATA_LINES[int(len(RAWDATA_LINES)*0.9):]
        else:
            RAWDATA_LINES = RAWDATA_LINES[:int(len(RAWDATA_LINES)*0.9)]

        self.data = []
        
        #   Init GloVe
        glove = GloVe(name='6B', dim=300, cache="./data/glove")

        y_idx = {}
        
        for line in RAWDATA_LINES:
            splits = line.split(",")

            TwitterID = splits[0]
            emo = splits[1]
            text = ''.join(splits[2:])
            text = text[:-1] if text[-1] == "\n" else text

            if emo not in y_idx:
                y_idx[emo] = len(y_idx)

            # Convert text to GloVe embeddings
            try:
                text_emb = torch.stack([glove[word.lower()] for word in text.split() if word.lower() in glove.stoi])
                item = {
                    "text_emb": text_emb,
                    "len": text_emb.shape[0],
                    "emotion": y_idx[emo]
                }
                if item["len"] <= 5:
                    raise ValueError("Too short")
                self.data.append(item)
            except:
                pass
        
        global CLASS_NUM
        CLASS_NUM = len(y_idx)

        self.len = len(self.data)
        '''

        #   Read data
        if not Test:
            RAWDATA_LINES = open(TRAIN_FILENAME).readlines()
        else:
            RAWDATA_LINES = open(TEST_FILENAME).readlines()

        self.data = []
        
        #   Init GloVe
        glove = GloVe(name='6B', dim=300, cache="./data/glove")

        global y_idx 
        y_idx = {}
        
        for line in RAWDATA_LINES:
            splits = line.split(";")

            text = splits[0]
            emo = splits[1]
            text = text[:-1] if text[-1] == "\n" else text

            if emo not in y_idx:
                y_idx[emo] = len(y_idx)

            # Convert text to GloVe embeddings
            try:
                text_emb = torch.stack([glove[word.lower()] for word in text.split() if word.lower() in glove.stoi])
                item = {
                    "text_emb": text_emb,
                    "len": text_emb.shape[0],
                    "emotion": y_idx[emo]
                }
                if item["len"] <= 5:
                    raise ValueError("Too short")
                self.data.append(item)
            except:
                pass
        
        global CLASS_NUM
        CLASS_NUM = len(y_idx)

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx]

TrainDataSet = MyDataset()
TestDataSet = MyDataset(Test=True)

def collate_fn(batch):
    #   sort the batch to pad seq.
    batch.sort(key=lambda x: x["len"], reverse=True)
    seq = [item["text_emb"] for item in batch]
    seq_padded = pad_sequence(seq, batch_first=True)

    # Get the labels
    labels = [item["emotion"] for item in batch]
    #masks = [[True]*item["len"] + [False]*(batch[0]["len"]-item["len"]) for item in batch]
    lens = [item["len"] for item in batch]

    return seq_padded, torch.tensor(labels), lens

MyTrainDataLoader = DataLoader(TrainDataSet,
                batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=8,
                pin_memory=True, pin_memory_device=DEVICE)

MyTestDataLoader = DataLoader(TestDataSet,
                batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=8,
                pin_memory=True, pin_memory_device=DEVICE)