import torch

DATA_FILENAME = "./data/tweet_emotions.csv"
TRAIN_FILENAME = "./data/train.txt"
TEST_FILENAME = "./data/test.txt"
BATCH_SIZE = 128
EMBED_DIM = 300
HEADS = 2
EPOCHS = 100
SEED = 114514

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"