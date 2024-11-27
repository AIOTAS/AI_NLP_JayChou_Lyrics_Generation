import torch.nn as nn
import pickle
from config import *
import torch.nn.functional as F


class LyricGenerationModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        word2idx , idx2word , self.ids = pickle.load(open("datas/word2idx_idx2word_ids.pkl" , "rb"))
        self.embedding = nn.Embedding(num_embeddings=len(word2idx) , embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim , hidden_size=hidden_size,num_layers=1)

        self.out = nn.Linear(hidden_size , len(word2idx))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x , hidden):
        x = self.embedding(x)

        x, hidden = self.gru(x.transpose(1,0) , hidden)

        x = F.relu(x)

        x = self.dropout(x)

        out = self.out(x)

        return out , hidden
    
    def init_hidden(self):
        return torch.zeros(1, batch_size, hidden_size)
    

if __name__ == "__main__":
    model = LyricGenerationModule()
    print(model)