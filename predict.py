import torch
from config import *
import pickle

def predict():
    model = torch.load("models/lyrics_generation_model_61.bin")

    word2idx , idx2word , ids = pickle.load(open("datas/word2idx_idx2word_ids.pkl", "rb"))

    current_word = "关于"

    model = model.to(device)
    model.eval()

    hidden = torch.zeros(1,1,hidden_size).to(device)

    with torch.no_grad():
        for i in range(200):
            print(current_word , end="")
            current_token = torch.tensor([[word2idx[current_word]]] , dtype=torch.long).to(device)
            y_pred , hidden = model(current_token , hidden)
            current_word = idx2word[torch.argmax(y_pred, dim=-1).item()]
        
        print(current_word)



if __name__ == "__main__":
    predict()