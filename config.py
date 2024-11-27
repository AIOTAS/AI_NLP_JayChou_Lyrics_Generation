import torch

batch_size = 64
embedding_dim = 768
hidden_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10000000000

if __name__ == "__main__":
    print(device)