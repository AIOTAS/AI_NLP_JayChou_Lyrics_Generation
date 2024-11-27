from torch.utils.data import Dataset
import pickle


class LyricGenerationDataset(Dataset):
    def __init__(self, num_chars):
        super().__init__()
        self.num_chars = num_chars
        word2idx , idx2word , self.ids = pickle.load(open("datas/word2idx_idx2word_ids.pkl" , "rb"))

        self.length = len(self.ids) - self.num_chars

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        x = self.ids[index : index + self.num_chars]
        y = self.ids[ index + 1 : index + self.num_chars + 1]

        return x, y
    

if __name__ == "__main__":
    train_datsets = LyricGenerationDataset(32)
    print(train_datsets[0])
        