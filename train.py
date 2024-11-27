import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import LyricGenerationDataset
from module import LyricGenerationModule
from config import *
from tqdm import tqdm

def pad_sequences(sequences, max_len=None, padding_value=0):
    """
    对输入的序列列表进行填充，使所有序列长度相同。
    
    参数:
    - sequences (list of list of int): 需要填充的序列列表。
    - max_len (int, optional): 填充到的最大长度。如果为 None，则使用序列中最长的长度。
    - padding_value (int, optional): 填充值。默认是 0。

    返回:
    - padded_sequences (list of list of int): 填充后的序列列表。
    - lengths (list of int): 每个序列的原始长度。
    """
    if not sequences:
        return [], []

    # 如果未指定 max_len，则取最长序列的长度
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    lengths = []

    for seq in sequences:
        # 记录原始长度
        original_length = len(seq)
        lengths.append(original_length)

        # 如果序列长度小于 max_len，用 padding_value 填充
        if original_length < max_len:
            seq = seq + [padding_value] * (max_len - original_length)
        # 如果序列长度大于 max_len，进行截断
        else:
            seq = seq[:max_len]

        padded_sequences.append(seq)

    return padded_sequences, lengths


def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x, _ = pad_sequences(x , padding_value=0)
    y, _ = pad_sequences(y, padding_value=0)

    return torch.tensor(x, dtype=torch.long), torch.tensor(y , dtype=torch.long)

def train():
    train_datasets = LyricGenerationDataset(32)
    train_dataloader = DataLoader(train_datasets , shuffle=True , batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
    model = LyricGenerationModule()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        total_loss = 0
        total_num = 0

        pg = tqdm(train_dataloader)

        for x , y in pg:
            hidden = model.init_hidden().to(device)
            x = x.to(device)
            y = y.to(device)

            y_pred , hidden = model(x , hidden)

            loss = criterion(y_pred.permute(1, 2, 0) , y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (loss.item())
            total_num += 1

            pg.update(1)
            pg.set_description(f"epoch : {epoch + 1} average_loss : {total_loss / total_num}")

        torch.save(model , f"models/lyrics_generation_model_{epoch + 1}.bin")

    

if __name__ == "__main__":
    train()