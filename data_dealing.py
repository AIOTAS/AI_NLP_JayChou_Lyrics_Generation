import jieba
import pickle

def data_dealing():
    word2idx = {}

    PAD_TOKEN = 0
    SPACE_TOKEN = 1

    word2idx["PAD"] = PAD_TOKEN
    word2idx[" "] = SPACE_TOKEN

    count = 2

    ids = []

    line_words = []

    with open("周杰伦歌词.txt" , "r" , encoding="utf-8") as fr:
        lines = fr.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            words = jieba.lcut(line)
            line_words.append(words)
            for word in words:
                if word not in word2idx:
                    word2idx[word] = count
                    count += 1

    # 生成ids
    for words in line_words:
        for word in words:
            ids.append(word2idx[word])

        ids.append(word2idx[" "])

    idx2word = {id : word for word , id in word2idx.items()}

    print(word2idx)
    print(idx2word)
    
    assert len(word2idx) == len(idx2word)

    print(ids)

    pickle.dump((word2idx , idx2word , ids) , open("datas/word2idx_idx2word_ids.pkl" , "wb"))


if __name__ == "__main__":
    data_dealing()