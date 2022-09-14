import numpy as np
from paddle.io import Dataset
from paddlenlp.transformers import ErnieTokenizer

__all__ = ["PuncDatasetFromErnieTokenizer"]

from tqdm import tqdm


class PuncDatasetFromErnieTokenizer(Dataset):
    def __init__(self, data_path, punc_path, pretrained_token='ernie-3.0-medium-zh', seq_len=100):
        super().__init__()
        self.tokenizer = ErnieTokenizer.from_pretrained(pretrained_token)
        self.paddingID = self.tokenizer.pad_token_id
        self.seq_len = seq_len
        self.punc2id = self.load_vocab(punc_path, extra_word_list=[" "])
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}
        tmp_seqs = open(data_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        self.preprocess(self.txt_seqs)

    def __len__(self):
        return self.in_len

    def __getitem__(self, index):
        return self.input_data[index], self.label[index]

    def load_vocab(self, vocab_path, extra_word_list=[]):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, txt_seqs: list):
        input_data = []
        label = []
        for i in tqdm(range(len(txt_seqs) - 1)):
            word = txt_seqs[i]
            punc = txt_seqs[i + 1]
            if word in self.punc2id:
                continue

            token = self.tokenizer(word)
            x = token["input_ids"][1:-1]
            input_data.extend(x)

            for i in range(len(x) - 1):
                label.append(self.punc2id[" "])

            if punc not in self.punc2id:
                label.append(self.punc2id[" "])
            else:
                label.append(self.punc2id[punc])

        if len(input_data) != len(label):
            assert 'error: length input_data != label'

        self.in_len = len(input_data) // self.seq_len
        len_tmp = self.in_len * self.seq_len
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]
        self.input_data = np.array(input_data, dtype='int64').reshape(-1, self.seq_len)
        self.label = np.array(label, dtype='int64').reshape(-1, self.seq_len)
