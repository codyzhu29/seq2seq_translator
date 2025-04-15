import os
from google.colab import drive
drive.mount('/content/drive')
!pip install sentencepiece
!pip install sacrebleu
!pip install beautifulsoup4
!pip install sentencepiece
import sentencepiece as spm


input_file_en = 'Wikipedia.en-fr.en'  
input_file_fr = 'Wikipedia.en-fr.fr'  

with open('wikipedia_en_fr.txt', 'w') as f_out:
    with open(input_file_en, 'r') as f_en, open(input_file_fr, 'r') as f_fr:
        for en_line, fr_line in zip(f_en, f_fr):
            f_out.write(en_line.strip() + ' ' + fr_line.strip() + '\n')

# train bpe_model
spm.SentencePieceTrainer.train(input='wikipedia_en_fr.txt', model_prefix='bpe_model', vocab_size=30000, character_coverage=0.995, model_type='bpe')

import torch
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import numpy as np
from bs4 import BeautifulSoup

sp_model = spm.SentencePieceProcessor()
sp_model.load("/content/drive/MyDrive/dataset/wikipedia_en_fr/bpe_model.model")


def tokenize_data(file_path, sp_model, is_sgm=False):
    if is_sgm:
        with open(file_path, 'r', encoding='utf-8') as f:
            sgm_content = f.read()
        soup = BeautifulSoup(sgm_content, 'html.parser')
        sentences = [sentence.text for sentence in soup.find_all('seg')]
        for sentence in sentences:
            yield sp_model.encode(sentence, out_type=int)
    else:
        with open(file_path, 'r') as f:
            for line in f:
                yield sp_model.encode(line.strip(), out_type=int)

# Tokenize（French→English）
train_fr = list(tokenize_data('/content/drive/MyDrive/dataset/wikipedia_en_fr/Wikipedia.en-fr.fr', sp_model))  # French
train_en = list(tokenize_data('/content/drive/MyDrive/dataset/wikipedia_en_fr/Wikipedia.en-fr.en', sp_model))  # English
# 测试集
bleu_fr = list(tokenize_data('/content/drive/MyDrive/testset/test-full/newstest2014-fren-src.fr.sgm', sp_model, is_sgm=True))  
bleu_en = list(tokenize_data('/content/drive/MyDrive/testset/test-full/newstest2014-fren-ref.en.sgm', sp_model, is_sgm=True))  

train_size = int(0.8 * len(train_fr))
test_size = len(train_fr) - train_size
train_fr, test_fr = train_fr[:train_size], train_fr[train_size:]
train_en, test_en = train_en[:train_size], train_en[train_size:]

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, max_len=50):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.max_len = max_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx][:self.max_len] + [0] * (self.max_len - len(self.src_data[idx]))
        tgt = self.tgt_data[idx][:self.max_len] + [0] * (self.max_len - len(self.tgt_data[idx]))
        return torch.tensor(src), torch.tensor(tgt)

train_dataset = TranslationDataset(train_fr, train_en)
test_dataset = TranslationDataset(test_fr, test_en)
bleu_dataset = TranslationDataset(bleu_fr, bleu_en)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
bleu_loader = DataLoader(bleu_dataset, batch_size=64, shuffle=False)