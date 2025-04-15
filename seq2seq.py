import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy as np
import torch.nn.functional as F
import random


# Encoder（bidirectional GRU）
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # word embedding
        self.gru = nn.GRU(emb_dim, hidden_dim, bidirectional=True, batch_first=True)  # bidirectional GRU
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # reduce bidirectional hidden states to one
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # word embedding + dropout
        outputs, hidden = self.gru(embedded)  # GRU calculate output and hidden state
        #concatenate last forward and backward hidden states of bidirectional GRU then linear and tanh
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        hidden = hidden.unsqueeze(0)  # adapt to the shape of decoder
        return outputs, hidden  # output is dim*2 for attention, hidden is dim*1 for decoder

# Attention（Bahdanau Attention）
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim) #reduce to hidden_dim*1
        self.coverage_layer = nn.Linear(1, hidden_dim)  # add coverage layer
        self.v = nn.Linear(hidden_dim, 1, bias=False) # reduce hidden_dim vector to single score

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2)  # reorder to (batch_size, 1, hidden_dim)
        hidden = hidden.repeat(1, src_len, 1)  #repeat decoder hidden state across the time steps
        coverage = coverage.unsqueeze(2)  # (batch, src_len, 1)
        coverage_feat = self.coverage_layer(coverage)  # (batch, src_len, hidden)
        # concatenate decoder hidden state with encoder output along time steps
        # through linear layer and tanh to compute attention score
        energy_input = torch.cat((hidden, encoder_outputs), dim=2)  # (batch, src_len, hidden*2)
        energy = torch.tanh(self.attn(energy_input) + coverage_feat)  # add coverage impact
        attention = self.v(energy).squeeze(2)  # squeeze the hidden_dim
        return F.softmax(attention, dim=1)  # attention weights after softmax

# Decoder（GRU）
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim) # word embedding
        self.gru = nn.GRU(hidden_dim + emb_dim, hidden_dim, batch_first=True) # input is hidden+embedding
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)  # reduce context+hidden+embedding
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, input, hidden, encoder_outputs, prev_word):
    def forward(self, input, hidden, encoder_outputs, coverage):
        input = input.unsqueeze(1)  # unsqueeze to (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # embedding layer and dropout
        attn_weights = self.attention(hidden, encoder_outputs, coverage).unsqueeze(1)  # calculate attention weights
        context = torch.bmm(attn_weights, encoder_outputs)  # Weighted sum of encoder outputs
        context = context[:, :, :hidden_dim]  # keep the first hidden_dim
        rnn_input = torch.cat((embedded, context), dim=2)  # concatenate embedded input and context vector
        output, hidden = self.gru(rnn_input, hidden)  # through GRU to get new hidden state
        # combine output, context, and embedded input, then pass through the output layer
        output = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        for i in range(output.shape[0]):  
              output[i, 19] -= 5  # giving "the" (token_id=19) punishment
        # punish repeating "the"
        #if prev_word is not None:
        #    for i in range(output.shape[0]):  
        #        last_word = prev_word[i].item() if prev_word is not None else None
        #        if last_word == 19:
        #            output[i, 19] -= 5  # giving "the" (token_id=19) punishment
        coverage = coverage + attn_weights.squeeze(1)  # update coverage
        return output, hidden, coverage, attn_weights

# Seq2Seq (connecting encoder, decoder, attention)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        encoder_outputs, hidden = self.encoder(src)
        src_len = encoder_outputs.shape[1]
        trg_len = trg.shape[1] if trg is not None else 50 # target_len for training, 50 for inference
        trg_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        coverage = torch.zeros(batch_size, src_len).to(self.device)
        attn_weights_list = []

        if trg is None: # inference mode：trg=None，output step by step**
            input = torch.tensor([2] * batch_size).to(self.device)  # start with <BOS> token
            for t in range(trg_len):
                output, hidden, coverage, attn_weights = self.decoder(input, hidden, encoder_outputs, coverage) # decode one step
                outputs[:, t, :] = output # store output
                attn_weights_list.append(attn_weights)  
                top1 = output.argmax(1)  # get token with biggest propability
                input = top1  # predicted token as next input
                #prev_word = input.detach()  # obtain prev_word
                if (top1 == sp_model.eos_id()).all():  # end if <EOS> token
                    break
        else: # train mode: use target or predicted for input according to teaching ratio            
            input = trg[:, 0] # first input is <BOS> token from target
            for t in range(1, trg_len): 
                output, hidden, coverage = self.decoder(input, hidden, encoder_outputs, coverage) # decode one step
                outputs[:, t, :] = output # store output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio # decide teaching
                top1 = output.argmax(1) # predicted token
                input = trg[:, t] if teacher_force else top1 # choose next input 
        attn_weights_tensor = torch.stack(attn_weights_list, dim=1) if attn_weights_list else None
        return outputs, attn_weights_tensor

