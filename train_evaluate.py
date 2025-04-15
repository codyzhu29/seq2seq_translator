import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy as np
import torch.nn.functional as F
import sacrebleu
import random

def save_checkpoint(model, optimizer, epoch, checkpoint_path="/content/drive/MyDrive/dataset/wikipedia_en_fr/checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path="/content/drive/MyDrive/dataset/wikipedia_en_fr/checkpoint.pth"):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return model, optimizer, start_epoch

def train(model, criterion, optimizer, train_loader, device, fraction=0.1):
    model.train()
    total_loss = 0
    num_batches = int(len(train_loader) * fraction)  # train fraction of dataset
    print(f"num_batches: {num_batches}")
    all_indices = list(range(len(train_loader)))
    selected_indices = random.sample(all_indices, num_batches)  # randomly pick batch
    for i, (src, tgt) in enumerate(train_loader):
        if i not in selected_indices:  # 只训练选中的 batch
            continue
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt) # Pass both src and tgt to the model
        output_flat = output.view(-1, output.shape[-1])
        tgt_flat = tgt.view(-1)
        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Train Step {i}, Loss: {loss.item()}")
    print(f"Epoch Training Loss: {total_loss / num_batches}")

def evaluate(model, test_loader, device, criterion, fraction=0.1):
    model.eval()
    total_loss = 0
    references, hypotheses = [], []
    num_batches = int(len(test_loader) * fraction)
    print(f"num_batches:{num_batches}")
    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_loader):
            if i >= num_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            output = output.float()
            output_tokens = output.argmax(dim=-1)
            loss = criterion(output.view(-1, output.shape[-1]), tgt.view(-1))
            total_loss += loss.item()
            idx_batch_num = i
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")


def bleu(model, test_loader, device, criterion, fraction=0.1):
    model.eval()
    total_loss = 0
    references, hypotheses = [], []
    num_batches = int(len(test_loader) * fraction)
    print(f"num_batches:{num_batches}")
    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_loader):
            if i >= num_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            output = output.float() 
            output_tokens = output.argmax(dim=-1)
            loss = criterion(output.view(-1, output.shape[-1]), tgt.view(-1))
            total_loss += loss.item()
            idx_batch_num = i
            #BLEU check
            for i in range(src.size(0)): #src.size(0) represents batch_size
                pred_tokens = output_tokens[i][output_tokens[i] != 0].tolist()  
                ref_tokens = tgt[i][tgt[i] != 0].tolist()  
                input_tokens = src[i][src[i] != 0].tolist()  
                if ref_tokens and pred_tokens:
                    pred_sentence = sp_model.decode(pred_tokens)
                    ref_sentence = sp_model.decode(ref_tokens)
                    input_sentence = sp_model.decode(input_tokens)
                    hypotheses.append(pred_sentence)
                    references.append(ref_sentence)
                    single_bleu = sacrebleu.sentence_bleu(pred_sentence, [ref_sentence])
                    if i == 10 and idx_batch_num==10: #choose certain sentence to show
                        print(f"Input Tokens: {input_sentence}")
                        print(f"Prediction: {pred_sentence}")
                        print(f"Reference: {ref_sentence}")
                        #print(f"Prediction Tokens: {pred_tokens}")
                        #print(f"Reference Tokens: {ref_tokens}")
                        print(f"Single BLEU Score: {single_bleu.score}")

    avg_loss = total_loss / len(test_loader)
    avg_bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Average BLEU Score: {avg_bleu.score}")
    return avg_bleu

input_dim = len(sp_model)
output_dim = len(sp_model)
embedding_dim = 256
hidden_dim = 512
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(input_dim, embedding_dim, hidden_dim, dropout)
decoder = Decoder(output_dim, embedding_dim, hidden_dim, dropout)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, the_id=19, penalty_weight=2.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.the_id = the_id
        self.penalty_weight = penalty_weight

    def forward(self, output, target):
        loss = self.criterion(output, target)
        mask = ((output.argmax(-1) == self.the_id) & (target != self.the_id)).float() 
        penalty = self.penalty_weight * mask.mean()  # punish "the" with 2x loss
        return loss + penalty

checkpoint_path = "/content/drive/MyDrive/dataset/wikipedia_en_fr/checkpoint.pth"
start_epoch = 0  
model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
#criterion = WeightedCrossEntropyLoss()

# try to load model if there is one
try:
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
except FileNotFoundError:
    print("No checkpoint found, starting fresh training.")

# for each epoch, train + evaluate + test(bleu) + save_checkpoint
epochs = 4
for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch}")
    #train fraction = 100%, 60%, 40%, 20%
    fraction = 1 if epoch == 0 else (0.6 if epoch == 1 else (0.4 if epoch == 2 else 0.2))
    train(model, criterion, optimizer, train_loader, device, fraction=fraction)
    evaluate(model, test_loader, device, criterion, fraction=1)
    bleu(model, bleu_loader, device, criterion, fraction=1)
    checkpoint_path = f"/content/drive/MyDrive/dataset/wikipedia_en_fr/checkpoint_en_fr_epoch_{epoch}.pth"
    save_checkpoint(model, optimizer, epoch + 1, checkpoint_path) #save checkpoint