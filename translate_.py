import pandas as pd
import re
from collections import Counter
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer

torch.manual_seed(23)


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device("cpu")


PATH = '../datasets/idiomas-engl-span.tsv'

with open(PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
pairs_engl_span = [line.strip().split('\t') for line in lines if '\t' in line]
#print(pairs_engl_span[:5])

engl_sentences = [pair[1] for pair in pairs_engl_span]
span_sentences = [pair[3] for pair in pairs_engl_span]
#print(engl_sentences[:10])
#print(span_sentences[:10])

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[á]+", "a", sentence)
    sentence = re.sub(r"[é]+", "e", sentence)
    sentence = re.sub(r"[í]+", "i", sentence)
    sentence = re.sub(r"[ó]+", "o", sentence)
    sentence = re.sub(r"[ú]+", "u", sentence)
    sentence = re.sub(r"[^a-z]+", " ", sentence)
    sentence = sentence.strip()
    sentence = '<sos> ' + sentence + ' <eos>'
    return sentence
#s1 = '¿Hola @ cómo estás? 123'
#print(s1)
#print(preprocess_sentence(s1))

engl_sentences = [preprocess_sentence(sentence) for sentence in engl_sentences]
span_sentences = [preprocess_sentence(sentence) for sentence in span_sentences]
#print(engl_sentences[:10])
#print(span_sentences[:10])
print(len(engl_sentences))
print(len(span_sentences))


def build_vocab(sentences):
    words = [word for sentence in sentences for word in sentence.split()]
    word_count = Counter(words)
    sorted_word_counts = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
    word2idx = {word: idx for idx, (word, _) in enumerate(sorted_word_counts, 2)}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word
eng_word2idx, eng_idx2word = build_vocab(engl_sentences)
spa_word2idx, spa_idx2word = build_vocab(span_sentences)
eng_vocab_size = len(eng_word2idx)
spa_vocab_size = len(spa_word2idx)
#print(eng_vocab_size, spa_vocab_size)
#print(eng_idx2word)
#print(spa_idx2word)

class EngSpaDataset(Dataset):
    def __init__(self, eng_sentences, spa_sentences, eng_word2idx, spa_word2idx):
        self.eng_sentences = eng_sentences
        self.spa_sentences = spa_sentences
        self.eng_word2idx = eng_word2idx
        self.spa_word2idx = spa_word2idx
        
    def __len__(self):
        return len(self.eng_sentences)
    
    def __getitem__(self, idx):
        eng_sentence = self.eng_sentences[idx]
        spa_sentence = self.spa_sentences[idx]
        # return tokens idxs
        eng_idxs = [self.eng_word2idx.get(word, self.eng_word2idx['<unk>']) for word in eng_sentence.split()]
        spa_idxs = [self.spa_word2idx.get(word, self.spa_word2idx['<unk>']) for word in spa_sentence.split()]
        
        return torch.tensor(eng_idxs), torch.tensor(spa_idxs)

def collate_fn(batch):
    eng_batch, spa_batch = zip(*batch)
    eng_batch = [seq[:MAX_SEQ_LEN].clone().detach() for seq in eng_batch]
    spa_batch = [seq[:MAX_SEQ_LEN].clone().detach() for seq in spa_batch]
    eng_batch = torch.nn.utils.rnn.pad_sequence(eng_batch, batch_first=True, padding_value=0)
    spa_batch = torch.nn.utils.rnn.pad_sequence(spa_batch, batch_first=True, padding_value=0)
    return eng_batch, spa_batch


def train(model, dataloader, loss_function, optimiser, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0 
        for i, (eng_batch, spa_batch) in enumerate(dataloader):
            eng_batch = eng_batch.to(device)
            spa_batch = spa_batch.to(device)
            # Decoder preprocessing
            target_input = spa_batch[:, :-1]
            target_output = spa_batch[:, 1:].contiguous().view(-1)
            # Zero grads
            optimiser.zero_grad()
            # run model
            output = model(eng_batch, target_input)
            output = output.view(-1, output.size(-1))
            # loss\
            loss = loss_function(output, target_output)
            # gradient and update parameters
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f'Epoch: {epoch}/{epochs}, Loss: {avg_loss:.4f}')
            
BATCH_SIZE = 64
dataset = EngSpaDataset(eng_sentences, spa_sentences, eng_word2idx, spa_word2idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = Transformer(d_model=512, num_heads=8, d_ff=2048, num_layers=6,
                    input_vocab_size=eng_vocab_size, target_vocab_size=spa_vocab_size,
                    max_len=MAX_SEQ_LEN, dropout=0.1)

model = model.to(device)
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimiser = optim.Adam(model.parameters(), lr=0.0001)

train(model, dataloader, loss_function, optimiser, epochs = 10)






