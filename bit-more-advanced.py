import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import random

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.words = self.tokenize(text)
        self.word_to_index = {word: i for i, word in enumerate(set(self.words))}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.word_indexes = [self.word_to_index[word] for word in self.words]

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def __len__(self):
        return len(self.word_indexes) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.word_indexes[idx:idx+self.seq_length]),
            torch.tensor(self.word_indexes[idx+1:idx+self.seq_length+1])
        )

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

class AdvancedLanguageModel:
    def __init__(self, embed_size=128, hidden_size=256, num_layers=2, seq_length=20, batch_size=32, learning_rate=0.001, epochs=10):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, text):
        dataset = TextDataset(text, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        vocab_size = len(dataset.word_to_index)
        self.model = LanguageModel(vocab_size, self.embed_size, self.hidden_size, self.num_layers).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                
                self.model.zero_grad()
                output, _ = self.model(batch_inputs)
                loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.dataset = dataset

    def generate(self, start_words, num_words=50, temperature=1.0):
        self.model.eval()
        words = start_words.lower().split()
        state = None
        
        for _ in range(num_words):
            x = torch.tensor([[self.dataset.word_to_index.get(w, random.randint(0, len(self.dataset.word_to_index)-1)) for w in words[:-1]]]).to(self.device)
            y_pred, state = self.model(x, state)
            
            last_word_logits = y_pred[0][-1] / temperature
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(p), p=p)
            words.append(self.dataset.index_to_word[word_index])
        
        return ' '.join(words)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerLanguageModel:
    def __init__(self, embed_size=128, nhead=4, num_layers=2, seq_length=20, batch_size=32, learning_rate=0.001, epochs=10):
        self.embed_size = embed_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, text):
        dataset = TextDataset(text, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        vocab_size = len(dataset.word_to_index)
        self.model = TransformerModel(vocab_size, self.embed_size, self.nhead, self.num_layers).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                
                self.model.zero_grad()
                output = self.model(batch_inputs)
                loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.dataset = dataset

    def generate(self, start_words, num_words=50, temperature=1.0):
        self.model.eval()
        words = start_words.lower().split()
        
        for _ in range(num_words):
            x = torch.tensor([[self.dataset.word_to_index.get(w, random.randint(0, len(self.dataset.word_to_index)-1)) for w in words[-self.seq_length:]]]).to(self.device)
            y_pred = self.model(x)
            
            last_word_logits = y_pred[0][-1] / temperature
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(p), p=p)
            words.append(self.dataset.index_to_word[word_index])
        
        return ' '.join(words)

def train_and_generate(model_class, text, start_words, num_words=50):
    model = model_class()
    model.train(text)
    generated_text = model.generate(start_words, num_words)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    sample_text = """
    The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question. All that glitters is not gold. Actions speak louder than words.
    """
    
    print("Training LSTM model:")
    train_and_generate(AdvancedLanguageModel, sample_text, "The quick")
    
    print("\nTraining Transformer model:")
    train_and_generate(TransformerLanguageModel, sample_text, "The quick")
