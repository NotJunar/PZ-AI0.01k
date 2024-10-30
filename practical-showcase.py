import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import random
import os

class RizzlineDataset(Dataset):
    def __init__(self, lines, seq_length):
        self.lines = lines
        self.seq_length = seq_length
        self.chars = sorted(list(set(''.join(lines))))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return sum(max(1, len(line) - self.seq_length) for line in self.lines)

    def __getitem__(self, index):
        for line in self.lines:
            if index < max(1, len(line) - self.seq_length):
                seq = line[index:index + self.seq_length].ljust(self.seq_length)
                target = line[index + 1:index + self.seq_length + 1].ljust(self.seq_length)
                return (
                    torch.tensor([self.char_to_idx.get(c, 0) for c in seq]),
                    torch.tensor([self.char_to_idx.get(c, 0) for c in target])
                )
            index -= max(1, len(line) - self.seq_length)
        raise IndexError('Index out of range')

class RizzlineGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RizzlineGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

class RizzlineMaster:
    def __init__(self, embed_size=64, hidden_size=256, num_layers=2, seq_length=20, batch_size=64, learning_rate=0.001, epochs=100):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, lines):
        dataset = RizzlineDataset(lines, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        
        self.model = RizzlineGenerator(dataset.vocab_size, self.embed_size, self.hidden_size, self.num_layers).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                
                self.model.zero_grad()
                output, _ = self.model(batch_inputs)
                loss = criterion(output.reshape(-1, dataset.vocab_size), batch_targets.reshape(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.dataset = dataset

    def generate(self, start_text, max_length=100, temperature=0.8):
        self.model.eval()
        chars = [ch for ch in start_text]
        hidden = None
        
        for _ in range(max_length - len(start_text)):
            x = torch.tensor([[self.dataset.char_to_idx.get(c, 0) for c in chars[-self.seq_length:]]]).to(self.device)
            output, hidden = self.model(x, hidden)
            
            probs = torch.softmax(output[0, -1] / temperature, dim=0).cpu().detach().numpy()
            next_char_idx = np.random.choice(len(probs), p=probs)
            next_char = self.dataset.idx_to_char[next_char_idx]
            
            chars.append(next_char)
            if next_char == '\n':
                break
        
        return ''.join(chars)

def load_rizzlines(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        print(f"File {file_path} not found. Using sample data.")
        return [
            "Are you a magician? Because whenever I look at you, everyone else disappers.",
            "Do you have a map? I just keep getting lost in your eyes.",
            "Is your name Google? Because you've got everything I've been searching for.",
            "Are you a camera? Every time I look at you, I smile.",
            "Do you believe in love at first sight, or should I walk by again?",
        ]

def generate_rizzlines(model, num_lines=5, temperature=0.8):
    start_phrases = [
        "Are you a",
        "Do you believe in",
        "I must be a",
        "Is your name",
        "Can I be your",
    ]
    
    generated_lines = []
    for _ in range(num_lines):
        start = random.choice(start_phrases)
        line = model.generate(start, temperature=temperature)
        generated_lines.append(line.strip())
    
    return generated_lines

if __name__ == "__main__":
    rizzlines = load_rizzlines("rizzlines.txt")
    
    rizzmaster = RizzlineMaster(epochs=100)
    rizzmaster.train(rizzlines)
    
    print("Generated Rizzlines:")
    for line in generate_rizzlines(rizzmaster, num_lines=5):
        print(line)

    while True:
        user_input = input("\nEnter a start phrase (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        generated = rizzmaster.generate(user_input, temperature=0.8)
        print(f"Generated Rizzline: {generated}")
