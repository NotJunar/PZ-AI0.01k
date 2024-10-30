import random
import re

class SimpleLanguageModel:
    def __init__(self, n=2):
        self.n = n
        self.model = {}

    def train(self, text):
        words = re.findall(r'\w+', text.lower())
        for i in range(len(words) - self.n):
            context = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            if context not in self.model:
                self.model[context] = {}
            if next_word not in self.model[context]:
                self.model[context][next_word] = 0
            self.model[context][next_word] += 1

    def generate(self, start_words, num_words=20):
        current = tuple(start_words)
        result = list(current)
        for _ in range(num_words):
            if current in self.model:
                next_word = random.choices(list(self.model[current].keys()),
                                           weights=list(self.model[current].values()))[0]
                result.append(next_word)
                current = tuple(result[-self.n:])
            else:
                break
        return ' '.join(result)

# Example usage
model = SimpleLanguageModel(n=2)
training_text = """
The quick brown fox jumps over the lazy dog. 
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
"""
model.train(training_text)

generated_text = model.generate(["the", "quick"], num_words=10)
print(generated_text)
