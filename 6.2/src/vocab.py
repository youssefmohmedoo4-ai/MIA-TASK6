import re
import pickle
from collections import Counter
from src.config import Config

class Vocabulary:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<start>"
    EOS_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold=Config.FREQ_THRESHOLD):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z\s]+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def build_vocabulary(self, sentence_list):
        counter = Counter()
        for sentence in sentence_list:
            cleaned = self.clean_text(sentence)
            tokens = cleaned.split()
            counter.update(tokens)

        for word, count in counter.items():
            if count >= self.freq_threshold and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, text: str, max_len=Config.MAX_SEQ_LEN) -> list:
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        indices = [self.word2idx[self.SOS_TOKEN]]
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]))
            
        indices.append(self.word2idx[self.EOS_TOKEN])
        
        # Truncate if necessary (keeping EOS at the end)
        if len(indices) > max_len:
            indices = indices[:max_len - 1] + [self.word2idx[self.EOS_TOKEN]]
            
        return indices

    def decode(self, indices, skip_special=True) -> str:
        special_tokens = {0, 1, 2, 3} if skip_special else set()
        words = []
        for idx in indices:
            idx = int(idx)
            if idx == self.word2idx[self.EOS_TOKEN]:
                break
            if idx not in special_tokens:
                words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return " ".join(words)

    def save(self, file_path=Config.VOCAB_PATH):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path=Config.VOCAB_PATH):
        with open(file_path, "rb") as f:
            return pickle.load(f)

# hello
