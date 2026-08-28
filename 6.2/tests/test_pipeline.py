import torch
import pytest
from src.vocab import Vocabulary
from src.models import EncoderCNN, DecoderLSTM, ImageCaptioningModel

def test_vocabulary_numericalize_and_decode():
    vocab = Vocabulary(freq_threshold=1)
    sentences = ["a dog running in the grass", "a cat sleeping"]
    vocab.build_vocabulary(sentences)

    assert "<pad>" in vocab.word2idx
    assert "<unk>" in vocab.word2idx
    assert "<start>" in vocab.word2idx
    assert "<end>" in vocab.word2idx
    
    tokens = vocab.numericalize("a dog running")
    assert tokens[0] == vocab.word2idx["<start>"]
    assert tokens[-1] == vocab.word2idx["<end>"]
    
    decoded = vocab.decode(tokens)
    assert decoded == "a dog running"

def test_model_forward_tensor_shapes():
    batch_size = 4
    feature_dim = 2048
    embed_dim = 300
    seq_len = 10
    vocab_size = 100

    features = torch.randn(batch_size, feature_dim)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = ImageCaptioningModel(vocab_size=vocab_size)
    outputs = model(features, captions)

    # Output shape should be (Batch_Size, Seq_Len, Vocab_Size)
    assert outputs.shape == (batch_size, seq_len, vocab_size)