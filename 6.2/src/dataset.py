import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.config import Config
from src.vocab import Vocabulary

def load_split_image_names(split_file):
    with open(split_file, "r") as f:
        return set(line.strip() for line in f if line.strip())

def parse_captions(token_file):
    image_to_captions = {}
    with open(token_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name = parts[0].split("#")[0].strip()
                caption = parts[1].strip()
                image_to_captions.setdefault(img_name, []).append(caption)
    return image_to_captions

class Flickr8kDataset(Dataset):
    def __init__(self, split_file, features_dict, vocab, is_train=True):
        self.features_dict = features_dict
        self.vocab = vocab
        self.is_train = is_train
        
        allowed_images = load_split_image_names(split_file)
        raw_captions = parse_captions(Config.TOKEN_FILE)
        
        self.samples = []
        for img_name, captions in raw_captions.items():
            if img_name in allowed_images and img_name in self.features_dict:
                for cap in captions:
                    self.samples.append((img_name, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        feature = self.features_dict[img_name]
        numericalized_caption = self.vocab.numericalize(caption)
        
        return {
            "image_feature": feature,
            "caption": torch.tensor(numericalized_caption, dtype=torch.long),
            "img_name": img_name,
            "raw_caption": caption
        }

class CollateCaps:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        features = torch.stack([item["image_feature"] for item in batch], dim=0)
        captions = [item["caption"] for item in batch]
        
        padded_captions = pad_sequence(
            captions, 
            batch_first=True, 
            padding_value=self.pad_idx
        )
        
        img_names = [item["img_name"] for item in batch]
        raw_captions = [item["raw_caption"] for item in batch]

        return {
            "image_features": features,
            "captions": padded_captions,
            "img_names": img_names,
            "raw_captions": raw_captions
        }

def get_loaders():
    if not os.path.exists(Config.FEATURES_PATH):
        raise FileNotFoundError(f"Missing {Config.FEATURES_PATH}. Run extract_features.py first.")
        
    features = torch.load(Config.FEATURES_PATH)
    
    # Build or load vocabulary
    if not os.path.exists(Config.VOCAB_PATH):
        print("Building vocabulary from training captions...")
        train_imgs = load_split_image_names(Config.TRAIN_IMAGES_FILE)
        raw_captions = parse_captions(Config.TOKEN_FILE)
        train_caps = [c for img, caps in raw_captions.items() if img in train_imgs for c in caps]
        vocab = Vocabulary()
        vocab.build_vocabulary(train_caps)
        vocab.save()
    else:
        vocab = Vocabulary.load()

    train_dataset = Flickr8kDataset(Config.TRAIN_IMAGES_FILE, features, vocab, is_train=True)
    val_dataset = Flickr8kDataset(Config.VAL_IMAGES_FILE, features, vocab, is_train=False)
    test_dataset = Flickr8kDataset(Config.TEST_IMAGES_FILE, features, vocab, is_train=False)

    collate_fn = CollateCaps(pad_idx=vocab.word2idx[Vocabulary.PAD_TOKEN])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, vocab