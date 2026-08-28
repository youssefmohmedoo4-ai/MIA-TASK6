import os
import torch

class Config:
    # Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_DIR, "Flickr8k_Dataset")
    TEXT_DIR = os.path.join(DATA_DIR, "Flickr8k_text")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    # File paths
    TOKEN_FILE = os.path.join(TEXT_DIR, "Flickr8k.token.txt")
    TRAIN_IMAGES_FILE = os.path.join(TEXT_DIR, "Flickr_8k.trainImages.txt")
    VAL_IMAGES_FILE = os.path.join(TEXT_DIR, "Flickr_8k.devImages.txt")
    TEST_IMAGES_FILE = os.path.join(TEXT_DIR, "Flickr_8k.testImages.txt")
    
    FEATURES_PATH = os.path.join(PROCESSED_DIR, "features_resnet50.pt")
    VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.pkl")
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_caption_model.pth")

    # Model Hyperparameters
    CNN_BACKBONE = "resnet50"
    FEATURE_DIM = 2048        # Output dimension of ResNet-50 pool layer
    EMBED_DIM = 300          # Word embedding dimension
    HIDDEN_DIM = 512         # LSTM hidden state size
    NUM_LAYERS = 1
    DROPOUT = 0.3

    # Training Parameters
    FREQ_THRESHOLD = 3       # Minimum word frequency for vocabulary
    MAX_SEQ_LEN = 35
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    EPOCHS = 20
    PATIENCE = 4
    GRAD_CLIP = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)