import math
import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import Config
from src.dataset import get_loaders
from src.models import ImageCaptioningModel

def train():
    device = torch.device(Config.DEVICE)
    train_loader, val_loader, _, vocab = get_loaders()
    
    pad_idx = vocab.word2idx[vocab.PAD_TOKEN]
    model = ImageCaptioningModel(vocab_size=len(vocab), pad_idx=pad_idx).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = math.inf
    stale_epochs = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Training
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]"):
            features = batch["image_features"].to(device)
            captions = batch["captions"].to(device)

            optimizer.zero_grad()
            outputs = model(features, captions)
            
            # targets: (B * Seq_Len,)
            targets = captions.reshape(-1)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]"):
                features = batch["image_features"].to(device)
                captions = batch["captions"].to(device)
                
                outputs = model(features, captions)
                targets = captions.reshape(-1)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "config": Config
            }, Config.MODEL_PATH)
            print(f"Saved optimal checkpoint: {Config.MODEL_PATH}")
        else:
            stale_epochs += 1
            if stale_epochs >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    train()