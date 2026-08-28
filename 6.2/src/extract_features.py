import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from src.config import Config

def extract_all_features():
    device = torch.device(Config.DEVICE)
    print(f"Loading pretrained ResNet-50 feature extractor on {device}...")
    
    # Load ResNet-50 without classification head
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    model = torch.nn.Sequential(*modules).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    features = {}
    image_files = [f for f in os.listdir(Config.IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Extracting features for {len(image_files)} images...")

    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(Config.IMAGE_DIR, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                feature = model(img_tensor)
                feature = feature.squeeze().cpu()  # Shape: (2048,)
                features[img_name] = feature
            except Exception as e:
                print(f"Error reading {img_name}: {e}")

    torch.save(features, Config.FEATURES_PATH)
    print(f"Features successfully saved to {Config.FEATURES_PATH}")

if __name__ == "__main__":
    extract_all_features()