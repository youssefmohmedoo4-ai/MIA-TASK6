import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from src.config import Config
from src.vocab import Vocabulary
from src.models import ImageCaptioningModel

class CaptionGenerator:
    def __init__(self, model_path=Config.MODEL_PATH, vocab_path=Config.VOCAB_PATH):
        self.device = torch.device(Config.DEVICE)
        self.vocab = Vocabulary.load(vocab_path)
        
        # Load feature extractor
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.extractor.eval()

        # Load Captioning Model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = ImageCaptioningModel(
            vocab_size=checkpoint["vocab_size"],
            pad_idx=self.vocab.word2idx[Vocabulary.PAD_TOKEN]
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate_caption(self, image_input, max_len=Config.MAX_SEQ_LEN) -> str:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.extractor(img_tensor).squeeze()
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            
            projected = self.model.encoder(feature) # (1, Embed_Dim)

            # Autoregressive generation
            states = None
            inputs = projected.unsqueeze(1) # First step is image projection
            result_indices = []

            for _ in range(max_len):
                hiddens, states = self.model.decoder.lstm(inputs, states)
                logits = self.model.decoder.fc(hiddens.squeeze(1))
                predicted_idx = int(logits.argmax(dim=1).item())

                if predicted_idx == self.vocab.word2idx[Vocabulary.EOS_TOKEN]:
                    break

                result_indices.append(predicted_idx)
                # Next step input is the embedding of the predicted token
                inputs = self.model.decoder.embedding(
                    torch.tensor([[predicted_idx]], device=self.device)
                )

        return self.vocab.decode(result_indices)