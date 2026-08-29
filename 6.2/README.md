# AI Image Caption Generator (ResNet-50 + LSTM)

## 📌 Project Overview
This project implements an end-to-end **Image Caption Generation** system using PyTorch. It bridges Computer Vision and Natural Language Processing by taking an input image, extracting its deep visual features using a convolutional backbone, and translating those features into a descriptive, grammatically coherent natural-language sentence using an autoregressive recurrent neural network.

---

## 📊 Dataset
* **Source:** **Flickr8k Dataset**
* **Contents:** 
  * 8,000 photographs collected from Flickr, depicting diverse everyday scenes, people, and animals.
  * Each image is paired with **5 independent human-written reference captions** to capture varied descriptions of the same scene.
* **Splits:** Standardized splits for training, validation, and testing.

---

## 🏗️ Architecture
The system follows an **Encoder-Decoder** paradigm:

1. **Vision Encoder (CNN):** 
   * Uses a pre-trained **ResNet-50** network (transfer learning).
   * The final classification layer is removed, yielding a fixed **2048-dimensional feature vector** for each image that encodes high-level visual concepts.
2. **Language Decoder (RNN/LSTM):** 
   * An embedding layer maps input caption tokens to dense vectors.
   * An autoregressive **LSTM network** takes the image feature vector as its initial hidden state and generates text token-by-token.

---

## ⚙️ Preprocessing
* **Images:** Resized to $224 \times 224$ pixels, normalized using ImageNet statistics, and processed through ResNet-50. Features are cached to disk to drastically accelerate training epochs.
* **Text:** 
  * Converted to lowercase, stripped of punctuation, and filtered by a minimum frequency threshold.
  * Augmented with special tokens: `<start>`, `<end>`, `<pad>`, and `<unk>`.
  * Padded/truncated to a fixed maximum sequence length.

---

## 🚀 Training Process
* **Optimization:** Optimized using the **Adam optimizer** with cross-entropy loss.
* **Loss Masking:** Padding tokens are explicitly ignored during loss calculation to prevent the model from learning to predict empty padding space.
* **Monitoring:** Evaluated on validation loss checkpoints to select the best-performing weights (`best_caption_model.pth`).

---

## 📈 Evaluation Metrics & Results
The model was rigorously evaluated on the test set against human reference captions using standard natural language generation metrics:

* **BLEU-1 Score:** `59.12%` (Measures unigram precision/vocabulary matching)
* **BLEU-2 Score:** `40.92%` (Measures bigram overlap)
* **BLEU-3 Score:** `28.00%` (Measures trigram overlap)
* **BLEU-4 Score:** `18.80%` (Measures 4-gram phrase precision)
* **ROUGE-L F1:** `44.63%` (Evaluates longest common subsequence and structural gist)

---

