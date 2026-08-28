import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from src.config import Config
from src.vocab import Vocabulary
from src.dataset import parse_captions, load_split_image_names
from src.inference import CaptionGenerator

def evaluate_test_set(sample_limit=None):
    generator = CaptionGenerator()
    test_images = list(load_split_image_names(Config.TEST_IMAGES_FILE))
    raw_captions = parse_captions(Config.TOKEN_FILE)

    if sample_limit:
        test_images = test_images[:sample_limit]

    references = []
    hypotheses = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []

    print(f"Evaluating {len(test_images)} test images...")

    for img_name in tqdm(test_images):
        img_path = os.path.join(Config.IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        pred_caption = generator.generate_caption(img_path)
        actual_captions = [Vocabulary.clean_text(c) for c in raw_captions.get(img_name, [])]

        # Multi-reference token lists for BLEU
        ref_tokens = [c.split() for c in actual_captions]
        hypo_tokens = pred_caption.split()

        references.append(ref_tokens)
        hypotheses.append(hypo_tokens)

        # ROUGE against the best-matching reference
        max_rouge = max([scorer.score(c, pred_caption)['rougeL'].fmeasure for c in actual_captions], default=0.0)
        rouge_scores.append(max_rouge)

    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth) * 100
    b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth) * 100
    b3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth) * 100
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100
    avg_rouge = (sum(rouge_scores) / max(len(rouge_scores), 1)) * 100

    print("\n" + "=" * 45)
    print("       MODEL TEST EVALUATION RESULTS")
    print("=" * 45)
    print(f"BLEU-1 Score: {b1:.2f}%")
    print(f"BLEU-2 Score: {b2:.2f}%")
    print(f"BLEU-3 Score: {b3:.2f}%")
    print(f"BLEU-4 Score: {b4:.2f}%")
    print(f"ROUGE-L F1  : {avg_rouge:.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    evaluate_test_set()