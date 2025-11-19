# utils.py
import io
import random
import json
from PIL import Image
import pandas as pd
import datetime

# Try importing torch + torchvision; otherwise provide dummy behavior
try:
    import torch
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Try to import transformers pipeline
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------------------
# Image model loader (ResNet18 feature + dummy classifier mapping)
# ---------------------
if TORCH_AVAILABLE:
    # CPU-only model is fine for POC
    image_model = models.resnet18(pretrained=True)
    image_model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
else:
    image_model = None
    transform = None

DUMMY_CLASSES = ['HFMD', 'Chickenpox', 'Allergic Rash', 'Measles', 'Eczema']

def classify_rash_pil(img_pil):
    """
    Input: PIL.Image
    Returns: (label, confidence float 0-1, debug_info)
    """
    if TORCH_AVAILABLE and image_model is not None:
        try:
            img = img_pil.convert("RGB")
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                feats = image_model(img_t)
            # POC mapping: use the max-index of features as deterministic pseudo-prediction
            scores = feats.squeeze().abs().mean().item()
            # deterministic-ish mapping to classes via sum of first row (not real)
            idx = int((feats.detach().cpu().numpy().sum() % len(DUMMY_CLASSES)))
            label = DUMMY_CLASSES[idx]
            confidence = min(max(0.6 + (scores % 0.4), 0.01), 0.99)
            debug = {"feats_sum": float(feats.detach().cpu().numpy().sum())}
            return label, float(confidence), debug
        except Exception as e:
            # fallback to dummy
            return _dummy_classify(img_pil, reason=f"torch_error: {e}")
    else:
        return _dummy_classify(img_pil, reason="torch_missing")

def _dummy_classify(img_pil, reason="none"):
    label = random.choice(DUMMY_CLASSES)
    confidence = round(random.uniform(0.65, 0.92), 2)
    debug = {"fallback": True, "reason": reason}
    return label, confidence, debug

# ---------------------
# Text extractor / symptom parser
# ---------------------
if TRANSFORMERS_AVAILABLE:
    try:
        nlp_feat = pipeline("feature-extraction", model="distilbert-base-uncased")
    except Exception:
        nlp_feat = None
else:
    nlp_feat = None

def extract_symptoms(text):
    """
    Very simple extractor:
      - if transformers available, return features (not used directly)
      - else, do small rule-based extraction and return normalized text + symptom flags
    """
    text_l = text.strip().lower()
    flags = {
        "fever": "fever" in text_l,
        "breathing_difficulty": any(x in text_l for x in ["breath", "wheeze", "difficulty breathing", "choking"]),
        "spots_hands_feet": any(x in text_l for x in ["hands", "feet", "palms", "soles"]),
        "blisters": "blister" in text_l or "vesicle" in text_l,
        "itching": "itch" in text_l,
    }
    features = {"text": text, "lower": text_l, "flags": flags}
    # optionally run transformer embedding (not used for triage here)
    if nlp_feat is not None:
        try:
            _ = nlp_feat(text_l)
            features["transformer_ok"] = True
        except Exception:
            features["transformer_ok"] = False
    return features

# ---------------------
# Red flag logic (simple)
# ---------------------
def check_red_flags(text_features, image_label, image_conf):
    flags = []
    f = text_features.get("flags", {})
    if f.get("breathing_difficulty"):
        flags.append("Difficulty breathing or noisy breathing — call 999 now.")
    if f.get("fever") and image_conf < 0.5:
        # low confidence with fever → escalate to clinician
        flags.append("High fever reported — seek urgent clinical review.")
    # skin-specific flags
    if image_label == "Measles":
        flags.append("Measles is infectious and children should be assessed by a clinician.")
    # generic severe skin sign check via keywords
    if "purple" in text_features["lower"] or "bleeding" in text_features["lower"]:
        flags.append("Purple spots, purpura, or unexplained bleeding — seek immediate medical help.")
    return flags

# ---------------------
# Fuse image + text into summary
# ---------------------
def generate_summary(image_label, image_conf, text_features, red_flags):
    score = float(image_conf) * 0.7 + (0.9 if any(text_features.get("flags", {}).values()) else 0.6) * 0.3
    score = max(0.0, min(1.0, score))
    advice = [
        "Most rashes in children are mild and self-limiting.",
        "Provide symptomatic care: fluids, paracetamol for fever (follow dosing guidance).",
        "If worried, contact your GP or NHS 111 for advice."
    ]
    if red_flags:
        advice = red_flags + advice

    summary = {
        "provisional_diagnosis": image_label,
        "confidence": round(score, 2),
        "symptom_summary": text_features.get("text", ""),
        "advice": advice,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    return summary

# ---------------------
# Simple logger to CSV via BytesIO for download
# ---------------------
def make_log_csv_row(summary, raw_text, image_label, image_conf, debug):
    df = pd.DataFrame([{
        "timestamp": summary["timestamp"],
        "diagnosis": summary["provisional_diagnosis"],
        "confidence": summary["confidence"],
        "symptoms": raw_text,
        "image_label_raw": image_label,
        "image_conf_raw": image_conf,
        "debug": json.dumps(debug)
    }])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------------------
# Load NHS conditions example (if provided)
# ---------------------
def load_nhs_conditions(path="nhs_conditions_example.json"):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}
