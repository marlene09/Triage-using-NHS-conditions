# Requirements: torch, torchvision, transformers, PIL, pandas
# pip install torch torchvision transformers pillow pandas

from PIL import Image
import torch
from torchvision import transforms, models
from transformers import pipeline

# -----------------------------
# 1. Image Classifier (Open-Source)
# -----------------------------

# Example: use a pre-trained ResNet for feature extraction (replace with pediatric rash model if available)
image_model = models.resnet18(pretrained=True)
image_model.eval()

# Simple transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_rash(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = image_model(img_t)
    # Placeholder: map features to dummy classes (replace with real pediatric rash classifier)
    dummy_classes = ['HFMD', 'Chickenpox', 'Allergic Rash', 'Measles']
    import random
    pred_class = random.choice(dummy_classes)
    confidence = round(random.uniform(0.7, 0.95), 2)
    return pred_class, confidence

# -----------------------------
# 2. Text Extraction (Open-Source NLP)
# -----------------------------
# Example: extract symptom info from text
nlp = pipeline("feature-extraction", model="distilbert-base-uncased")

def extract_text_features(text):
    features = nlp(text)
    # Simple summary: extract key symptoms (dummy example)
    key_symptoms = text.lower()  # for POC, just lowercase text
    return key_symptoms

# -----------------------------
# 3. Multi-Modal Fusion
# -----------------------------
def generate_provisional_summary(image_path, text):
    img_class, img_conf = classify_rash(image_path)
    text_features = extract_text_features(text)

    # Simple weighting: 70% image, 30% text
    final_score = 0.7 * img_conf + 0.3 * 0.9  # 0.9 is dummy text confidence
    summary = f"""
    Provisional Diagnosis: {img_class} (confidence: {final_score:.2f})
    Physical Characteristics: {text_features}
    Guidance for Patient (for clinician review):
      - Most cases are mild; monitor symptoms.
      - Symptomatic relief: hydration, paracetamol for fever.
      - Red flags: contact 999 if severe rash, difficulty breathing, or persistent high fever.
    """
    return summary

# -----------------------------
# 4. Example Usage
# -----------------------------
image_path = "/Users/marlenepostop/Documents/NHS Conditions/image.png"
patient_text = "Child has red spots around hands and feet, mild fever."

summary = generate_provisional_summary(image_path, patient_text)
print(summary)
