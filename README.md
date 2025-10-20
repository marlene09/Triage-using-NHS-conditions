# Triage-using-NHS-conditions

# NHS Triage Assistant — Open Source Proof of Concept

This is a **multimodal prototype** that demonstrates how an NHS-style triage chatbot could combine
✅ **text understanding** (NHS.uk guidance) and  
🖼️ **image understanding** (skin rash photos).

> ⚠️ For demonstration only — not medical advice.


## Problem solving
?. combining/weigthing text and image input --> describe physical characteristics and create summary with provisional diagnosis--> clinician approves or disapproves ---> message back to patient. During wait, the chat bot provides guidelines such as if conditions worsens contact 999, or in the meantime symptomatic relief such as .... 

## Workflow

***Pediatric Rash Triage POC Workflow***

**User Input**

Upload a photo of the rash.

Optionally, enter textual info: age, symptoms, duration, location.

**Multi-Modal Analysis (Image + Text)**

Image → Medical rash classifier identifies patterns (HFMD, chickenpox, allergic rash, etc.).

Text → NLP module extracts context (fever, itching, recent exposures).

Combined → Weighted reasoning: LLM fuses image + text to describe physical characteristics and propose a provisional diagnosis.

Example output: “Child has red vesicular spots around hands, feet, and mouth, consistent with early-stage HFMD.”

**Provisional Guidance Generation**

LLM produces a patient-friendly summary:

Provisional diagnosis

Typical course

Symptomatic relief (hydration, paracetamol, calamine lotion)

Red flags (“Contact 999 if…”)

*This is not sent to the patient yet*

**Human-in-the-Loop Review**

Clinician receives image + text summary + provisional diagnosis.

Approves, edits, or disapproves.

**Patient Communication**

If approved, the final message is sent to the patient.

Includes:

Diagnosis summary

Symptomatic care instructions

Urgent action guidance if condition worsens
---

## ✳️ Features
- Uses **BLIP** (open-source) to describe uploaded images.  
- Uses **LlamaIndex** with NHS web pages for evidence-based answers.  
- Uses a **local Ollama model** (`mistral`, `llama3`, etc.) for reasoning — no cloud APIs.  
- Runs fully offline after setup.

---

## 🧱 Installation

1. **Install dependencies**
   ```bash
   git clone https://github.com/<yourname>/nhs-triage-bot
   
   pip install -r requirements.txt
