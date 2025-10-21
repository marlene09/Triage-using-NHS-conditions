# Triage-using-NHS-conditions
Current pathway via phone or online, with areas of improvement
<img width="1543" height="833" alt="Screenshot 2025-10-21 at 09 19 51" src="https://github.com/user-attachments/assets/a36bf769-fea8-43d0-a131-d441bbee84b4" />


# NHS Triage Assistant — Proof of Concept

This is a **multimodal prototype** that demonstrates how an NHS-style triage chatbot could combine
✅ **text understanding** (NHS.uk guidance) and  
🖼️ **image understanding** (skin rash photos to start with).

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



Symptomatic relief (hydration, paracetamol, calamine lotion)

Red flags (“Contact 999 if…”)



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
- Uses **CLIP** (open-source) to describe uploaded images.  
- Uses **LlamaIndex** with NHS web pages for evidence-based answers.  
- Uses a **local Ollama model** (`mistral`, `llama3`, etc.) for reasoning — no cloud APIs.  
- Runs fully offline after setup.

---
## Have a play
 paediatric.py - Retrieves and indexes NHS rash information from the web using vector embeddings.
Uses CLIP to analyse an uploaded rash image and estimate probabilities of matching conditions.
Combines text and image insights to generate a structured, clinician-style summary via a local LLM (Mistral).
** install Ollama locally :https://ollama.ai/download

## 🧱 Installation
coming soon

