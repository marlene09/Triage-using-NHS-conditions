# app.py
import streamlit as st
from PIL import Image
from utils import (
    classify_rash_pil,
    extract_symptoms,
    check_red_flags,
    generate_summary,
    make_log_csv_row,
    load_nhs_conditions
)

st.set_page_config(page_title="Pediatric Rash Triage (POC)", layout="centered")

st.title("🩺 Pediatric Rash Triage — Prototype")
st.caption("This is a prototype. All outputs are provisional and **for clinician review only**.")

# sidebar
st.sidebar.header("Settings")
st.sidebar.info("Ensure you activated your venv (python 3.10 recommended).")

# load nhs conditions (optional)
nhs_conditions = load_nhs_conditions()

# upload image + text input
uploaded = st.file_uploader("Upload a rash photo (jpg/png)", type=["jpg", "jpeg", "png"])
symptom_text = st.text_area("Describe the symptoms or ask a question", height=140, placeholder="e.g. red spots on hands and feet, mild fever for 2 days")

col1, col2 = st.columns([1, 1])
with col1:
    if uploaded:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error("Could not open image: " + str(e))
            image = None
    else:
        image = None

with col2:
    st.markdown("### Quick help")
    st.markdown("- Mention duration of symptoms (e.g. '2 days')\n- Mention breathing, feeding, high fever\n- Mention distribution (hands/feet/face/body)")

if st.button("Run Triage"):
    if not uploaded and not symptom_text.strip():
        st.warning("Please provide an image or describe symptoms.")
    else:
        # classify image (or fallback)
        if image is not None:
            label, conf, debug = classify_rash_pil(image)
        else:
            label, conf, debug = ("No Image", 0.0, {"fallback": "no_image"})

        # text features
        text_features = extract_symptoms(symptom_text)

        # red flags
        red_flags = check_red_flags(text_features, label, conf)

        # summary
        summary = generate_summary(label, conf, text_features, red_flags)

        # show results
        st.subheader("Provisional result")
        st.metric("Provisional diagnosis", f"{summary['provisional_diagnosis']} ({int(summary['confidence']*100)}%)")
        st.write("**Symptoms (as entered):**")
        st.write(summary["symptom_summary"])

        if red_flags:
            st.error("⚠️ Red flags detected")
            for rf in red_flags:
                st.write(f"- {rf}")

        st.write("**Advice**")
        for line in summary["advice"]:
            st.write(f"- {line}")

        # debug panel (collapsible)
        with st.expander("Debug info"):
            st.write("raw image label:", label, "raw_conf:", conf)
            st.json(debug)
            st.json(text_features)

        # logger download
        csv_buf = make_log_csv_row(summary, symptom_text, label, conf, debug)
        st.download_button("Download triage log (CSV)", data=csv_buf, file_name="triage_log.csv", mime="text/csv")

        # optional: show related NHS conditions snippet if available
        if nhs_conditions:
            st.markdown("---")
            st.subheader("Related NHS condition info (example)")
            # naive match: show first condition containing the predicted label (case-insensitive)
            matches = [c for c in nhs_conditions.get("conditions", []) if summary["provisional_diagnosis"].lower() in c.get("name","").lower()]
            if matches:
                for m in matches:
                    st.write(f"**{m.get('name')}** — {m.get('summary','')}")
            else:
                st.info("No local NHS condition matched the provisional diagnosis.")

st.markdown("---")
st.caption("Prototype — not a clinical decision support system. Always consult a qualified clinician for diagnosis and urgent care.")
