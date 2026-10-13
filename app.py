import os, re, json, hashlib, time
from pathlib import Path
import numpy as np
import streamlit as st
import requests
from pypdf import PdfReader

APP_DIR = Path(__file__).parent
DOCS_DIR = APP_DIR / "documents"
INDEX_DIR = APP_DIR / "index"
DOCS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Official Government of India PMAY-U 2.0 guideline sources.
OFFICIAL_GUIDELINES = {
    "PMAY-U-2.0-Guidelines-English.pdf": "https://pmay-urban.gov.in/uploads/guidelines/Operational-Guidelines-of-PMAY-U-2.pdf",
    "PMAY-U-2.0-Guidelines-Hindi.pdf": "https://pmay-urban.gov.in/uploads/guidelines/Operational-Guidelines-of-PMAY-U-2-Hindi.pdf",
}

def seed_official_guidelines():
    # Fetch the official PDFs on first run; keep user-uploaded files untouched.
    status = []
    for filename, url in OFFICIAL_GUIDELINES.items():
        target = DOCS_DIR / filename
        if target.exists() and target.stat().st_size > 10_000:
            continue
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "SchemeSaathi research prototype"})
            response.raise_for_status()
            if response.content[:4] == b"%PDF" and len(response.content) > 10_000:
                target.write_bytes(response.content)
                status.append(f"Downloaded {filename}")
            else:
                status.append(f"Could not verify PDF download for {filename}; use the official link below.")
        except Exception:
            status.append(f"Automatic download unavailable for {filename}; use the official link below.")
    return status

SEED_STATUS = seed_official_guidelines()

st.set_page_config(page_title="SchemeSaathi — Hindi/Hinglish RAG", page_icon="🇮🇳", layout="wide")

st.title("🇮🇳 SchemeSaathi")
st.caption("Hindi • Hinglish • English | Government-scheme document assistant")
st.warning("Research prototype — not an official government service. Answers are based on retrieved guideline text and may be incomplete. Verify eligibility, benefits, deadlines and application decisions with the relevant authority.")

with st.expander("Official PMAY-U 2.0 source documents", expanded=False):
    st.markdown("The app attempts to fetch these official documents automatically on first launch. If hosting blocks downloads, download them from the Ministry website and upload them using the sidebar.")
    st.markdown("- [PMAY-U 2.0 Scheme Guidelines — Hindi (official PDF)](https://pmay-urban.gov.in/uploads/guidelines/Operational-Guidelines-of-PMAY-U-2-Hindi.pdf)")
    st.markdown("- [PMAY-U 2.0 Scheme Guidelines — English (official PDF)](https://pmay-urban.gov.in/uploads/guidelines/Operational-Guidelines-of-PMAY-U-2.pdf)")
    st.markdown("- [Ministry's official guidelines listing](https://pmay-urban.gov.in/guideline)")
    for msg in SEED_STATUS:
        st.caption(msg)

with st.sidebar:
    st.header("Knowledge base")
    st.write("Official PMAY-U 2.0 PDFs are auto-loaded when hosting allows. You can add more official scheme PDFs below. The app retrieves evidence passages and shows source pages.")
    uploads = st.file_uploader("Upload official PDF guidelines", type=["pdf"], accept_multiple_files=True)
    if uploads:
        for uploaded in uploads:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded.name)
            path = DOCS_DIR / safe_name
            path.write_bytes(uploaded.getvalue())
        st.success(f"Saved {len(uploads)} PDF file(s).")
    st.caption(f"PDFs in knowledge base: {len(list(DOCS_DIR.glob('*.pdf')))}")
    if not list(DOCS_DIR.glob('*.pdf')):
        st.info("No guideline PDF is available yet. Use the official links above, download the PDF, then upload it here.")

    st.divider()
    st.subheader("Retrieval settings")
    chunk_size = st.slider("Chunk size (characters)", 500, 1800, 1000, 100)
    overlap = st.slider("Chunk overlap", 0, 250, 120, 10)
    top_k = st.slider("Passages to retrieve", 1, 8, 4)
    use_dense = st.checkbox("Use multilingual semantic retrieval", value=True,
                             help="First run downloads a multilingual embedding model and may take several minutes. Disable to use keyword retrieval only.")
    st.caption("Semantic model: paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource(show_spinner="Loading retrieval model…")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_chunks(pdf_path, size, overlap_chars):
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        return [], [f"Could not open {pdf_path.name}: {e}"]
    chunks, warnings = [], []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            warnings.append(f"{pdf_path.name}, page {page_no}: no selectable text (possibly scanned).")
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            piece = text[start:end].strip()
            if len(piece) >= 40:
                chunks.append({
                    "text": piece,
                    "source": pdf_path.name,
                    "page": page_no,
                    "chunk_id": len(chunks)
                })
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
    return chunks, warnings

def build_corpus(size, overlap_chars):
    all_chunks, warnings = [], []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        chunks, warns = extract_chunks(pdf, size, overlap_chars)
        all_chunks.extend(chunks)
        warnings.extend(warns)
    return all_chunks, warnings

def lexical_score(query, text):
    # Lightweight fallback: character/token overlap with a small phrase bonus.
    def tokens(s):
        return set(re.findall(r"[\w\u0900-\u097F]+", s.lower()))
    q = tokens(query)
    d = tokens(text)
    if not q or not d:
        return 0.0
    overlap_count = len(q & d)
    score = overlap_count / max(1, len(q)) + 0.15 * (overlap_count / max(1, len(d)) * 10)
    if query.lower() in text.lower():
        score += 0.5
    return float(score)

@st.cache_data(show_spinner=False)
def _embed_texts(texts):
    model = load_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def retrieve(query, chunks, k, dense):
    if not chunks:
        return []
    if dense:
        model = load_model()
        texts = [c["text"] for c in chunks]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        qvec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(np.asarray(vecs), np.asarray(qvec))
    else:
        scores = np.array([lexical_score(query, c["text"]) for c in chunks])
    ranked = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked if float(scores[i]) > 0]

def make_answer(question, results):
    if not results:
        return ("Mujhe uploaded documents mein is sawaal ka relevant evidence nahi mila. "
                "Main guess nahi karunga. Kripya official guideline upload/check karein ya sawaal ko thoda specific banayein.")
    # Extractive, evidence-constrained response: avoids inventing facts without an LLM.
    lead = "Uploaded official document(s) se sabse relevant passages ye hain. Inke basis par hi answer verify karein:\n\n"
    body = []
    for i, r in enumerate(results, 1):
        body.append(f"**Evidence {i} — {r['source']}, page {r['page']}**\n\n> {r['text']}")
    tail = ("\n\n**Note:** Yeh prototype abhi retrieved passages dikha raha hai, independently verified eligibility decision nahi de raha. "
            "Agar passages mein direct answer nahi hai, ise confirmed answer na samjhein.")
    return lead + "\n\n".join(body) + tail

with st.expander("How to use", expanded=True):
    st.markdown("1. Sidebar se official scheme PDF upload karein.\n2. Neeche apna sawaal Hindi, Roman Hindi/Hinglish, ya English mein likhein.\n3. Retrieve par click karein.\n4. Answer ke saath source filename aur page number check karein. Agar evidence nahi milta, chatbot abstain karega.")

language = st.selectbox("Response language / जवाब की भाषा", ["Match my question (Hindi/Hinglish/English)", "Hindi (हिन्दी)", "Hinglish (Roman Hindi)", "English"])
question = st.text_area("Apna sawaal likho / अपना सवाल लिखें / Ask a question", placeholder="Example: PMAY-U 2.0 ke liye kaun eligible hai? / इस योजना के लिए कौन पात्र है? / Who is eligible?", height=90)
col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("🔎 Retrieve evidence", type="primary", use_container_width=True)
with col2:
    st.caption("Answers are grounded in uploaded PDFs. No claim is made that this prototype is an official service.")

if run:
    if not question.strip():
        st.info("Pehle sawaal likho.")
    else:
        with st.spinner("Documents read karke relevant evidence search kar raha hoon…"):
            chunks, warnings = build_corpus(chunk_size, overlap)
            if warnings:
                with st.expander(f"Document extraction notes ({len(warnings)})"):
                    for warning in warnings[:30]:
                        st.write("• " + warning)
            if not chunks:
                st.error("Abhi searchable text nahi mila. Sidebar mein official PDF upload karo. Scanned PDF ho to OCR ki zaroorat ho sakti hai.")
            else:
                try:
                    hits = retrieve(question, chunks, top_k, use_dense)
                except Exception as e:
                    st.error(f"Retrieval model error: {e}. Semantic retrieval off karke dobara try karo.")
                    hits = retrieve(question, chunks, top_k, False)
                st.subheader("Retrieved answer / Source-grounded response")
                if language == "Hindi (हिन्दी)":
                    st.caption("नोट: नीचे दिए गए अंश आधिकारिक दस्तावेज़ों से खोजे गए हैं। पात्रता या लाभ का अंतिम निर्णय संबंधित सरकारी प्राधिकरण से सत्यापित करें।")
                elif language == "Hinglish (Roman Hindi)":
                    st.caption("Note: Neeche official documents se relevant passages diye gaye hain. Eligibility/benefit ka final decision authority se verify karein.")
                elif language == "English":
                    st.caption("The passages below were retrieved from the available documents. Verify final eligibility and benefits with the relevant authority.")
                st.markdown(make_answer(question, hits))
                st.subheader("Retrieval diagnostics")
                st.write(f"Extracted passages: {len(chunks)} · Returned passages: {len(hits)} · Retrieval: {'multilingual semantic' if use_dense else 'keyword'}")
                if hits:
                    st.dataframe([{"Rank": i+1, "Source": h["source"], "Page": h["page"], "Score": round(h["score"], 4), "Passage preview": h["text"][:240]} for i,h in enumerate(hits)], use_container_width=True, hide_index=True)
                else:
                    st.warning("Koi relevant passage retrieve nahi hua. Isse answer unknown maana jayega, not 'no'.")

st.divider()
st.caption("Research note: this is an initial extractive RAG prototype. Before reporting research results, freeze a benchmark, manually verify retrieved passages and answers, and report measured metrics rather than assumed accuracy.")
