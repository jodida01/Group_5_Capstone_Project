import streamlit as st
import os
import unicodedata
import re
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Notebook Functions Adapted ---

@st.cache_data  # Cache for efficiency (runs once)
def load_and_clean_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    # Cleaning function from notebook
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    
    # Remove page numbers (simplified)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n\n".join(lines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    return text

@st.cache_data
def generate_stats(text):
    chars = len(text)
    words = len(text.split())
    return chars, words

@st.cache_data
def chunk_text(text, chunk_size=512, overlap=128):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_resource  # Cache model and index (heavy)
def build_faiss_index(chunks, model_path="output/minilm_tuned"):
    model = SentenceTransformer(model_path)
    embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks

def search(query, model, index, chunks, k=5):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k)
    results = [(chunks[i], dist) for i, dist in zip(indices[0], distances[0])]
    return results

# --- Streamlit App ---

st.title("FinComBot: Compliance Chatbot")
st.markdown("Ask questions about KYC/AML policies from the bank's manual. Powered by embeddings and retrieval.")

# Load document (cache it)
doc_path = "Data/SEC5 - OPENING OF ACCOUNTS (004).docx"  # Adjust path as needed
if not os.path.exists(doc_path):
    st.error("Document not found! Place it in Data/ folder.")
else:
    cleaned_text = load_and_clean_docx(doc_path)
    
    # Sidebar: Stats and Visuals
    with st.sidebar:
        st.header("Document Stats")
        orig_chars, orig_words = generate_stats(cleaned_text)  # Notebook uses 'text' as original, but here we use cleaned
        st.write(f"Characters: {orig_chars}")
        st.write(f"Words: {orig_words}")
        
        # Simple word cloud (from notebook EDA)
        stop_words = set(stopwords.words('english'))
        word_freq = Counter(word.lower() for word in re.findall(r'\w+', cleaned_text) if word.lower() not in stop_words)
        wc = WordCloud(width=400, height=200).generate_from_frequencies(word_freq)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # Chunk and build index
    chunks = chunk_text(cleaned_text)
    model, index, _ = build_faiss_index(chunks)  # Uses fine-tuned model

    # Query Input
    query = st.text_input("Enter your compliance query (e.g., 'Requirements for foreign student account'):")
    
    if query:
        results = search(query, model, index, chunks, k=5)
        st.header("Top Relevant Sections")
        for rank, (chunk, score) in enumerate(results, 1):
            st.subheader(f"Rank {rank} (Similarity Score: {score:.3f})")
            st.text_area("Excerpt:", chunk, height=150)

# Run with: streamlit run app.py