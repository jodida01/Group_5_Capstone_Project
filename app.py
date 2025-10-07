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

# Custom CSS
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 10px;
    }
    .process-flow {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 25px;
        margin-top: 30px;
    }
    .flow-step {
        background: #2a5298;
        color: white;
        padding: 15px 25px;
        border-radius: 25px;
        margin: 10px;
        font-weight: 500;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Functions ---
@st.cache_data(hash_funcs={st.runtime.uploaded_file_manager.UploadedFile: lambda x: x.getvalue()})
def load_and_clean_docx(file_path):
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00A0", " ").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n\n".join(lines)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return ""

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

@st.cache_resource
def build_faiss_index(chunks, model_path="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_path)
        embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return model, index, chunks
    except Exception as e:
        st.error(f"Error building FAISS index: {str(e)}")
        return None, None, chunks

def search(query, model, index, chunks, k=1):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, k)
    results = [(chunks[i], dist) for i, dist in zip(indices[0], distances[0])]
    return results

def check_crypto_activity(activity):
    return 'crypto' in activity.lower()

def calculate_risk_score(chunk_text, customer_details):
    risk_indicators = ['high-risk', 'pep', 'sanctions', 'edd', 'politically exposed']
    high_risk_types = ['Business (High Risk)', 'Foreign Student', 'Trust', 'Forex Bureaus', 'MTOs', 'Betting Firms', 'NGOs', 'PSPs']
    score = 0
    if any(indicator in chunk_text.lower() for indicator in risk_indicators):
        score += 3
    if customer_details.get('nationality') in ['Foreign', 'High-risk jurisdiction']:
        score += 3
    if customer_details.get('customer_type') in high_risk_types:
        score += 3
    if customer_details.get('is_pep'):
        score += 3
    if check_crypto_activity(customer_details.get('expected_activity', '')):
        score += 3
    risk_level = 'Low' if score < 3 else 'Medium' if score < 5 else 'High'
    return score, risk_level

def generate_policy_text(customer_type, nationality, is_pep):
    base_policy = """
    KYC/AML Compliance Policy for Customer Onboarding

    1. Customer Identification Program (CIP)
    All new customers must provide valid identification documents prior to account opening. Acceptable documents include:
    - Government-issued photo ID (e.g., passport, Military Service Card, national ID).
    - For foreign nationals, a valid visa, Work Permit, or residence permit is required.
    - Minors: Parents or guardians must provide National ID, Passport Copy, or Military Service Card (whichever applies) plus the minor's birth certificate or birth notification.

    2. Verification Process
    Identity verification involves cross-checking documents against third-party databases. Enhanced Due Diligence (EDD) is mandatory for high-risk customers, including:
    - Politically Exposed Persons (PEPs) or sanctioned individuals.
    - Customers from high-risk jurisdictions.
    - Accounts with expected high transaction volumes.
    EDD includes verifying the source of funds and purpose of the account.

    4. Anti-Money Laundering (AML) Monitoring
    - Transactions exceeding $10,000 must be reported to regulatory authorities within 5 business days.
    - Suspicious Activity Reports (SARs) are filed for unusual patterns, such as rapid fund transfers or inconsistent transaction purposes.
    - Continuous monitoring is required for accounts flagged as high-risk.

    5. Record Keeping
    All KYC records must be retained for a minimum of 7 years after account closure. Records include identification documents, verification results, and transaction histories.

    6. Training
    Staff must complete annual KYC/AML training to ensure compliance with regulations.
    """
    type_specific_policies = {
        "Individual": """
        3. Account Types and Requirements
        - Individual Accounts: Obtain National ID, Passport Copy, or Military Service Card, physical address, main source of income, and area of operation.
        """,
        "Business (Local)": """
        3. Account Types and Requirements
        - Business Accounts (Local): Must submit business registration documents, beneficial owner details (at least 25% ownership), and recent financial statements.
        """,
        "Business (High Risk)": """
        3. Account Types and Requirements
        - Business Accounts (High Risk, including Foreign Owned): Require enhanced due diligence, source of funds, beneficial owner details (at least 25% ownership), recent financial statements, and regulatory licenses. Mandatory sanctions screening and senior management approval.
        """,
        "Foreign Student": """
        3. Account Types and Requirements
        - Foreign Student Accounts: Require student visa, proof of enrollment, local address verification, and source of funds. Enhanced due diligence is mandatory due to high-risk status.
        """,
        "Trust": """
        3. Account Types and Requirements
        - Trust Accounts: Need trust deeds, identification of trustees and beneficiaries, and source of funds. Enhanced due diligence and sanctions screening are mandatory due to high-risk status.
        """,
        "Forex Bureaus": """
        3. Account Types and Requirements
        - Forex Bureaus: Require business registration, regulatory licenses from central bank, beneficial owner details, source of funds, and transaction monitoring plans. Enhanced due diligence and daily sanctions screening are mandatory due to high-risk status.
        """,
        "MTOs": """
        3. Account Types and Requirements
        - Money Transfer Operators (MTOs): Require business registration, regulatory licenses, beneficial owner details, source of funds, and anti-money laundering program documentation. Enhanced due diligence and continuous transaction monitoring are mandatory due to high-risk status.
        """,
        "Betting Firms": """
        3. Account Types and Requirements
        - Betting Firms: Require business registration, betting control board licenses, beneficial owner details, source of funds, and transaction monitoring systems. Enhanced due diligence and sanctions screening are mandatory due to high-risk status.
        """,
        "NGOs": """
        3. Account Types and Requirements
        - Non-Governmental Organizations (NGOs): Require registration certificates, beneficial owner details, source of funds, and program activity reports. Enhanced due diligence and sanctions screening are mandatory due to high-risk status.
        """,
        "PSPs": """
        3. Account Types and Requirements
        - Payment Service Providers (PSPs): Require business registration, regulatory licenses, beneficial owner details, source of funds, and transaction monitoring systems. Enhanced due diligence and daily sanctions screening are mandatory due to high-risk status.
        """
    }
    additional_policy = ""
    if nationality in ['Foreign', 'High-risk jurisdiction']:
        additional_policy += """
        - Foreign and High-Risk Jurisdiction Customers: Require enhanced due diligence, including source of wealth verification, sanctions screening, and senior management approval.
        """
    if is_pep:
        additional_policy += """
        - Politically Exposed Persons (PEPs) or Sanctioned Individuals: Require enhanced due diligence, source of wealth verification, sanctions screening, and senior management approval.
        """
    if check_crypto_activity(customer_details.get('expected_activity', '')):
        additional_policy += """
        - Crypto Currency Accounts: Prohibited due to high risk of money laundering. Any crypto-related activity requires immediate review and potential account rejection.
        """
    policy_text = base_policy + type_specific_policies.get(customer_type, "") + additional_policy
    return policy_text.strip()

# --- Streamlit App ---
st.title("FinComBot - Compliance Chatbot")
st.markdown("Query KYC/AML policies and get personalized onboarding recommendations.")

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "selected_step" not in st.session_state:
    st.session_state.selected_step = None

# Customer Details Form
st.header("Customer Onboarding Assistant")
customer_type = st.selectbox("Customer Type:", ["Individual", "Business (Local)", "Business (High Risk)", "Foreign Student", "Trust", "Forex Bureaus", "MTOs", "Betting Firms", "NGOs", "PSPs"])
nationality = st.selectbox("Nationality/Origin:", ["Local", "Foreign", "High-risk jurisdiction"])
expected_activity = st.text_input("Expected Transaction Activity (e.g., high volume):")
is_pep = st.checkbox("Is Politically Exposed Person (PEP) or Sanctioned?")
customer_details = {
    "customer_type": customer_type,
    "nationality": nationality,
    "expected_activity": expected_activity,
    "is_pep": is_pep
}

# Check for crypto activity
if check_crypto_activity(expected_activity):
    st.warning("Crypto currency activities are prohibited due to high risk of money laundering.")

# File upload
uploaded_file = st.file_uploader("Upload a KYC/AML .docx file (optional)", type=["docx"])
if uploaded_file:
    temp_file = "temp_doc.docx"
    try:
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Loading and cleaning uploaded document..."):
            cleaned_text = load_and_clean_docx(temp_file)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)  # Clean up temporary file
else:
    cleaned_text = generate_policy_text(customer_type, nationality, is_pep)

# Sidebar: Stats, Word Cloud, Query History
with st.sidebar:
    st.header("Document Stats")
    orig_chars, orig_words = generate_stats(cleaned_text)
    st.write(f"Characters: {orig_chars}")
    st.write(f"Words: {orig_words}")
    
    st.subheader("Word Cloud")
    max_words = st.slider("Max words:", 50, 200, 100)
    stop_words = set(stopwords.words('english'))
    word_freq = Counter(word.lower() for word in re.findall(r'\w+', cleaned_text) if word.lower() not in stop_words)
    wc = WordCloud(width=400, height=200, max_words=max_words).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
    st.subheader("Recent Queries")
    for i, past_query in enumerate(st.session_state.query_history[-5:][::-1]):
        if st.button(past_query, key=f"history_{i}"):
            st.session_state.query_input = past_query
            st.experimental_rerun()

# Chunk and index
with st.spinner("Building search index..."):
    chunks = chunk_text(cleaned_text)
    model, index, _ = build_faiss_index(chunks)
    if model is None or index is None:
        st.error("Failed to build search index. Please try again.")
        st.stop()

# Query input
query = st.text_input("Enter your KYC/AML query (e.g., 'Requirements for this customer'):", key="query_input")

# Advanced query construction
if query:
    advanced_query = f"{query} for {customer_type} customer from {nationality}"
    if is_pep:
        advanced_query += " who is PEP or sanctioned"
    if expected_activity:
        advanced_query += f" with {expected_activity}"

# Search settings
score_threshold = st.slider("Minimum similarity score:", 0.0, 1.0, 0.0, 0.05)

# Process query
if query:
    try:
        if not query.strip():
            st.warning("Please enter a valid query.")
            st.stop()
        if query.strip() and query not in st.session_state.query_history:
            st.session_state.query_history.append(query.strip())
        
        results = search(advanced_query, model, index, chunks, k=1)
        if not results:
            st.warning("No results found for the query.")
            st.stop()
        
        filtered_results = [(chunk, score) for chunk, score in results if score >= score_threshold]
        
        if filtered_results:
            chunk, score = filtered_results[0]
            risk_score, risk_level = calculate_risk_score(chunk, customer_details)
            
            st.header("KYC Recommendation")
            st.metric("Risk Level", risk_level)
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(["Similarity Score", "Risk Score"], [score, risk_score], color=['#4CAF50', '#FF5722'])
            ax.set_ylabel("Score")
            ax.set_title("Analysis Scores")
            plt.ylim(0, max(score, risk_score) * 1.1)
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander(f"Policy Excerpt (Similarity: {score:.3f})"):
                st.text_area("", chunk, height=150, key="excerpt_1")
            
            st.subheader("Advanced Actions")
            st.write("- Perform EDD if risk is High.")
            st.write("- Screen against sanctions lists.")
            st.write("- Verify source of funds.")
            if risk_level == "High":
                st.warning("Recommend manual review and additional verification.")
        else:
            st.warning(f"No relevant policy found meeting the similarity threshold ({score_threshold}).")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}. Please try again or check the uploaded file.")
else:
    st.info("Enter a query and customer details for personalized KYC advice.")

# KYC/AML Process Flow
st.markdown('<div class="process-flow">', unsafe_allow_html=True)
st.header("KYC/AML Process Flow")
process_steps = {
    "identity": {
        "title": "Customer Identity Verification",
        "content": [
            "**Required Documents & Information:**",
            "- Individual Customers: Valid government-issued photo ID, proof of address, Social Security Number",
            "- Corporate Customers: Articles of incorporation, business license, beneficial owner information",
            "- Foreign Nationals: Passport, visa documentation, local address verification",
            "- High-Risk Customers: Enhanced documentation, source of funds verification",
            "**Regulatory Requirements:** Customer identification must comply with CBK guidelines and FATF recommendations for proper customer due diligence.",
            "**Verification Process:** All identity documents must be verified through approved verification services or manual review by trained staff. Digital copies must be clear and legible."
        ]
    },
    "risk": {
        "title": "Risk Assessment",
        "content": [
            "**Risk Categories:**",
            "- Low Risk: Domestic customers with stable employment and clear source of funds",
            "- Medium Risk: New businesses, high-cash-transaction customers, foreign nationals",
            "- High Risk: Politically Exposed Persons (PEPs), customers from high-risk jurisdictions, complex ownership structures",
            "**Critical Alert:** All high-risk customers require additional approval from compliance officers and enhanced monitoring procedures.",
            "**Assessment Factors:** Geographic risk, customer type, product/service risk, delivery channel risk, and transaction patterns are evaluated using the bank's risk assessment matrix.",
            "**Required Actions:** Risk ratings must be reviewed annually or when customer circumstances change significantly."
        ]
    },
    "edd": {
        "title": "Enhanced Due Diligence (EDD)",
        "content": [
            "**When EDD is Required:**",
            "- High-Risk Customers: PEPs, customers from FATF non-cooperative countries",
            "- Unusual Transactions: Large cash deposits, frequent international transfers",
            "- Complex Structures: Trusts, shell companies, correspondent banking relationships",
            "- Regulatory Triggers: Sanctions screening hits, adverse media coverage",
            "**EDD Requirements:** Enhanced customer information, source of wealth verification, increased transaction monitoring, and senior management approval.",
            "**Documentation Standards:** All EDD findings must be thoroughly documented with supporting evidence and regular reviews scheduled based on risk level.",
            "**Approval Process:** EDD cases require approval from designated compliance officers and may need additional sign-off from senior management for highest-risk categories."
        ]
    },
    "monitoring": {
        "title": "Ongoing Monitoring",
        "content": [
            "**Continuous Monitoring Requirements:**",
            "- Transaction Monitoring: Automated systems screening for unusual patterns and suspicious activities",
            "- Account Reviews: Regular review of customer profiles and risk ratings",
            "- Sanctions Screening: Daily screening against updated sanctions lists",
            "- PEP Monitoring: Ongoing screening for political exposure changes",
            "**Alert Management:** All system-generated alerts must be investigated within specified timeframes and properly documented.",
            "**Review Frequencies:** Low Risk: Annual review; Medium Risk: Semi-annual review; High Risk: Quarterly review or more frequent as required.",
            "**Escalation Procedures:** Suspicious activities must be escalated to the compliance team for potential SAR filing within regulatory timeframes."
        ]
    },
    "reporting": {
        "title": "Regulatory Reporting",
        "content": [
            "**Required Reports:**",
            "- Suspicious Activity Reports (SARs): Filed within 2 days of detection",
            "- Currency Transaction Reports (CTRs): For cash transactions exceeding $10,000",
            "- OFAC Reports: Immediate reporting of sanctions violations",
            "- BSA/AML Reports: Annual compliance program assessments",
            "**Filing Requirements:** All reports must be filed through appropriate regulatory channels with complete and accurate information.",
            "**Record Keeping:** All compliance documentation must be retained for minimum 7 years and made available for regulatory examination upon request.",
            "**Quality Control:** All reports undergo internal review before submission to ensure accuracy and completeness. Regular training ensures staff understanding of reporting obligations.",
            "**Confidentiality:** SAR filings and related investigations are strictly confidential and subject to strict access controls."
        ]
    }
}

num_cols = min(len(process_steps), 3)  # Responsive layout
cols = st.columns(num_cols)
for i, (step, details) in enumerate(process_steps.items()):
    with cols[i % num_cols]:
        st.markdown(f'<div class="flow-step">{details["title"]}</div>', unsafe_allow_html=True)
        if st.button("View Details", key=f"step_{step}"):
            st.session_state.selected_step = step

if st.session_state.selected_step:
    step = st.session_state.selected_step
    details = process_steps[step]
    with st.expander(f"{details['title']} Details", expanded=True):
        for item in details["content"]:
            if item.startswith("**"):
                st.markdown(f"**{item.strip('**')}**")
            elif item.startswith("-"):
                st.markdown(f"• {item.lstrip('- ')}")
            else:
                st.markdown(item)
        if "Critical Alert" in " ".join(details["content"]):
            st.warning("All high-risk customers require additional approval and enhanced monitoring.")
st.markdown('</div>', unsafe_allow_html=True)