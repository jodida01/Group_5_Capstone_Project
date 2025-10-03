# FinComBot - Compliance Chatbot

## Project Overview

FinComBot is an AI-powered compliance chatbot designed to help bank staff quickly access information from KYC/AML/CFT/CPF policy documents. The system uses semantic search with sentence embeddings to retrieve relevant compliance information.

![AI Chatbot](Images/HeaderImage.jpg)

## Team Members

1. Agnes Chomba
2. Derrick Malinga
3. Erick Okacha
4. Judah Odida
5. Lucas Ominde
6. Nick Mwai
7. Olgah Omollo

## 1. Business Understanding

Financial institutions face increasing pressure to comply with stringent regulatory frameworks governing customer onboarding, Know Your Customer (KYC), Customer Due Diligence (CDD), Enhanced Due Diligence (EDD), Anti-Money Laundering (AML), Counter Terrorism Financing, Counter Proliferation Financing (CPF), and sanctions screening. These obligations are complex, continuously evolving, and vary across jurisdictions.

Staff often face difficulties accessing and interpreting regulatory documents and internal policies, leading to:
-	Delays in onboarding, affecting customer experience and revenue.
-	Inconsistent application of compliance procedures.
-	Overdependence on compliance officers for basic guidance.
-	Increased risk of regulatory breaches which may lead to fining by regulators and put the bank at risk of its license being suspended.

This creates a clear need for a real-time compliance chatbot that can provide instant, accurate answers to compliance-related queries, improve consistency, and reduce operational risk

## 2. Business Objective

 Build a chatbot that retrieves accurate compliance information from the bank’s KYC policy and responds to staff queries.

## 3. Target Audience

a.) Front office / Relationship Managers (who onboard customers)

b.)  Operations staff (who process documents)

c.) Compliance officers (for guidance validation)

d.) New staff (as a training tool)

e.) Risk & Audit teams (for oversight)

## 4. Data Understanding 

Data Source: 
a. Internal compliance policy, stored in Word (.docx) format,  Contains: KYC procedures, AML red flags, CDD/EDD checklists, risk rating methodology, regulatory guidelines (FATF, CBK, CMA)

Data Characteristics:Unstructured text (paragraphs, checklists), Multiple sections (policies, procedures, workflows), Needs preprocessing before AI ingestion. 

Data Protection: Given the sensitive nature of the data used in this project, it has been excluded from version control by adding it to .gitignore to maintain security and confidentiality.

## 5. Installation

Prerequisites
- Python 3.8 or higher
- pip package manager

## 6. Setup Steps

a.)  Clone the repository
   
   git clone https://github.com/jodida01/Group_5_Capstone_Project.git cd Group_5_Capstone_Project

b.) Create virtual environment (recommended)
- python -m venv venv

Windows- venv\Scripts\activate

Mac/Linux- source venv/bin/activate

## 7. Installation Dependancies
   pip install -r requirements.txt

## 8. Verify data files
   
Ensure the following files exist:

- Data/SEC5_embeddings.pkl
- Model will be downloaded automatically on first run

5. ## Running the Application
   
Local Development

streamlit run app.py

The application will open in your browser at http://localhost:8501

## Production Deployment

Option 1: Streamlit Cloud

Push code to GitHub

Connect repository to Streamlit Cloud

Deploy with one click

Option 2: Docker (Coming Soon)

docker build -t fincombot .

docker run -p 8501:8501 fincombot

## Usage Guide

For End Users

Access the application through your browser
- Enter your question in the search box (e.g., "What documents are needed for church account opening?")
- Review results ranked by relevance
- Download results for reference if needed

Sample Queries

- "What documents are needed for church account opening?"
- "How to open a minor's account?"
- "What are the AML red flags?"
- "CDD requirements for foreign nationals"
- "Enhanced due diligence procedures"

## Technical Details

Architecture

- Frontend: Streamlit
- Search Engine: FAISS with cosine similarity
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Document Processing: python-docx

## Performance Metrics

- Total Chunks: 109 balanced text segments
- Embedding Dimension: 384
- Average Query Time: < 1 second
- Index Type: FAISS IndexFlatIP (Inner Product)

## Evaluation Results

- Precision@5: 0.33
- Recall@5: 0.56
- MRR: 0.33

## Data Security

⚠️ Important: The compliance documents contain sensitive information:

- Documents are excluded from version control (added to .gitignore)
- Embeddings file is gitignored for security
- For deployment, ensure secure storage and access controls

## Known Limitations

- Limited to Account Opening Policy: Currently indexes only SEC5 document
- No Real-time Updates: Embeddings must be regenerated when documents change
- English Only: No multi-language support
- Context Window: Limited to 500-character chunks

## Future Enhancements

1. Short-term (Next Sprint)

 - Add more policy documents (AML manual, credit policies)
 - Implement user feedback mechanism
 - Add query history and analytics
 - Improve chunk boundaries (section-aware splitting)

2. Medium-term

 - Multi-document search across all compliance manuals
 - Fine-tune embeddings on banking domain data
 - Add user authentication and role-based access
 - Integrate with core banking system

3. Long-term

 - Conversational interface with context memory
 - Automatic document updates and re-indexing
 - Multi-language support (Swahili, French)
 - Mobile application

## Troubleshooting

Common Issues

Issue: FileNotFoundError: Data/SEC5_embeddings.pkl

- Solution: Ensure embeddings file is in the Data/ folder
- Run the notebook to regenerate embeddings if needed

Issue: Slow loading on first run

Solution: Model downloads on first use (~100MB). Subsequent runs are faster.

Issue: Poor search results

Solution: Try rephrasing your query or using keywords from the document

## Contributing
This is an academic project. For the capstone presentation:

License
This project is for educational purposes as part of the Moringa School Data Science program.

Contact
For questions or issues, contact the team:

Project Lead: Teambers listed above

GitHub: https://github.com/jodida01/Group_5_Capstone_Project

## Modeling & Evaluation

## Modeling approach
- Document Embeddings: Compliance policies are preprocessed, cleaned, and transformed into vector embeddings.
- Embedding Storage: These embeddings are serialized (SEC5_embeddings.pkl) for efficient reuse.
- Similarity Search: A FAISS index is used to match user queries against the embedded policy text, enabling semantic navigation beyond keyword matching.
- Model Goal: Ensure relevant, context-aware answers that align with compliance rules and procedures.

## Evaluation
- Retrieval Testing: Queries are benchmarked against expected policy responses to confirm accuracy.
- Relevance Check: Evaluates whether retrieved excerpts correctly reflect the compliance intent.
- Efficiency: Verifies that FAISS provides scalable, low-latency search over large document sets.
- Validation: Continuous validation with subject-matter experts (compliance officers) to confirm correctness.
***
## Conclusion
FinComBot provides a structured pipeline for cleaning, preprocessing, and analyzing regulatory documents, preparing them for embedding-based retrieval. The chatbot will empower financial institution staff to access compliance guidance instantly, improving efficiency, reducing regulatory risks, and enhancing customer experience.
## Repository Navigation
```plaintext
index.ipynb          # Main notebook containing workflow and experiments
│
├── 1. Project Overview
│   ├── Background, Objectives, and Target Audience
│
├── 2. Data Handling
│   ├── 4.1 Loading Data
│   ├── 4.2 Data Preprocessing
│   │    ├── Cleaning & Normalization
│   │    ├── Exploratory Document Statistics
│   │    └── Text Cleaning Functions
│
├── 3. Exploratory Data Analysis (EDA)
│   ├── 5.1 Visual comparison of original vs cleaned text
│   ├── 5.2 Text chunking (~500 words)
│   └── 5.3 Handling chunk size outliers
│
└── 4. Next Steps (Future Work)
    ├── Embedding generation
    ├── Retrieval-based QA/chatbot integration
    └── Model evaluation & fine-tuning
```



