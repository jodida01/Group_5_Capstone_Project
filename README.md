# FINCOMBOT -- COMPLIANCE CHATBOT

FinComBot is an AI-powered compliance chatbot designed to streamline customer onboarding and strengthen regulatory compliance in financial institutions. It achieves this by providing staff with instant, policy-aligned answers to KYC and AML-related queries. The project will follow a phased implementation approach, beginning with a pilot focused on KYC and onboarding procedures.

![AI Chatbot](Images/HeaderImage.jpg)


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

## 5. Modeling & Evaluation

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
## 6. Conclusion

## 7. Repository Navigation

## 8. License

[MIT License](LICENSE)
