# FinComBot -- Compliance Chatbot

![AI Chatbot](Images/HeaderImage.jpg)
## 1. Background

Financial institutions face increasing pressure to comply with stringent regulatory frameworks governing customer onboarding, Know Your Customer (KYC), Customer Due Diligence (CDD), Enhanced Due Diligence (EDD), Anti-Money Laundering (AML), Counter Terrorism Financing, Counter Proliferation Financing (CPF), and sanctions screening. These obligations are complex, continuously evolving, and vary across jurisdictions.

Staff often face difficulties accessing and interpreting regulatory documents and internal policies, leading to:
-	Delays in onboarding, affecting customer experience and revenue.
-	Inconsistent application of compliance procedures.
-	Overdependence on compliance officers for basic guidance.
-	Increased risk of regulatory breaches which may lead to fining by regulators and put the bank at risk of its license being suspended.

## 2. Project Objective

-  Build a chatbot that retrieves accurate compliance information from the bank’s KYC policy and responds to staff queries.

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

## 6. Conclusion

## 7. Repository Navigation

-   **Framework**: Implemented in Jupyter Notebook.\
-   **Core Libraries**: Likely includes Python packages for NLP,
    embeddings, and chatbot interaction.\
-   **Model**: Leverages language models for natural language
    understanding.\
-   **Deployment**: Can be integrated into internal compliance systems
    or chat platforms.

## 5. Getting Started

### Prerequisites

-   Python 3.8+\
-   Jupyter Notebook\
-   Required libraries (see `requirements.txt` if available).

### Installation

``` bash
git clone <repo-url>
cd fincombot
pip install -r requirements.txt
```

### Running the Notebook

``` bash
jupyter notebook index.ipynb
```

## 6. Future Work

-   Deploy as a web-based or Slack/Teams chatbot.\
-   Integrate with live regulatory data sources.\
-   Expand to cover multiple jurisdictions.

## 7. License

[MIT License](LICENSE)
