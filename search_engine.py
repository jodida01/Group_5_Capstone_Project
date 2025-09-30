"""
Setup script to create embeddings and FAISS index for Streamlit deployment
Run this BEFORE deploying to Streamlit Cloud
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents = []
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path):
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                logger.error(f"Error reading TXT {file_path}: {e}")
                return ""
    
    def split_into_chunks(self, text, max_length=500):
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def process_documents(self, folder_path):
        """Process all documents in folder"""
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return False
        
        supported_extensions = {'.pdf', '.docx', '.txt'}
        files_found = []
        
        # Find all supported files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    files_found.append(os.path.join(root, file))
        
        if not files_found:
            logger.error(f"No documents found in {folder_path}")
            logger.info(f"Supported formats: {supported_extensions}")
            return False
        
        logger.info(f"Found {len(files_found)} documents")
        
        # Process each file
        for file_path in files_found:
            filename = os.path.basename(file_path)
            logger.info(f"Processing: {filename}")
            
            # Extract text based on extension
            if file_path.lower().endswith('.pdf'):
                content = self.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                content = self.extract_text_from_docx(file_path)
            elif file_path.lower().endswith('.txt'):
                content = self.extract_text_from_txt(file_path)
            else:
                continue
            
            if content.strip():
                # Split into chunks
                chunks = self.split_into_chunks(content, max_length=500)
                logger.info(f"  - Created {len(chunks)} chunks from {filename}")
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        self.documents.append({
                            'filename': filename,
                            'chunk': chunk,
                            'chunk_id': i + 1,
                            'total_chunks': len(chunks)
                        })
            else:
                logger.warning(f"  - No content extracted from {filename}")
        
        if not self.documents:
            logger.error("No documents were successfully processed")
            return False
        
        logger.info(f"Total chunks created: {len(self.documents)}")
        return True
    
    def create_embeddings_and_index(self, output_folder='Data'):
        """Create embeddings and FAISS index"""
        if not self.documents:
            logger.error("No documents to process")
            return False
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Create embeddings
        logger.info("Creating embeddings...")
        chunks_text = [doc['chunk'] for doc in self.documents]
        embeddings = self.model.encode(chunks_text, show_progress_bar=True)
        
        # Prepare data for pickle
        embeddings_data = []
        for i, doc in enumerate(self.documents):
            embeddings_data.append({
                'chunk': doc['chunk'],
                'embedding': embeddings[i],
                'filename': doc['filename'],
                'chunk_id': doc['chunk_id']
            })
        
        # Save embeddings
        embeddings_file = os.path.join(output_folder, 'SEC5_embeddings.pkl')
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        logger.info(f"✓ Saved embeddings to {embeddings_file}")
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        embeddings_array = np.array(embeddings, dtype='float32')
        dimension = embeddings_array.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save FAISS index
        index_file = os.path.join(output_folder, 'SEC5_faiss.index')
        faiss.write_index(index, index_file)
        logger.info(f"✓ Saved FAISS index to {index_file}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SETUP COMPLETE!")
        logger.info("="*50)
        logger.info(f"Documents processed: {len(set(d['filename'] for d in self.documents))}")
        logger.info(f"Total chunks: {len(self.documents)}")
        logger.info(f"Embedding dimension: {dimension}")
        logger.info(f"\nFiles created:")
        logger.info(f"  - {embeddings_file}")
        logger.info(f"  - {index_file}")
        logger.info("\nYou can now deploy to Streamlit!")
        logger.info("="*50)
        
        return True

def main():
    # Configuration
    DOCUMENTS_FOLDER = 'documents'  # Change this to your documents folder
    OUTPUT_FOLDER = 'Data'
    
    logger.info("="*50)
    logger.info("Document Processing Setup")
    logger.info("="*50)
    
    # Check if documents folder exists
    if not os.path.exists(DOCUMENTS_FOLDER):
        logger.error(f"\n❌ ERROR: Documents folder '{DOCUMENTS_FOLDER}' not found!")
        logger.info("\nPlease:")
        logger.info("1. Create a folder with your documents (PDFs, DOCX, or TXT files)")
        logger.info("2. Update DOCUMENTS_FOLDER variable in this script")
        logger.info("3. Run this script again")
        return
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process documents
    if not processor.process_documents(DOCUMENTS_FOLDER):
        logger.error("\n❌ Failed to process documents")
        return
    
    # Create embeddings and index
    if not processor.create_embeddings_and_index(OUTPUT_FOLDER):
        logger.error("\n❌ Failed to create embeddings and index")
        return
    
    logger.info("\n✓ Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Commit the Data folder to your repository:")
    logger.info("   git add Data/")
    logger.info("   git commit -m 'Add embeddings and FAISS index'")
    logger.info("   git push")
    logger.info("2. Deploy your Streamlit app")

if __name__ == "__main__":
    main()