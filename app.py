import streamlit as st
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceChatbot:
    def __init__(self):
        self.model = None
        self.chunks = None
        self.index = None
        self.embeddings = None
        self.load_resources()
    
    @st.cache_resource
    def load_resources(_self):
        """Load embeddings, chunks, and FAISS index"""
        try:
            # Check if Data directory exists
            if not os.path.exists("Data"):
                st.error("❌ Data directory not found. Please ensure the 'Data' folder is included in your deployment.")
                st.info("Required files:\n- Data/SEC5_embeddings.pkl\n- Data/SEC5_faiss.index (optional)")
                logger.error("Data directory not found")
                return False
            
            # Check if embeddings file exists
            embeddings_path = "Data/SEC5_embeddings.pkl"
            if not os.path.exists(embeddings_path):
                st.error(f"❌ Embeddings file not found: {embeddings_path}")
                st.info("Please ensure 'SEC5_embeddings.pkl' is in the Data folder and committed to your repository.")
                logger.error(f"Embeddings file not found: {embeddings_path}")
                return False
            
            # Load embeddings and chunks
            with open(embeddings_path, "rb") as f:
                data = pickle.load(f)
            
            _self.chunks = [d["chunk"] for d in data]
            _self.embeddings = np.array([d["embedding"] for d in data], dtype="float32")
            
            # Validate loaded data
            if not _self.chunks or len(_self.chunks) == 0:
                st.error("❌ No document chunks found in embeddings file.")
                logger.error("Empty chunks in embeddings file")
                return False
            
            # Load or create FAISS index
            faiss_path = "Data/SEC5_faiss.index"
            if os.path.exists(faiss_path):
                _self.index = faiss.read_index(faiss_path)
                logger.info(f"Loaded existing FAISS index from {faiss_path}")
            else:
                # Create index if it doesn't exist
                st.info("Creating FAISS index (first-time setup)...")
                embedding_dim = _self.embeddings.shape[1]
                _self.index = faiss.IndexFlatL2(embedding_dim)
                _self.index.add(_self.embeddings)
                
                # Try to save index (might fail in read-only deployment)
                try:
                    faiss.write_index(_self.index, faiss_path)
                    logger.info(f"Created and saved FAISS index to {faiss_path}")
                except Exception as e:
                    logger.warning(f"Could not save FAISS index (running in memory): {str(e)}")
            
            # Load sentence transformer model
            st.info("Loading AI model...")
            _self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            logger.info(f"Successfully loaded {len(_self.chunks)} chunks and FAISS index with {_self.index.ntotal} vectors")
            st.success(f"✅ Loaded {len(_self.chunks)} document chunks successfully!")
            return True
            
        except FileNotFoundError as e:
            st.error(f"❌ File not found: {str(e)}")
            st.info("**Deployment Checklist:**\n1. Ensure 'Data' folder is in your repository\n2. Commit SEC5_embeddings.pkl file\n3. Check file paths are relative (not absolute)")
            logger.error(f"File not found error: {str(e)}")
            return False
            
        except pickle.UnpicklingError as e:
            st.error(f"❌ Error loading pickle file. The file may be corrupted: {str(e)}")
            logger.error(f"Pickle error: {str(e)}")
            return False
            
        except Exception as e:
            st.error(f"❌ Unexpected error loading resources: {str(e)}")
            st.info("Please check the application logs for more details.")
            logger.error(f"Error loading resources: {str(e)}", exc_info=True)
            return False
    
    def search_documents(self, query, k=5):
        """Search for relevant documents given a query"""
        if not all([self.model, self.chunks, self.index]):
            st.error("⚠️ Search system not initialized. Please reload the page.")
            logger.error("Search attempted with uninitialized components")
            return [], False
        
        try:
            # Encode query
            query_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
            
            # Search FAISS index
            distances, indices = self.index.search(query_emb, k)
            
            # Get results
            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                results.append({
                    "rank": i + 1,
                    "chunk": self.chunks[idx],
                    "distance": float(dist),
                    "similarity": 1 / (1 + dist)  # Convert distance to similarity score
                })
            
            return results, True
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)
            st.error(f"Search error: {str(e)}")
            return [], False

def main():
    # Page configuration
    st.set_page_config(
        page_title="FinComBot - Compliance Chatbot",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-score {
        background-color: #e8f4fd;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏦 FinComBot - Compliance Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading compliance documents and models..."):
            st.session_state.chatbot = ComplianceChatbot()
            
            # Check if initialization was successful
            if not st.session_state.chatbot.chunks:
                st.error("❌ Failed to initialize the chatbot. Please check the errors above.")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About FinComBot")
        st.write("""
        FinComBot helps bank staff quickly find relevant information from KYC/AML/CFT compliance policies.
        
        **Target Users:**
        - Front office staff
        - Relationship Managers
        - Operations staff
        - Compliance officers
        - New staff (training)
        """)
        
        st.header("🎯 How to Use")
        st.write("""
        1. Enter your compliance question in the search box
        2. Click 'Search' or press Enter
        3. Review the most relevant policy sections
        4. Click 'Show More' to see full content
        """)
        
        st.header("📊 System Info")
        if hasattr(st.session_state.chatbot, 'chunks') and st.session_state.chatbot.chunks:
            st.metric("Documents Loaded", len(st.session_state.chatbot.chunks))
            st.metric("Search Method", "Semantic Similarity")
            st.metric("Model", "MiniLM-L6-v2")
        else:
            st.warning("⚠️ System not ready")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Compliance Question")
        
        # Sample queries
        sample_queries = [
            "What documents are required to open a church account?",
            "How do we verify foreign national customers?",
            "What are the KYC requirements for minors?",
            "What documents are needed for CDD?",
            "How to handle suspicious transactions?",
            "What are the account opening requirements for companies?"
        ]
        
        selected_sample = st.selectbox(
            "Try a sample query:",
            [""] + sample_queries,
            key="sample_query"
        )
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=selected_sample if selected_sample else "",
            height=100,
            placeholder="e.g., What documents are required for opening a business account?"
        )
        
        # Search parameters
        col_a, col_b = st.columns([1, 1])
        with col_a:
            num_results = st.slider("Number of results:", 1, 10, 5)
        with col_b:
            min_similarity = st.slider("Minimum similarity:", 0.0, 1.0, 0.1, 0.1)
        
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        st.header("📈 Query Statistics")
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        st.metric("Total Searches", len(st.session_state.search_history))
        if st.session_state.search_history:
            avg_results = np.mean([len(h['results']) for h in st.session_state.search_history])
            st.metric("Avg Results per Query", f"{avg_results:.1f}")
    
    # Perform search
    if search_clicked and query.strip():
        with st.spinner("Searching compliance documents..."):
            results, success = st.session_state.chatbot.search_documents(query, num_results)
            
            if success and results:
                # Filter by minimum similarity
                filtered_results = [r for r in results if r['similarity'] >= min_similarity]
                
                # Store in history
                st.session_state.search_history.append({
                    'query': query,
                    'results': filtered_results,
                    'timestamp': datetime.now()
                })
                
                st.success(f"Found {len(filtered_results)} relevant results")
                
                # Display results
                for result in filtered_results:
                    with st.expander(f"📄 Result #{result['rank']} - Similarity: {result['similarity']:.3f}", expanded=True):
                        
                        # Show similarity score
                        st.markdown(f"""
                        <div class="similarity-score">
                            Similarity Score: {result['similarity']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show chunk content
                        chunk_preview = result['chunk'][:400] + "..." if len(result['chunk']) > 400 else result['chunk']
                        st.write(chunk_preview)
                        
                        # Full content toggle
                        if len(result['chunk']) > 400:
                            if st.button(f"Show Full Content", key=f"show_full_{result['rank']}"):
                                st.text_area("Full Content:", result['chunk'], height=200, key=f"full_{result['rank']}")
                        
                        # Highlight query terms
                        query_words = query.lower().split()
                        highlighted_words = []
                        for word in query_words:
                            if word in result['chunk'].lower():
                                highlighted_words.append(word)
                        
                        if highlighted_words:
                            st.caption(f"🎯 Matched terms: {', '.join(highlighted_words)}")
                
            elif success and not results:
                st.warning("No results found. Try rephrasing your question or reducing the minimum similarity threshold.")
            else:
                st.error("Search failed. Please check if the documents are loaded correctly.")
    
    elif search_clicked:
        st.warning("Please enter a question to search.")
    
    # Search History
    if st.session_state.search_history:
        with st.expander("📚 Recent Searches", expanded=False):
            for i, search in enumerate(reversed(st.session_state.search_history[-5:])):  # Show last 5
                st.write(f"**Q{len(st.session_state.search_history)-i}:** {search['query']}")
                st.write(f"*{search['timestamp'].strftime('%Y-%m-%d %H:%M')} - {len(search['results'])} results*")
                st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🏦 FinComBot v1.0 | Compliance Document Retrieval System<br>
        <small>Built for KYC/AML/CFT Policy Navigation | Internal Use Only</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()