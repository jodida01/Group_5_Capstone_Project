import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import re
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedComplianceSearch:
    def __init__(self, embeddings_file: str, faiss_index_file: str, model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings_file = embeddings_file
        self.faiss_index_file = faiss_index_file
        self.model_name = model_name
        
        self.model = None
        self.chunks = None
        self.index = None
        self.embeddings = None
        self.chunk_metadata = None
        
    def load_resources(self) -> bool:
        """Load all necessary resources for search"""
        try:
            # Load embeddings and chunks
            with open(self.embeddings_file, "rb") as f:
                data = pickle.load(f)
            
            self.chunks = [d["chunk"] for d in data]
            self.embeddings = np.array([d["embedding"] for d in data], dtype="float32")
            
            # Create metadata for each chunk
            self.chunk_metadata = self._create_chunk_metadata()
            
            # Load FAISS index
            if Path(self.faiss_index_file).exists():
                self.index = faiss.read_index(self.faiss_index_file)
            else:
                self._create_faiss_index()
            
            # Load sentence transformer
            self.model = SentenceTransformer(self.model_name)
            
            logger.info(f"Successfully loaded {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            return False
    
    def _create_faiss_index(self):
        """Create and save FAISS index"""
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.embeddings)
        faiss.write_index(self.index, self.faiss_index_file)
    
    def _create_chunk_metadata(self) -> List[Dict]:
        """Create metadata for each chunk to help with categorization"""
        metadata = []
        
        for i, chunk in enumerate(self.chunks):
            meta = {
                'chunk_id': i,
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'category': self._categorize_chunk(chunk),
                'keywords': self._extract_keywords(chunk)
            }
            metadata.append(meta)
        
        return metadata
    
    def _categorize_chunk(self, chunk: str) -> str:
        """Categorize chunk based on content patterns"""
        chunk_lower = chunk.lower()
        
        if any(word in chunk_lower for word in ['church', 'religious', 'pastor', 'reverend']):
            return 'religious_organizations'
        elif any(word in chunk_lower for word in ['minor', 'child', 'guardian', 'parent']):
            return 'minor_accounts'
        elif any(word in chunk_lower for word in ['company', 'corporate', 'business', 'director']):
            return 'corporate_accounts'
        elif any(word in chunk_lower for word in ['foreign', 'non-resident', 'nationality']):
            return 'foreign_nationals'
        elif any(word in chunk_lower for word in ['kyc', 'know your customer', 'identification']):
            return 'kyc_requirements'
        elif any(word in chunk_lower for word in ['aml', 'money laundering', 'suspicious']):
            return 'aml_compliance'
        elif any(word in chunk_lower for word in ['cdd', 'due diligence', 'enhanced']):
            return 'due_diligence'
        else:
            return 'general'
    
    def _extract_keywords(self, chunk: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from chunk"""
        # Simple keyword extraction - can be enhanced with more sophisticated NLP
        words = re.findall(r'\b[A-Za-z]{3,}\b', chunk.lower())
        
        # Common compliance terms to prioritize
        priority_terms = [
            'kyc', 'aml', 'cdd', 'edd', 'account', 'customer', 'document', 
            'verification', 'identification', 'compliance', 'requirement',
            'certificate', 'signature', 'mandate'
        ]
        
        keywords = []
        for term in priority_terms:
            if term in words:
                keywords.append(term)
        
        # Add other frequent words
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and add top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:max_keywords-len(keywords)]:
            keywords.append(word)
        
        return keywords[:max_keywords]
    
    def semantic_search(self, query: str, k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Perform semantic search using embeddings"""
        if not self._is_ready():
            return []
        
        try:
            # Encode query
            query_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
            
            # Search FAISS index
            distances, indices = self.index.search(query_emb, k * 2)  # Get more results for filtering
            
            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                
                if similarity >= min_similarity:
                    result = {
                        'rank': len(results) + 1,
                        'chunk_id': int(idx),
                        'chunk': self.chunks[idx],
                        'distance': float(dist),
                        'similarity': float(similarity),
                        'metadata': self.chunk_metadata[idx],
                        'matched_keywords': self._find_matched_keywords(query, self.chunks[idx])
                    }
                    results.append(result)
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            return []
    
    def keyword_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform keyword-based search as fallback"""
        if not self.chunks:
            return []
        
        query_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', query.lower()))
        
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(re.findall(r'\b[A-Za-z]{3,}\b', chunk.lower()))
            
            # Calculate overlap score
            overlap = len(query_terms.intersection(chunk_words))
            total_query_terms = len(query_terms)
            
            if total_query_terms > 0:
                score = overlap / total_query_terms
                scores.append((i, score, overlap))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        results = []
        for rank, (chunk_id, score, overlap) in enumerate(scores[:k], 1):
            if score > 0:  # Only return results with some overlap
                result = {
                    'rank': rank,
                    'chunk_id': chunk_id,
                    'chunk': self.chunks[chunk_id],
                    'keyword_score': score,
                    'matched_terms': overlap,
                    'metadata': self.chunk_metadata[chunk_id],
                    'matched_keywords': self._find_matched_keywords(query, self.chunks[chunk_id])
                }
                results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, min_similarity: float = 0.1, 
                     semantic_weight: float = 0.7) -> List[Dict]:
        """Combine semantic and keyword search for better results"""
        
        semantic_results = self.semantic_search(query, k * 2, min_similarity)
        keyword_results = self.keyword_search(query, k * 2)
        
        # Combine and re-rank results
        combined_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {
                'semantic_score': result['similarity'],
                'keyword_score': 0,
                'result': result
            }
        
        # Add keyword scores
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['keyword_score'] = result.get('keyword_score', 0)
            else:
                combined_scores[chunk_id] = {
                    'semantic_score': 0,
                    'keyword_score': result.get('keyword_score', 0),
                    'result': result
                }
        
        # Calculate hybrid scores
        final_results = []
        for chunk_id, scores in combined_scores.items():
            hybrid_score = (semantic_weight * scores['semantic_score'] + 
                          (1 - semantic_weight) * scores['keyword_score'])
            
            result = scores['result'].copy()
            result['hybrid_score'] = hybrid_score
            result['semantic_score'] = scores['semantic_score']
            result['keyword_score'] = scores['keyword_score']
            
            final_results.append(result)
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(final_results[:k], 1):
            result['rank'] = i
        
        return final_results[:k]
    
    def _find_matched_keywords(self, query: str, chunk: str) -> List[str]:
        """Find which query terms appear in the chunk"""
        query_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', query.lower()))
        chunk_words = set(re.findall(r'\b[A-Za-z]{3,}\b', chunk.lower()))
        
        return list(query_terms.intersection(chunk_words))
    
    def _is_ready(self) -> bool:
        """Check if all resources are loaded"""
        return all([self.model, self.chunks, self.index, self.embeddings])
    
    def get_statistics(self) -> Dict:
        """Get search engine statistics"""
        if not self.chunks:
            return {}
        
        categories = {}
        total_words = 0
        
        for meta in self.chunk_metadata:
            category = meta['category']
            categories[category] = categories.get(category, 0) + 1
            total_words += meta['word_count']
        
        return {
            'total_chunks': len(self.chunks),
            'total_words': total_words,
            'avg_words_per_chunk': total_words / len(self.chunks) if self.chunks else 0,
            'categories': categories,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': self.model_name
        }