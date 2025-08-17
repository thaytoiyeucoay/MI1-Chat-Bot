"""
Advanced Semantic Chunking Module for AI Trợ Giảng Toán Tin
Implements multiple intelligent chunking strategies to improve chatbot accuracy
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticChunker:
    """Advanced semantic chunking with multiple strategies"""
    
    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings, 
                 chunk_size: int = 800, 
                 chunk_overlap: int = 100,
                 similarity_threshold: float = 0.75):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Fallback splitter for basic chunking
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document], 
                       strategy: str = "adaptive") -> List[Document]:
        """
        Main chunking method with multiple strategies
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy ('adaptive', 'semantic', 'topic', 'sentence')
        
        Returns:
            List of chunked documents
        """
        try:
            if strategy == "adaptive":
                return self._adaptive_chunking(documents)
            elif strategy == "semantic":
                return self._semantic_chunking(documents)
            elif strategy == "topic":
                return self._topic_based_chunking(documents)
            elif strategy == "sentence":
                return self._sentence_based_chunking(documents)
            else:
                # Fallback to basic chunking
                return self._basic_chunking(documents)
                
        except Exception as e:
            self.logger.error(f"Chunking failed with strategy {strategy}: {e}")
            # Fallback to basic chunking
            return self._basic_chunking(documents)
    
    def _basic_chunking(self, documents: List[Document]) -> List[Document]:
        """Basic recursive character text splitting"""
        chunks = []
        for doc in documents:
            doc_chunks = self.fallback_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        return chunks
    
    def _adaptive_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Adaptive chunking that chooses the best strategy based on content analysis
        """
        chunks = []
        
        for doc in documents:
            content = doc.page_content
            content_type = self._analyze_content_type(content)
            
            if content_type == "mathematical":
                # Use smaller, precise chunks for mathematical content
                doc_chunks = self._mathematical_chunking([doc])
            elif content_type == "narrative":
                # Use semantic chunking for narrative content
                doc_chunks = self._semantic_chunking([doc])
            elif content_type == "structured":
                # Use topic-based chunking for structured content
                doc_chunks = self._topic_based_chunking([doc])
            else:
                # Default to sentence-based chunking
                doc_chunks = self._sentence_based_chunking([doc])
            
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _analyze_content_type(self, content: str) -> str:
        """Analyze content type to determine best chunking strategy"""
        # Mathematical content indicators
        math_patterns = [
            r'\b(định lý|công thức|phương trình|bất đẳng thức|tích phân|đạo hàm)\b',
            r'[∫∑∏√±×÷≤≥≠∞]',
            r'\b(sin|cos|tan|log|ln|exp)\b',
            r'\$.*?\$',  # LaTeX math
            r'\\[a-zA-Z]+\{.*?\}'  # LaTeX commands
        ]
        
        # Structured content indicators
        structured_patterns = [
            r'^\s*\d+\.',  # Numbered lists
            r'^\s*[a-zA-Z]\)',  # Lettered lists
            r'^\s*[-•]\s',  # Bullet points
            r'\b(Bài \d+|Chương \d+|Phần \d+)\b'  # Chapters/sections
        ]
        
        math_score = sum(len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)) 
                        for pattern in math_patterns)
        structured_score = sum(len(re.findall(pattern, content, re.MULTILINE)) 
                              for pattern in structured_patterns)
        
        if math_score > 5:
            return "mathematical"
        elif structured_score > 3:
            return "structured"
        elif len(content.split('.')) > 10:  # Many sentences
            return "narrative"
        else:
            return "general"
    
    def _mathematical_chunking(self, documents: List[Document]) -> List[Document]:
        """Specialized chunking for mathematical content"""
        chunks = []
        
        for doc in documents:
            content = doc.page_content
            
            # Split by mathematical sections
            sections = re.split(r'\n\s*(?=Bài \d+|Định lý \d+|Công thức \d+)', content)
            
            for section in sections:
                if len(section.strip()) < 50:  # Skip very short sections
                    continue
                
                # Further split long mathematical sections
                if len(section) > self.chunk_size:
                    # Split by paragraphs first
                    paragraphs = section.split('\n\n')
                    current_chunk = ""
                    
                    for para in paragraphs:
                        if len(current_chunk + para) > self.chunk_size and current_chunk:
                            # Create chunk
                            chunk_doc = Document(
                                page_content=current_chunk.strip(),
                                metadata={
                                    **doc.metadata,
                                    "chunk_type": "mathematical",
                                    "chunk_method": "mathematical_section"
                                }
                            )
                            chunks.append(chunk_doc)
                            current_chunk = para
                        else:
                            current_chunk += "\n\n" + para if current_chunk else para
                    
                    # Add remaining content
                    if current_chunk.strip():
                        chunk_doc = Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                **doc.metadata,
                                "chunk_type": "mathematical",
                                "chunk_method": "mathematical_section"
                            }
                        )
                        chunks.append(chunk_doc)
                else:
                    chunk_doc = Document(
                        page_content=section.strip(),
                        metadata={
                            **doc.metadata,
                            "chunk_type": "mathematical",
                            "chunk_method": "mathematical_complete"
                        }
                    )
                    chunks.append(chunk_doc)
        
        return chunks
    
    def _sentence_based_chunking(self, documents: List[Document]) -> List[Document]:
        """Sentence-based chunking with semantic similarity"""
        chunks = []
        
        for doc in documents:
            sentences = sent_tokenize(doc.page_content)
            
            if len(sentences) <= 3:
                # Too few sentences, keep as single chunk
                chunk_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "sentence_based",
                        "sentence_count": len(sentences)
                    }
                )
                chunks.append(chunk_doc)
                continue
            
            # Group sentences by semantic similarity
            sentence_groups = self._group_sentences_by_similarity(sentences)
            
            for group in sentence_groups:
                chunk_text = " ".join(group)
                if len(chunk_text.strip()) > 50:  # Minimum chunk size
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "sentence_based",
                            "sentence_count": len(group)
                        }
                    )
                    chunks.append(chunk_doc)
        
        return chunks
    
    def _group_sentences_by_similarity(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences by semantic similarity"""
        if len(sentences) <= 2:
            return [sentences]
        
        try:
            # Get embeddings for sentences (with rate limiting)
            embeddings_list = []
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # Skip very short sentences
                    embedding = self.embeddings.embed_query(sentence)
                    embeddings_list.append(embedding)
                    time.sleep(0.1)  # Rate limiting
                else:
                    embeddings_list.append([0] * 768)  # Placeholder
            
            if not embeddings_list:
                return [sentences]
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings_list)
            
            # Group sentences based on similarity threshold
            groups = []
            used_indices = set()
            
            for i, sentence in enumerate(sentences):
                if i in used_indices:
                    continue
                
                current_group = [sentence]
                used_indices.add(i)
                current_length = len(sentence)
                
                # Find similar sentences to group together
                for j in range(i + 1, len(sentences)):
                    if j in used_indices:
                        continue
                    
                    if (similarity_matrix[i][j] > self.similarity_threshold and 
                        current_length + len(sentences[j]) < self.chunk_size):
                        current_group.append(sentences[j])
                        used_indices.add(j)
                        current_length += len(sentences[j])
                
                groups.append(current_group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Sentence grouping failed: {e}")
            # Fallback to simple grouping
            return self._simple_sentence_grouping(sentences)
    
    def _simple_sentence_grouping(self, sentences: List[str]) -> List[List[str]]:
        """Simple sentence grouping by length"""
        groups = []
        current_group = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_group:
                groups.append(current_group)
                current_group = [sentence]
                current_length = len(sentence)
            else:
                current_group.append(sentence)
                current_length += len(sentence)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """Semantic chunking using embedding similarity"""
        chunks = []
        
        for doc in documents:
            # Split into paragraphs first
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            
            if len(paragraphs) <= 2:
                chunk_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "semantic",
                        "paragraph_count": len(paragraphs)
                    }
                )
                chunks.append(chunk_doc)
                continue
            
            # Group paragraphs by semantic similarity
            paragraph_groups = self._group_paragraphs_by_similarity(paragraphs)
            
            for group in paragraph_groups:
                chunk_text = "\n\n".join(group)
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "semantic",
                        "paragraph_count": len(group)
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def _group_paragraphs_by_similarity(self, paragraphs: List[str]) -> List[List[str]]:
        """Group paragraphs by semantic similarity"""
        if len(paragraphs) <= 2:
            return [paragraphs]
        
        try:
            # Get embeddings for paragraphs
            embeddings_list = []
            for para in paragraphs:
                if len(para.strip()) > 20:
                    embedding = self.embeddings.embed_query(para)
                    embeddings_list.append(embedding)
                    time.sleep(0.1)  # Rate limiting
                else:
                    embeddings_list.append([0] * 768)  # Placeholder
            
            # Use clustering to group similar paragraphs
            n_clusters = min(max(2, len(paragraphs) // 3), len(paragraphs))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_list)
            
            # Group paragraphs by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(paragraphs[i])
            
            # Ensure chunks don't exceed size limit
            final_groups = []
            for cluster_paras in clusters.values():
                current_group = []
                current_length = 0
                
                for para in cluster_paras:
                    if current_length + len(para) > self.chunk_size and current_group:
                        final_groups.append(current_group)
                        current_group = [para]
                        current_length = len(para)
                    else:
                        current_group.append(para)
                        current_length += len(para)
                
                if current_group:
                    final_groups.append(current_group)
            
            return final_groups
            
        except Exception as e:
            self.logger.error(f"Paragraph grouping failed: {e}")
            # Fallback to simple grouping
            return self._simple_paragraph_grouping(paragraphs)
    
    def _simple_paragraph_grouping(self, paragraphs: List[str]) -> List[List[str]]:
        """Simple paragraph grouping by length"""
        groups = []
        current_group = []
        current_length = 0
        
        for para in paragraphs:
            if current_length + len(para) > self.chunk_size and current_group:
                groups.append(current_group)
                current_group = [para]
                current_length = len(para)
            else:
                current_group.append(para)
                current_length += len(para)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _topic_based_chunking(self, documents: List[Document]) -> List[Document]:
        """Topic-based chunking using content structure analysis"""
        chunks = []
        
        for doc in documents:
            content = doc.page_content
            
            # Identify topic boundaries
            topic_boundaries = self._identify_topic_boundaries(content)
            
            if not topic_boundaries:
                # No clear topics found, use semantic chunking
                chunks.extend(self._semantic_chunking([doc]))
                continue
            
            # Split content by topic boundaries
            topics = []
            start = 0
            
            for boundary in topic_boundaries:
                if boundary > start:
                    topic_content = content[start:boundary].strip()
                    if topic_content:
                        topics.append(topic_content)
                start = boundary
            
            # Add remaining content
            if start < len(content):
                remaining = content[start:].strip()
                if remaining:
                    topics.append(remaining)
            
            # Create chunks from topics
            for i, topic in enumerate(topics):
                if len(topic) > self.chunk_size:
                    # Topic too long, split further
                    sub_chunks = self.fallback_splitter.split_text(topic)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_doc = Document(
                            page_content=sub_chunk,
                            metadata={
                                **doc.metadata,
                                "chunk_type": "topic_based",
                                "topic_index": i,
                                "sub_chunk_index": j
                            }
                        )
                        chunks.append(chunk_doc)
                else:
                    chunk_doc = Document(
                        page_content=topic,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "topic_based",
                            "topic_index": i
                        }
                    )
                    chunks.append(chunk_doc)
        
        return chunks
    
    def _identify_topic_boundaries(self, content: str) -> List[int]:
        """Identify topic boundaries in content"""
        boundaries = []
        
        # Look for common topic indicators
        patterns = [
            r'\n\s*(Bài \d+|Chương \d+|Phần \d+|Mục \d+)',
            r'\n\s*(Định lý \d+|Định nghĩa \d+|Ví dụ \d+)',
            r'\n\s*\d+\.\s*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]',
            r'\n\s*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+\n'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                boundaries.append(match.start())
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # Filter out boundaries that are too close to each other
        filtered_boundaries = []
        min_distance = 200  # Minimum distance between boundaries
        
        for boundary in boundaries:
            if not filtered_boundaries or boundary - filtered_boundaries[-1] > min_distance:
                filtered_boundaries.append(boundary)
        
        return filtered_boundaries
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        chunk_types = [chunk.metadata.get("chunk_type", "unknown") for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "chunk_types": dict(zip(*np.unique(chunk_types, return_counts=True))),
            "total_characters": sum(chunk_lengths)
        }
        
        return stats


class ChunkingManager:
    """Manager class for handling different chunking strategies"""
    
    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings):
        self.chunker = SemanticChunker(embeddings)
        self.strategies = {
            "basic": "Cắt nhỏ cơ bản theo ký tự",
            "adaptive": "Tự động chọn phương pháp phù hợp",
            "semantic": "Cắt nhỏ theo ngữ nghĩa",
            "sentence": "Cắt nhỏ theo câu",
            "topic": "Cắt nhỏ theo chủ đề",
            "mathematical": "Chuyên biệt cho nội dung toán học"
        }
    
    def process_documents(self, documents: List[Document], 
                         strategy: str = "adaptive") -> Tuple[List[Document], Dict[str, Any]]:
        """
        Process documents with specified chunking strategy
        
        Returns:
            Tuple of (chunked_documents, statistics)
        """
        start_time = time.time()
        
        # Validate strategy
        if strategy not in self.strategies:
            strategy = "adaptive"
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents, strategy)
        
        # Get statistics
        stats = self.chunker.get_chunk_statistics(chunks)
        stats["processing_time"] = time.time() - start_time
        stats["strategy_used"] = strategy
        stats["strategy_description"] = self.strategies[strategy]
        
        return chunks, stats
    
    def get_available_strategies(self) -> Dict[str, str]:
        """Get available chunking strategies"""
        return self.strategies.copy()
