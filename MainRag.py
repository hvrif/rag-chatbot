# ============================================================================
# ULTIMATE IHRP RAG CHATBOT - NO LANGCHAIN
# Pure implementation with: OpenAI API + Qdrant + Sentence Transformers
# No LangChain dependency issues!
# ============================================================================

import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from datetime import datetime
import sqlite3
import hashlib
import schedule
import threading
import time
from collections import deque

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from sentence_transformers import SentenceTransformer, CrossEncoder
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup

import gradio as gr
import numpy as np
from uuid import uuid4

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for IHRP RAG Chatbot"""
    
    # ========== DATA SOURCES ==========
    JSONL_FILE_FAQ = "./Dataset/ihrp_support_data.jsonl"
    JSONL_FILE_DATASET = "./Dataset/ihrp_dataset_cleaned_final.jsonl"
    
    WEBSITE_URLS = [

         # General Overview Pages
        "https://ihrp.sg/certifications-overview/",
        "https://ihrp.sg/enhanced-subsidy-for-hr-certification/",
        "https://ihrp.sg/renew-your-certification/",

        # IHRP-CA: Certified Associate (Use specific section links for clarity)
        # Fix for CA
        # IHRP-CA: Certified Associate - More explicit URLs
        "https://ihrp.sg/certifications/ihrp-certified-associate-ihrp-ca/?id-process=id-overview",
        "https://ihrp.sg/certifications/ihrp-certified-associate-ihrp-ca/?id-process=id-fee",
        "https://ihrp.sg/certifications/ihrp-certified-associate-ihrp-ca/?id-process=id-date",
        "https://ihrp.sg/certifications/ihrp-certified-associate-ihrp-ca/?id-process=id-process",
        "https://ihrp.sg/certifications/ihrp-certified-associate-ihrp-ca/?id-process=id-preparation",

        # IHRP-CP: Certified Professional (Use specific section links for clarity - ASSUMING STRUCTURE IS SIMILAR TO CA)
        # Suggested Fix for CP (Requires verification of the actual CP tab IDs)
        "https://ihrp.sg/certifications/ihrp-certified-professional-ihrp-cp/?id-process=id-overview",
        "https://ihrp.sg/certifications/ihrp-certified-professional-ihrp-cp/?id-process=id-fee",
        "https://ihrp.sg/certifications/ihrp-certified-professional-ihrp-cp/?id-process=id-process",
        "https://ihrp.sg/certifications/ihrp-certified-professional-ihrp-cp/?id-process=id-preparation",

        # IHRP-SP: Senior Professional (Keeping original for general coverage)
        "https://ihrp.sg/certifications/certified-ihrp-senior-professional-ihrp-sp/",
        
        # Other Key Pages (Retained from original list)
        "https://ihrp.sg/microsoftcrmportals.com/professionals/",
        "https://ihrp.sg/body-of-competencies/",
        "https://ihrp.sg/context-and-methodology/",
        "https://ihrp.sg/case-studies/",
        "https://ihrp.sg/explore-job-roles/",
        "https://ihrp.sg/tools-and-resources/",
        "https://ihrp.sg/job-redesign-evaluation-tool/?tab=overview/",
        "https://ihrp.sg/job-redesign-evaluation-tool/?tab=jr-process/",
        "https://ihrp.sg/job-redesign-evaluation-tool/?tab=jr-expert-panel/",
        "https://ihrp.sg/job-redesign-evaluation-tool/?tab=resources/",
        "https://ihrp.sg/job-redesign-evaluation-tool/?tab=jr-evaluation-tool/",
        "https://ihrp.sg/hcdt-overview/",
        "https://ihrp.sg/for-organisation/",
        "https://ihrp.sg/skill-badges-overview/",
        "https://ihrp.sg/for-professionals/",
        "https://ihrp.sg/find-a-course/",
        "https://ihrp.sg/corporate-partner-programme-overview/",
        "https://ihrp.sg/eco-system-partnership/",
        "https://ihrp.sg/international-recognition/",
        "https://ihrp.sg/cipd-partnership/",
        "https://ihrp.sg/shrm-partnership/",
        "https://ihrp.sg/resources-overview/",
        "https://ihrp.sg/playbooks/",
        "https://ihrp.sg/resources/research-insights/",
        "https://ihrp.sg/about-ihrp/",
        "https://ihrp.sg/our-board/",
        "https://ihrp.sg/our-management/",
        "https://ihrp.sg/our-committees/",
        "https://ihrp.sg/ihrp-master-professionals/"


    ]
    
    
    # Source credibility weights
    SOURCE_CREDIBILITY = {
        "ihrp.sg/certifications": 1.0,
        "ihrp.sg/body-of-competencies": 0.95,
        "IHRP FAQ": 0.9,
        "IHRP Dataset": 0.85,
        "default": 0.7
    }
    
    USE_JSONL_FAQ = True 
    USE_JSONL_DATASET = False
    USE_WEBSITES = True
    
    # ========== MODEL SETTINGS ==========
    # Embedding model (sentence-transformers)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
    
    # LLM settings
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0.1
    
    # Reranker
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # ========== RETRIEVAL SETTINGS ==========
    CHUNK_SIZE = 1500  # Increased from 1000 to capture complete pricing tables
    CHUNK_OVERLAP = 300  # Increased from 200 for better context preservation
    INITIAL_K = 25  # Increased from 20 for more comprehensive retrieval
    FINAL_K = 8  
    
    # ========== MEMORY & GUARDRAILS ==========
    CONVERSATION_MEMORY_WINDOW = 10  # messages
    ENABLE_GUARDRAILS = True
    CONFIDENCE_THRESHOLD = 0.6
    
    # ========== AUTO-UPDATE SETTINGS ==========
    AUTO_UPDATE_ENABLED = True
    UPDATE_SCHEDULE = "weekly"
    UPDATE_DAY = "monday"
    UPDATE_TIME = "03:00"
    
    # ========== DATABASE ==========
    FEEDBACK_DB = "./feedback.db"
    CONTENT_HASH_FILE = "./content_hashes.json"
    
    # ========== API KEYS ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")  # Optional: use local if not provided
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = "ihrp-rag-ultimate"
    
    PLAYWRIGHT_TIMEOUT_MS = 90000

# ============================================================================
# DOCUMENT CLASS (Simple replacement for LangChain Document)
# ============================================================================

class Document:
    """Simple document class"""
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(content={self.page_content[:50]}..., metadata={self.metadata})"

# ============================================================================
# TEXT SPLITTER (Simple replacement for RecursiveCharacterTextSplitter)
# ============================================================================

class TextSplitter:
    """Simple text splitter"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # At least 50% of chunk
                    end = start + break_point + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_docs.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, 'chunk_index': i}
                ))
        
        return split_docs

# ============================================================================
# CONVERSATION MEMORY (Simple replacement for LangChain Memory)
# ============================================================================

class ConversationMemory:
    """Simple conversation memory with sliding window"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages = deque(maxlen=window_size)
    
    def add_message(self, role: str, content: str):
        """Add a message to memory"""
        self.messages.append({"role": role, "content": content})
    
    def get_history(self) -> str:
        """Get formatted conversation history"""
        if not self.messages:
            return "No previous conversation."
        
        history = []
        for msg in self.messages:
            if msg['role'] == 'user':
                history.append(f"User: {msg['content']}")
            else:
                history.append(f"Assistant: {msg['content']}")
        
        return "\n".join(history)
    
    def clear(self):
        """Clear conversation history"""
        self.messages.clear()

# ============================================================================
# FEEDBACK DATABASE
# ============================================================================

class FeedbackManager:
    """Manage user feedback"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                question TEXT,
                answer TEXT,
                rating INTEGER,
                comment TEXT,
                sources TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Feedback database initialized")
    
    def add_feedback(self, question: str, answer: str, rating: int, 
                     comment: str = "", sources: str = ""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (timestamp, question, answer, rating, comment, sources)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), question, answer, rating, comment, sources))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(rating) as avg_rating,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
            FROM feedback
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "total_queries": result[0],
            "avg_rating": round(result[1], 2) if result[1] else 0,
            "positive_feedback": result[2],
            "negative_feedback": result[3]
        }

# ============================================================================
# CONTENT CHANGE DETECTOR
# ============================================================================

class ContentChangeDetector:
    """Detect changes in website content"""
    
    def __init__(self, hash_file):
        self.hash_file = hash_file
        self.hashes = self._load_hashes()
    
    def _load_hashes(self):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_hashes(self):
        with open(self.hash_file, 'w') as f:
            json.dump(self.hashes, f, indent=2)
    
    def compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    def has_changed(self, url: str, content: str) -> bool:
        current_hash = self.compute_hash(content)
        previous_hash = self.hashes.get(url)
        
        if previous_hash != current_hash:
            self.hashes[url] = current_hash
            self._save_hashes()
            return True
        return False
    
    def get_changed_urls(self, url_content_pairs: List[Tuple[str, str]]) -> List[str]:
        changed = []
        for url, content in url_content_pairs:
            if self.has_changed(url, content):
                changed.append(url)
        return changed

# ============================================================================
# SOURCE CREDIBILITY SCORER
# ============================================================================

class SourceCredibilityScorer:
    """Score documents based on source credibility"""
    
    def __init__(self, credibility_map: Dict[str, float]):
        self.credibility_map = credibility_map
    
    def get_credibility_score(self, source: str) -> float:
        for pattern, score in self.credibility_map.items():
            if pattern in source:
                return score
        return self.credibility_map.get("default", 0.5)

# ============================================================================
# GUARDRAILS VALIDATOR
# ============================================================================

class GuardrailsValidator:
    """Validate responses for accuracy"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.validation_prompt = """You are a quality validator for IHRP certification information.

Analyze this response and determine if it meets these criteria:
1. Factually accurate (no made-up information)
2. Relevant to the question
3. Professional and helpful
4. Contains no harmful or misleading content

Question: {question}
Response: {response}

Return ONLY a JSON object with:
{{"is_valid": true/false, "confidence": 0.0-1.0, "issues": ["list of issues"], "suggestion": "improvement"}}"""
    
    def validate_response(self, question: str, response: str) -> Dict:
        try:
            prompt = self.validation_prompt.format(question=question, response=response)
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            result = json.loads(completion.choices[0].message.content)
            return result
        except:
            return {
                "is_valid": True,
                "confidence": 0.5,
                "issues": ["Validation check failed"],
                "suggestion": ""
            }

# ============================================================================
# DATA LOADER - FIXED VERSION
# ============================================================================

class DataLoader:
    """Load data from JSONL and websites"""
    
    @staticmethod
    def load_jsonl_file(filepath, source_name, source_type):
        print(f"📂 Loading data from {source_name} ({filepath})...")
        
        if not os.path.exists(filepath):
            print(f"❌ Error: {filepath} not found!")
            return []
        
        documents = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        instruction = item.get('instruction', '')
                        output = item.get('output', '')
                        
                        content = f"""Question: {instruction}

Answer: {output}"""
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': source_name,
                                'question': instruction,
                                'type': source_type,
                                'index': line_num,
                                'certification_level': 'unknown'  # JSONL doesn't have URL
                            }
                        )
                        documents.append(doc)
                        
                    except json.JSONDecodeError:
                        print(f"⚠️  Skipping invalid JSON on line {line_num}")
                        continue
            
            print(f"✅ Loaded {len(documents)} articles from {source_name}")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading {source_name}: {e}")
            return []
    
    @staticmethod
    def load_websites(urls, timeout_ms, return_content_pairs=False):
        """Enhanced web scraping with proper dynamic tab handling"""
        if not urls:
            return [] if not return_content_pairs else ([], [])

        print(f"\n📥 Loading {len(urls)} website pages...")
        
        documents = []
        content_pairs = []
        successful_count = 0
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.new_page()
                page.set_default_timeout(timeout_ms)

                for idx, url in enumerate(urls, 1):
                    try:
                        print(f"  [{idx}/{len(urls)}] {url}")
                        
                        # Navigate to page
                        page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                        
                        # Special handling for pages with tab parameters
                        if '?id-process=' in url or '?tab=' in url:
                            try:
                                page.wait_for_selector(".tab-content, .elementor-tab-content", timeout=10000)
                            except:
                                pass
                            
                            tab_param = url.split('?')[-1].split('=')[-1] if '?' in url else None
                            if tab_param:
                                selectors_to_try = [
                                    f"[data-tab='{tab_param}']",
                                    f"a[href*='{tab_param}']",
                                    f".elementor-tab-title[data-tab='{tab_param}']",
                                    f"[id*='{tab_param}']"
                                ]
                                
                                for selector in selectors_to_try:
                                    try:
                                        if page.locator(selector).count() > 0:
                                            page.locator(selector).first.click()
                                            print(f"    ✓ Clicked tab: {tab_param}")
                                            page.wait_for_timeout(2000)
                                            break
                                    except:
                                        continue
                        
                        page.wait_for_timeout(3000)
                        
                        try:
                            page.wait_for_selector("main, .entry-content, .elementor-widget-container", 
                                                 timeout=5000, state="visible")
                        except:
                            pass
                        
                        content = page.content()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        for element in soup(["script", "style", "nav", "header", "footer", 
                                           "iframe", "noscript"]):
                            element.extract()
                        
                        main_content = soup.find('main') or soup.find(class_='entry-content')
                        if main_content:
                            text_content = main_content.get_text(separator=' ', strip=True)
                        else:
                            text_content = soup.get_text(separator=' ', strip=True)
                        
                        text_content = ' '.join(text_content.split())
                        
                        if len(text_content.strip()) < 100:
                            print(f"    ⚠️ Warning: Low content extracted ({len(text_content)} chars)")
                        
                        # Detect certification level from URL
                        cert_level = 'unknown'
                        if 'certified-associate' in url or 'ihrp-ca' in url:
                            cert_level = 'CA'
                        elif 'certified-professional' in url or 'ihrp-cp' in url:
                            cert_level = 'CP'
                        elif 'senior-professional' in url or 'ihrp-sp' in url:
                            cert_level = 'SP'
                        
                        doc = Document(
                            page_content=text_content,
                            metadata={
                                'source': url,
                                'type': 'website',
                                'certification_level': cert_level,
                                'title': page.title() if page.title() else url,
                                'last_updated': datetime.now().isoformat(),
                                'content_length': len(text_content),
                                'has_tab_param': '?' in url
                            }
                        )
                        documents.append(doc)
                        content_pairs.append((url, text_content))
                        successful_count += 1
                        
                        print(f"    ✓ Extracted {len(text_content)} characters (Level: {cert_level})")

                    except PlaywrightTimeoutError:
                        print(f"    ❌ Failed: Timeout")
                    except Exception as e:
                        print(f"    ❌ Failed: {type(e).__name__}: {str(e)[:50]}")
                
                context.close()
                browser.close()
                
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            import traceback
            traceback.print_exc()
            return [] if not return_content_pairs else ([], [])

        print(f"\n✅ Successfully loaded {successful_count}/{len(urls)} pages")
        
        documents = [doc for doc in documents if len(doc.page_content.strip()) > 100]
        print(f"📊 Kept {len(documents)} documents after content filtering")
        
        if return_content_pairs:
            return documents, content_pairs
        return documents

# ============================================================================
# COMPLETE RAG FUSION RETRIEVER - WITH BOTH METHODS
# ============================================================================

class RAGFusionRetriever:
    """RAG Fusion with multiple retrieval strategies"""
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str,
                 embedding_model: SentenceTransformer, reranker: CrossEncoder,
                 credibility_scorer: SourceCredibilityScorer):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.credibility_scorer = credibility_scorer
    
    def retrieve_with_fusion(self, queries: List[str], k: int, filter_cert_level: str = None) -> List[Dict]:
        """Retrieve documents using multiple strategies with optional filtering"""
        all_results = []
        seen_ids = set()
        
        # Check if this is a pricing query
        is_pricing_query = any(
            term in ' '.join(queries).lower() 
            for term in ['cost', 'fee', 'price', 'how much', 'subsidy', 'payment']
        )
        
        # If pricing query without specific level, search all levels
        if is_pricing_query and not filter_cert_level:
            print("💰 Detected pricing query - retrieving from all certification levels")
            filter_cert_level = None  # Ensure we get CA, CP, and SP results
        
        for query in queries:
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Build filter if certification level specified
            search_filter = None
            if filter_cert_level and not is_pricing_query:
                try:
                    search_filter = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.certification_level",
                                match=MatchValue(value=filter_cert_level)
                            )
                        ]
                    )
                    print(f"🔍 Filtering for certification level: {filter_cert_level}")
                except Exception as e:
                    print(f"⚠️  Filtering not available: {e}")
                    search_filter = None
            
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=k if not is_pricing_query else k * 2  # Get more results for pricing
                )
                
                for result in results:
                    if result.id not in seen_ids:
                        seen_ids.add(result.id)
                        all_results.append({
                            'id': result.id,
                            'content': result.payload['content'],
                            'metadata': result.payload['metadata'],
                            'score': result.score
                        })
            
            except Exception as e:
                if search_filter:
                    print(f"⚠️  Filtered search failed, retrying without filter...")
                    results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        query_filter=None,
                        limit=k
                    )
                    
                    for result in results:
                        if result.id not in seen_ids:
                            seen_ids.add(result.id)
                            all_results.append({
                                'id': result.id,
                                'content': result.payload['content'],
                                'metadata': result.payload['metadata'],
                                'score': result.score
                            })
                else:
                    raise e
        
        print(f"🔀 RAG Fusion: Retrieved {len(all_results)} unique documents")
        return all_results
    
    def rerank_with_credibility(self, question: str, documents: List[Dict], 
                                final_k: int) -> List[Dict]:
        """Rerank documents with relevance + credibility"""
        if not documents:
            return []
        
        # Get relevance scores from cross-encoder
        pairs = [[question, doc['content']] for doc in documents]
        relevance_scores = self.reranker.predict(pairs)
        
        # Combine with credibility
        final_scores = []
        for i, doc in enumerate(documents):
            source = doc['metadata'].get('source', '')
            cred_score = self.credibility_scorer.get_credibility_score(source)
            
            # Normalize relevance score
            rel_score = (relevance_scores[i] + 1) / 2  # -1 to 1 -> 0 to 1
            
            # Combined score: 70% relevance, 30% credibility
            combined_score = (0.7 * rel_score) + (0.3 * cred_score)
            final_scores.append((doc, combined_score))
        
        # Sort and return top K
        final_scores.sort(key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, score in final_scores[:final_k]]
        
        print(f"🎯 Reranked to top {len(reranked)} documents")
        return reranked

# ============================================================================
# PROMPTS
# ============================================================================

ADVANCED_PROMPT = """You are IHRP-GEN, an expert advisor on all IHRP certifications (CA, CP, SP, MTP).
Use the provided CONTEXT to answer accurately and in a professional, structured tone.

**CRITICAL PRICING GUIDELINES:**
When answering about fees or costs:

1. **For general "How much does certification cost?" questions:**
   - Present a COMPARISON TABLE showing all three main levels: CA, CP, SP
   - Show BOTH Singaporean/PR subsidized rates AND Non-Singaporean rates
   - Format: 
     ```
     Certification Level | SG/PR Standard Fee | Non-SG Standard Fee
     IHRP-CA            | S$272.50          | S$272.50
     IHRP-CP            | S$272.50          | S$1,635.00
     IHRP-SP            | S$381.50          | S$2,725.00
     ```

2. **For specific level questions (CA, CP, or SP):**
   - Show the standard fees for THAT LEVEL ONLY
   - Always include BOTH residency categories (SG/PR vs Non-SG)
   - Mention Enhanced Subsidy option (lowers fee to ~S$163.50)
   - Include recertification fee (S$490.50 for all levels)

3. **CORRECT FEE STRUCTURE (from context):**
   - **IHRP-CA**: S$272.50 for both SG/PR and Non-SG (standard)
   - **IHRP-CP**: S$272.50 (SG/PR subsidized), S$1,635.00 (Non-SG)
   - **IHRP-SP**: S$381.50 (SG/PR subsidized), S$2,725.00 (Non-SG)
   - **Enhanced Subsidy**: Further reduces to ~S$163.50 for eligible SG/PR
   - **Recertification**: S$490.50 (all levels, all nationalities)
   - **Additional fees**: On-site assessment S$43.60, Re-assessment S$163.50

4. **Other Important Points:**
   - Subsidies only for Singapore Citizens/PRs
   - SkillsFuture Credit can be used by eligible Singaporeans
   - Certification valid for 3 years
   - Enhanced Subsidy availability varies - check IHRP website

5. **ALWAYS:**
   - Use clear Markdown tables
   - Distinguish between subsidized and unsubsidized rates
   - Mention that fees are inclusive of GST
   - Note that Enhanced Subsidy is subject to availability

**For eligibility questions:**
- List requirements as bullet points
- Explain work experience requirements clearly
- Mention educational prerequisites

**AVOID:**
- Mixing fees from different categories
- Presenting incomplete pricing (must show both SG/PR and Non-SG)
- Vague statements - be specific with exact amounts
- Omitting the Enhanced Subsidy option

**ALWAYS END WITH:**
"For the most accurate and updated details, including current subsidy availability, please refer to the official IHRP website at https://ihrp.sg"

---------------------
CONTEXT:
{context}
---------------------

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

Respond as a confident IHRP expert. If pricing is incomplete in context, acknowledge gaps but provide what is available."""


# ============================================================================
# IMPROVED QUERY EXPANSION FOR PRICING QUESTIONS
# ============================================================================

QUERY_EXPANSION_PROMPT = """Given a user question about IHRP certifications, generate 3 related search queries.

Question: {question}

Guidelines:
- If question is about COSTS/FEES, generate queries about:
  1. Standard fees and pricing structure
  2. Subsidies and enhanced subsidy details  
  3. Additional fees (on-site, re-assessment, recertification)

- If question is about ELIGIBILITY, generate queries about:
  1. Work experience requirements
  2. Educational prerequisites
  3. Assessment process

- If question is about COMPARISON, generate queries about:
  1. Differences between certification levels
  2. Career progression pathways
  3. Benefits of each level

Return ONLY the 3 queries, one per line."""

# ============================================================================
# ULTIMATE CHATBOT (NO LANGCHAIN)
# ============================================================================

class UltimateIHRPChatbot:
    """Ultimate RAG Chatbot without LangChain"""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.qdrant_client = None
        self.embedding_model = None
        self.reranker = None
        self.memory = ConversationMemory(config.CONVERSATION_MEMORY_WINDOW)
        self.feedback_manager = FeedbackManager(config.FEEDBACK_DB)
        self.change_detector = ContentChangeDetector(config.CONTENT_HASH_FILE)
        self.credibility_scorer = SourceCredibilityScorer(config.SOURCE_CREDIBILITY)
        self.fusion_retriever = None
        self.guardrails = None
        self.update_thread = None
        
# ============================================================================
# FIXED SETUP METHOD WITH PAYLOAD INDEX
# ============================================================================

    def setup(self):
        """Initialize the system"""
        print("\n" + "="*60)
        print("🚀 ULTIMATE IHRP RAG CHATBOT SETUP (NO LANGCHAIN)")
        print("="*60 + "\n")
    
        # Load models
        print("🔢 Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
    
        print("🎯 Loading reranker...")
        self.reranker = CrossEncoder(self.config.RERANKER_MODEL)
        print("✅ Reranker loaded")
    
        # Setup Qdrant
        print("\n☁️  Setting up Qdrant...")
        if self.config.QDRANT_URL and self.config.QDRANT_API_KEY:
            self.qdrant_client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY
            )
            print("✅ Connected to Qdrant Cloud")
        else:
            self.qdrant_client = QdrantClient(path="./qdrant_local")
            print("✅ Using local Qdrant")
    
        # Create collection with payload schema
        from qdrant_client.models import PayloadSchemaType
    
        try:
            self.qdrant_client.create_collection(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.config.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print("✅ Created new collection")
        
            # Create payload index for certification_level field
            self.qdrant_client.create_payload_index(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                field_name="metadata.certification_level",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print("✅ Created payload index for certification_level")
        
        except Exception as e:
            print(f"⚠️  Collection may already exist: {e}")
            # Try to create index anyway
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.config.QDRANT_COLLECTION_NAME,
                    field_name="metadata.certification_level",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print("✅ Created payload index for certification_level")
            except:
                print("✅ Using existing collection and index")
    
        # Load documents
        all_documents = []
        content_pairs = []
    
        if self.config.USE_JSONL_FAQ:
            faq_docs = DataLoader.load_jsonl_file(
                self.config.JSONL_FILE_FAQ, 
                'IHRP FAQ', 
                'faq'
            )
            all_documents.extend(faq_docs)
    
        if self.config.USE_JSONL_DATASET:
            dataset_docs = DataLoader.load_jsonl_file(
                self.config.JSONL_FILE_DATASET, 
                'IHRP Dataset', 
                'dataset'
            )
            all_documents.extend(dataset_docs)
    
        if self.config.USE_WEBSITES:
            website_docs, content_pairs = DataLoader.load_websites(
                self.config.WEBSITE_URLS,
                self.config.PLAYWRIGHT_TIMEOUT_MS,
                return_content_pairs=True
            )
            all_documents.extend(website_docs)
    
        print(f"\n📚 Total documents loaded: {len(all_documents)}")
    
        # Split documents
        print("\n✂️  Splitting documents...")
        text_splitter = TextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"✅ Created {len(chunks)} chunks")
    
        # Create embeddings and upload to Qdrant
        print("\n💾 Creating embeddings and uploading to Qdrant...")
        points = []
        batch_size = 100
    
        for i, chunk in enumerate(chunks):
            if i % batch_size == 0:
                print(f"  Processing {i}/{len(chunks)}...")
        
            vector = self.embedding_model.encode(chunk.page_content).tolist()
        
            point = PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                }
            )
            points.append(point)
        
            # Upload in batches
            if len(points) >= batch_size:
                self.qdrant_client.upsert(
                    collection_name=self.config.QDRANT_COLLECTION_NAME,
                    points=points
                )
                points = []
    
        # Upload remaining
        if points:
            self.qdrant_client.upsert(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                points=points
            )
    
        print("✅ All documents uploaded to Qdrant")
    
        # Initialize components
        if self.config.ENABLE_GUARDRAILS:
            self.guardrails = GuardrailsValidator(self.openai_client)
            print("✅ Guardrails enabled")
    
        self.fusion_retriever = RAGFusionRetriever(
            self.qdrant_client,
            self.config.QDRANT_COLLECTION_NAME,
            self.embedding_model,
            self.reranker,
            self.credibility_scorer
        )
    
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60 + "\n")
    
    def _expand_query(self, question: str) -> List[str]:
        """Expand query into multiple variations"""
        try:
            chat_history = self.memory.get_history()
            prompt = QUERY_EXPANSION_PROMPT.format(question=question)
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            expanded = completion.choices[0].message.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip()]
            
            return [question] + expanded[:3]
        except:
            return [question]
        
# ============================================================================
# CHATBOT ASK METHOD - FIXED VERSION
# ============================================================================

    def ask(self, question: str) -> Tuple[str, Dict]:
        """Ask a question with improved pricing handling"""
        if not self.qdrant_client:
            return "❌ System not initialized.", {}
    
        try:
            question_lower = question.lower()
        
            # Detect if this is a pricing query
            is_pricing_query = any(
                term in question_lower 
                for term in ['cost', 'fee', 'price', 'how much', 'subsidy', 'payment', 'expensive']
            )
        
            # Detect certification level
            cert_level = None
            if any(term in question_lower for term in ['ca', 'certified associate', 'ihrp-ca', 'ihrp ca']):
                cert_level = 'CA'
            elif any(term in question_lower for term in ['cp', 'certified professional', 'ihrp-cp', 'ihrp cp']):
                cert_level = 'CP'
            elif any(term in question_lower for term in ['sp', 'senior professional', 'ihrp-sp', 'ihrp sp']):
                cert_level = 'SP'
        
            # For generic pricing questions, don't filter by level
            if is_pricing_query and not cert_level:
                print("💰 Generic pricing question detected - will retrieve all levels")
                cert_level = None
            elif cert_level:
                print(f"🎯 Detected certification level: {cert_level}")
        
            # Expand query
            queries = self._expand_query(question)
            print(f"\n🔍 Generated {len(queries)} query variations")
        
            # RAG Fusion retrieval
            fusion_docs = self.fusion_retriever.retrieve_with_fusion(
                queries, 
                self.config.INITIAL_K,
                filter_cert_level=cert_level
            )
        
            # Rerank with credibility
            final_docs = self.fusion_retriever.rerank_with_credibility(
                question, 
                fusion_docs, 
                self.config.FINAL_K if not is_pricing_query else self.config.FINAL_K + 3  # More context for pricing
            )
        
            # Create context
            context = "\n\n".join([doc['content'] for doc in final_docs])
            chat_history = self.memory.get_history()
        
            # Generate answer
            prompt = ADVANCED_PROMPT.format(
                context=context,
                question=question,
                chat_history=chat_history
            )
        
            completion = self.openai_client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.LLM_TEMPERATURE
        )
        
            answer = completion.choices[0].message.content
        
            # Guardrails
            validation_result = {"is_valid": True, "confidence": 1.0}
            if self.config.ENABLE_GUARDRAILS:
                validation_result = self.guardrails.validate_response(question, answer)
                print(f"🛡️  Guardrails: Valid={validation_result['is_valid']}, "
                    f"Confidence={validation_result['confidence']:.2f}")
            
                if not validation_result['is_valid'] or validation_result['confidence'] < self.config.CONFIDENCE_THRESHOLD:
                    answer += "\n\n⚠️ **Note**: I have moderate confidence in this response. " \
                            "Please verify with IHRP at support@ihrp.sg"
        
            # Update memory
            self.memory.add_message('user', question)
            self.memory.add_message('assistant', answer)
        
            # Format response
            result = f"{answer}\n\n"
            result += "───────────────────────────────\n"
            result += "📚 **Sources (Ranked by Relevance + Credibility):**\n"
        
            seen_sources = set()
            source_count = 0
            sources_list = []
        
            for doc in final_docs[:5]:  # Show more sources for pricing
                source = doc['metadata'].get('source', 'Unknown')
                cert_level_meta = doc['metadata'].get('certification_level', 'unknown')
            
                if source not in seen_sources:
                    seen_sources.add(source)
                    source_count += 1
                    sources_list.append(source)
                
                    cred_score = self.credibility_scorer.get_credibility_score(source)
                    cred_indicator = "🟢" if cred_score >= 0.9 else "🟡" if cred_score >= 0.7 else "🟠"
                
                    display_source = source if len(source) < 70 else source[:67] + "..."
                    level_tag = f" [{cert_level_meta}]" if cert_level_meta != 'unknown' else ""
                    result += f"{source_count}. {cred_indicator} {display_source}{level_tag}\n"
        
            metadata = {
                "question": question,
                "answer": answer,
                "sources": sources_list,
                "detected_level": cert_level,
                "is_pricing_query": is_pricing_query,
                "validation": validation_result,
                "timestamp": datetime.now().isoformat()
            }
        
            return result, metadata
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", {}
    
    def add_feedback(self, metadata: Dict, rating: int, comment: str = ""):
        """Add user feedback"""
        self.feedback_manager.add_feedback(
            question=metadata.get("question", ""),
            answer=metadata.get("answer", ""),
            rating=rating,
            comment=comment,
            sources=json.dumps(metadata.get("sources", []))
        )
        print(f"✅ Feedback recorded: {rating}/5 stars")
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        return self.feedback_manager.get_statistics()
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("🧹 Conversation memory cleared")
    
    def load_existing(self):
        """Load existing setup"""
        print("📂 Loading existing setup...")
        
        try:
            # Load models
            print("🔢 Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            
            print("🎯 Loading reranker...")
            self.reranker = CrossEncoder(self.config.RERANKER_MODEL)
            
            # Connect to Qdrant
            if self.config.QDRANT_URL and self.config.QDRANT_API_KEY:
                self.qdrant_client = QdrantClient(
                    url=self.config.QDRANT_URL,
                    api_key=self.config.QDRANT_API_KEY
                )
                print("✅ Connected to Qdrant Cloud")
            else:
                self.qdrant_client = QdrantClient(path="./qdrant_local")
                print("✅ Connected to local Qdrant")
            
            # Check collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.config.QDRANT_COLLECTION_NAME for c in collections)
            
            if not collection_exists:
                print("❌ Collection not found. Please run setup() first.")
                return False
            
            # Initialize components
            if self.config.ENABLE_GUARDRAILS:
                self.guardrails = GuardrailsValidator(self.openai_client)
            
            self.fusion_retriever = RAGFusionRetriever(
                self.qdrant_client,
                self.config.QDRANT_COLLECTION_NAME,
                self.embedding_model,
                self.reranker,
                self.credibility_scorer
            )
            
            print("✅ Loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading: {e}")
            return False

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface(chatbot):
    """Create Gradio interface"""
    
    last_metadata = {}
    
    def chat_function(message, history):
        response, metadata = chatbot.ask(message)
        last_metadata['current'] = metadata
        return response
    
    def submit_feedback(rating, comment):
        if 'current' in last_metadata:
            chatbot.add_feedback(last_metadata['current'], int(rating), comment)
            return f"✅ Thank you! Feedback recorded: {int(rating)}/5 stars"
        return "⚠️ No recent query to rate"
    
    def show_stats():
        stats = chatbot.get_feedback_stats()
        return f"""📊 **Feedback Statistics**
        
Total Queries: {stats['total_queries']}
Average Rating: {stats['avg_rating']}/5.0 ⭐
Positive Feedback: {stats['positive_feedback']} 👍
Negative Feedback: {stats['negative_feedback']} 👎"""
    
    def clear_chat_memory():
        chatbot.clear_memory()
        return "🧹 Memory cleared! Starting fresh conversation."
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            """
            # IHRP Support Assistant
            
            **Advanced Features:**
            - 🔀 **RAG Fusion** - Multi-strategy retrieval
            - 🎯 **Smart Reranking** - Relevance + credibility scoring
            - 🛡️ **Guardrails** - Automatic validation
            - 💬 **Memory** - Context-aware conversations
            - 📊 **Feedback** - Continuous improvement
            - ☁️ **Qdrant** - Cloud or local vector DB
            
            **Tech Stack:** OpenAI API + Qdrant + Sentence-Transformers + Playwright
            """
        )
        
        chatbot_ui = gr.ChatInterface(
            fn=chat_function,
            examples=[
                "I have 10 years in Business management but no HR experience. Which certification?",
                "What's the difference between CA and CP?",
                "What are the eligibility requirements for IHRP-CP?",
                "How much does certification cost?",
                "Can my business experience count toward certification?",
                "I'm transitioning from business analyst to HR. What should I do?",
            ],
            retry_btn="🔄 Retry",
            undo_btn="↩️ Undo",
            clear_btn="🗑️ Clear Chat",
        )
        
        gr.Markdown("### 📝 Rate This Response")
        
        with gr.Row():
            rating_slider = gr.Slider(
                minimum=1, maximum=5, step=1, value=5,
                label="Rating (1=Poor, 5=Excellent)"
            )
            feedback_comment = gr.Textbox(
                label="Comments (Optional)",
                placeholder="What could be improved?"
            )
        
        with gr.Row():
            submit_feedback_btn = gr.Button("📤 Submit Feedback", variant="primary")
            feedback_status = gr.Textbox(label="Status", interactive=False)
        
        submit_feedback_btn.click(
            fn=submit_feedback,
            inputs=[rating_slider, feedback_comment],
            outputs=feedback_status
        )
        
        gr.Markdown("### ⚙️ Controls")
        
        with gr.Row():
            clear_memory_btn = gr.Button("🧹 Clear Memory", variant="secondary")
            stats_btn = gr.Button("📊 Statistics", variant="secondary")
            memory_status = gr.Textbox(label="Status", interactive=False, scale=2)
        
        clear_memory_btn.click(fn=clear_chat_memory, outputs=memory_status)
        stats_btn.click(fn=show_stats, outputs=memory_status)
        
        gr.Markdown(
            """
            ---
            ### 💡 Legend
            - 🟢 Official IHRP source (highest credibility)
            - 🟡 Verified FAQ/Dataset (high credibility)  
            - 🟠 Secondary source (moderate credibility)
            
            The chatbot remembers conversation context. Rate responses to help improve!
            """
        )
    
    return demo

# ============================================================================
# DEPLOYMENT FILES GENERATOR
# ============================================================================

def create_deployment_files():
    """Create all deployment files"""
    
    # requirements.txt
    requirements = """openai==1.12.0
qdrant-client==1.7.3
sentence-transformers==2.3.1
gradio==4.16.0
playwright==1.41.0
beautifulsoup4==4.12.3
python-dotenv==1.0.0
schedule==1.2.0
numpy==1.24.3"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Created requirements.txt")
    
    # .env.example
    env_example = """# Required
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx

# Optional: Use Qdrant Cloud (recommended for production)
QDRANT_URL=https://xxxxx.qdrant.io
QDRANT_API_KEY=xxxxxxxxxxxxx

# If not provided, uses local Qdrant storage"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    print("✅ Created .env.example")
    
    # Dockerfile
    dockerfile = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget gnupg && \\
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application
COPY . .

# Expose port
EXPOSE 7860

# Run
CMD ["python", "ultimate_ihrp_rag.py"]"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    print("✅ Created Dockerfile")
    
    # docker-compose.yml
    compose = """version: '3.8'

services:
  ihrp-chatbot:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    volumes:
      - ./feedback.db:/app/feedback.db
      - ./content_hashes.json:/app/content_hashes.json
      - ./qdrant_local:/app/qdrant_local
    restart: unless-stopped"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose)
    print("✅ Created docker-compose.yml")
    
    # README.md
    readme = """# Ultimate IHRP RAG Chatbot

## 🚀 Features
- No LangChain dependencies (pure implementation)
- RAG Fusion retrieval
- Source credibility scoring
- Guardrails validation
- Feedback system
- Cloud or local deployment

## 📦 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run
```bash
python ultimate_ihrp_rag.py
```

## 🐳 Docker Deployment
```bash
docker-compose up -d
```

## ☁️ Cloud Deployment

### Hugging Face Spaces
1. Create new Space at huggingface.co
2. Upload all files
3. Add secrets in Settings

### Railway
1. Connect GitHub repo
2. Add environment variables
3. Deploy

## 📊 Tech Stack
- OpenAI API (GPT-4o)
- Qdrant (vector database)
- Sentence-Transformers (embeddings)
- Cross-Encoder (reranking)
- Playwright (web scraping)
- Gradio (UI)

## 🔧 Configuration
Edit `Config` class in the code to customize:
- Models
- Retrieval parameters
- Auto-update schedule
- Credibility scoring"""
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme)
    print("✅ Created README.md")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    
    print("\n" + "🚀 "*30)
    print("ULTIMATE IHRP RAG CHATBOT")
    print("NO LANGCHAIN - PURE & STABLE")
    print("🚀 "*30 + "\n")
    
    config = Config()
    
    if not config.OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("💡 Create .env file with: OPENAI_API_KEY=sk-...")
        return
    
    # Create deployment files
    print("📝 Creating deployment files...")
    create_deployment_files()
    
    chatbot = UltimateIHRPChatbot(config)
    
    # Check if setup needed
    try:
        if config.QDRANT_URL:
            print("☁️  Attempting to connect to Qdrant Cloud...")
            if chatbot.load_existing():
                print("✅ Connected to existing collection")
            else:
                print("🆕 Creating new collection...")
                chatbot.setup()
        else:
            print("💾 Using local Qdrant storage")
            if os.path.exists("./qdrant_local") and os.listdir("./qdrant_local"):
                print("📂 Found existing local data")
                print("\nOptions:")
                print("  1. Load existing (fast)")
                print("  2. Rebuild from scratch")
                
                choice = input("\nChoice [1/2]: ").strip()
                
                if choice == "1":
                    chatbot.load_existing()
                else:
                    import shutil
                    shutil.rmtree("./qdrant_local")
                    chatbot.setup()
            else:
                print("🆕 No existing data found")
                chatbot.setup()
    
    except Exception as e:
        print(f"⚠️  Error during initialization: {e}")
        print("🔄 Running full setup...")
        chatbot.setup()
    
    print("\n🌐 Launching web interface...")
    print("\n📖 Deployment files created:")
    print("   ✅ requirements.txt")
    print("   ✅ .env.example")
    print("   ✅ Dockerfile")
    print("   ✅ docker-compose.yml")
    print("   ✅ README.md")
    
    demo = create_interface(chatbot)
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

def cli_mode():
    """CLI mode"""
    config = Config()
    chatbot = UltimateIHRPChatbot(config)
    
    if chatbot.load_existing():
        print("✅ Loaded existing system")
    else:
        print("🆕 Setting up...")
        chatbot.setup()
    
    print("\n" + "="*60)
    print("🚀 CLI Mode")
    print("="*60)
    print("Commands: quit | clear | stats | rate <1-5>\n")
    
    last_metadata = {}
    
    while True:
        question = input("\n❓ You: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question.lower() == 'clear':
            chatbot.clear_memory()
            print("🧹 Memory cleared!")
            continue
        
        if question.lower() == 'stats':
            stats = chatbot.get_feedback_stats()
            print(f"\n📊 Stats:")
            print(f"   Total: {stats['total_queries']}")
            print(f"   Avg: {stats['avg_rating']}/5.0")
            continue
        
        if question.lower().startswith('rate '):
            try:
                rating = int(question.split()[1])
                if 1 <= rating <= 5 and last_metadata:
                    chatbot.add_feedback(last_metadata, rating)
                else:
                    print("❌ Invalid")
            except:
                print("❌ Usage: rate <1-5>")
            continue
        
        if not question.strip():
            continue
        
        print("\n🤖 Assistant:")
        print("─" * 60)
        answer, metadata = chatbot.ask(question)
        last_metadata = metadata
        print(answer)
        print("─" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        cli_mode()
    else:
        main()