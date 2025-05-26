import json
import asyncio
import aiohttp
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
from datetime import datetime
import time
import argparse
from collections import defaultdict, Counter
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
import torch
import gc
import psutil

warnings.filterwarnings('ignore')

# Try to import required packages, install if missing
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    pip_command = [sys.executable, "-m", "pip", "install", "sentence-transformers", "scikit-learn", "pandas",
                   "matplotlib", "seaborn", "plotly", "tqdm", "torch", "psutil"]
    try:
        import subprocess

        subprocess.check_call(pip_command)
    except Exception as e:
        print(f"Failed to install packages: {e}")
        sys.exit(1)
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

# Configuration
KEYWORDS_FILE = 'keywords.txt'
TRAINING_SET_FILE = 'training_set.json'
VALIDATION_SET_FILE = 'validation_set.json'
EMBEDDINGS_CACHE_FILE_PREFIX = 'embeddings_cache'
RESULTS_DIR = 'RAG2/rag_results'
VISUALIZATIONS_DIR = 'RAG2/rag_visualizations'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Gemini API Configuration
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# LaBSE Model Configuration
LABSE_MODEL = {
    'id': 'labse',
    'name': 'sentence-transformers/LaBSE',
    'type': 'sentence-transformers',
    'description': 'Language-agnostic BERT for 109+ languages including Lithuanian',
    'dimensions': 768
}

# Parameter Grids
FOCUSED_PARAMETER_GRID = {
    'chunk_size': [350, 400, 450],
    'chunk_overlap': [50, 75, 100],
    'top_k_retrieval': [4, 5, 6],
    'similarity_threshold': [0.20, 0.25, 0.30]
}
ULTRA_FOCUSED_GRID = {
    'chunk_size': [400, 450],
    'chunk_overlap': [75, 100],
    'top_k_retrieval': [5, 6],
    'similarity_threshold': [0.23, 0.25, 0.27]
}
QUICK_TEST_GRID = {
    'chunk_size': [400, 500],
    'chunk_overlap': [100],
    'top_k_retrieval': [5],
    'similarity_threshold': [0.25]
}

# Global model instance (shared across threads)
GLOBAL_MODEL = None
MODEL_LOCK = threading.Lock()
DEVICE = None


def initialize_global_model():
    """Initialize the global model with GPU support if available."""
    global GLOBAL_MODEL, DEVICE

    # Check CUDA availability
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        DEVICE = 'cpu'
        print("⚠️ CUDA not available, using CPU")

    with MODEL_LOCK:
        if GLOBAL_MODEL is None:
            print(f"Loading LaBSE model on {DEVICE}...")
            GLOBAL_MODEL = SentenceTransformer(LABSE_MODEL['name'], device=DEVICE)
            # Set model to eval mode and optimize
            GLOBAL_MODEL.eval()
            if DEVICE == 'cuda':
                # Enable GPU optimizations
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()

    return GLOBAL_MODEL


def get_memory_info():
    """Get current memory usage info."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / 1024 ** 3  # GB

    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        return f"RAM: {ram_usage:.2f}GB, GPU: {gpu_usage:.2f}/{gpu_reserved:.2f}GB"
    return f"RAM: {ram_usage:.2f}GB"


@dataclass
class DocumentChunk:
    text: str
    original_persons: List[str]
    anonymized_persons: List[str]
    source_url: str
    title: str
    chunk_id: int
    replacements: List[Dict] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class ClassificationResult:
    item_index: int
    param_config: str
    true_label: str
    predicted_label: str
    confidence: str
    explanation: str
    retrieval_scores: List[float]
    processing_time: float
    retrieval_time: float
    api_time: float
    all_ground_truth: List[str]
    title: str
    url: str
    anonymized_text_length: int
    timestamp: str
    chunk_size: int
    chunk_overlap: int
    top_k_retrieval: int
    similarity_threshold: float


@dataclass
class ParameterConfiguration:
    chunk_size: int
    chunk_overlap: int
    top_k_retrieval: int
    similarity_threshold: float

    def to_string(self):
        return f"chunk{self.chunk_size}_overlap{self.chunk_overlap}_topk{self.top_k_retrieval}_sim{self.similarity_threshold:.2f}"

    def to_dict(self):
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k_retrieval': self.top_k_retrieval,
            'similarity_threshold': self.similarity_threshold
        }


class OptimizedRAGKnowledgeBase:
    def __init__(self, param_config: ParameterConfiguration):
        self.param_config = param_config
        self.model_id = LABSE_MODEL['id']
        self.model_name = LABSE_MODEL['name']

        # Use global model instead of creating new instance
        self.encoder = GLOBAL_MODEL
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False
        self.person_mapping = {}
        self.build_stats = {}

        self.param_cache_file = os.path.join(
            RESULTS_DIR,
            f"cache_kb_{self.param_config.to_string()}.pkl"
        )

    def extract_person_mappings(self, replacements: List[Dict]) -> Dict[str, str]:
        person_map = {}
        if replacements:
            for replacement in replacements:
                if replacement.get('type') == 'PERSON':
                    original = replacement.get('original', '')
                    replacement_id = replacement.get('replacement', '')
                    if original and replacement_id:
                        person_map[replacement_id] = original
        return person_map

    def chunk_text(self, text: str) -> List[str]:
        chunk_size = self.param_config.chunk_size

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        if not sentences:
            return self.fallback_chunk(text, chunk_size, self.param_config.chunk_overlap)

        current_chunk_text = ""
        for i, sentence in enumerate(sentences):
            if len(current_chunk_text) + len(sentence) + 1 <= chunk_size:
                current_chunk_text += sentence + " "
            else:
                if current_chunk_text.strip():
                    chunks.append(current_chunk_text.strip())
                current_chunk_text = sentence + " "

        if current_chunk_text.strip():
            chunks.append(current_chunk_text.strip())

        if not chunks or (len(chunks) == 1 and len(chunks[0]) > chunk_size * 1.5):
            return self.fallback_chunk(text, chunk_size, self.param_config.chunk_overlap)

        return chunks

    def fallback_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                actual_end = text.rfind(' ', max(start, end - 50), end)
                if actual_end != -1:
                    end = actual_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            start = end - overlap
            if start <= (end - chunk_size + overlap):
                start = end - chunk_size + overlap + 1 if overlap < chunk_size else end
            if start >= end:
                start = end

        return chunks

    def build_knowledge_base(self, validation_data: List[Dict], use_embeddings_cache: bool = True):
        build_start_time = time.time()
        current_process_id = os.getpid()

        if use_embeddings_cache and os.path.exists(self.param_cache_file):
            try:
                with open(self.param_cache_file, 'rb') as f:
                    param_cache = pickle.load(f)
                if param_cache.get('param_config_dict') == self.param_config.to_dict():
                    self.chunks = param_cache['chunks']
                    self.embeddings = param_cache['embeddings']
                    self.person_mapping = param_cache['person_mapping']
                    self.build_stats = param_cache.get('build_stats', {})
                    self.is_built = True
                    self.build_stats['load_from_cache_time'] = time.time() - build_start_time
                    self.build_stats['status'] = f"Loaded from cache"
                    return
            except Exception as e:
                print(f"Failed to load cache {self.param_cache_file}: {e}")

        self._rebuild_chunks_with_params(validation_data)

        if self.is_built and use_embeddings_cache:
            try:
                self.build_stats['rebuild_time'] = time.time() - build_start_time
                param_cache_data = {
                    'param_config_dict': self.param_config.to_dict(),
                    'chunks': self.chunks,
                    'embeddings': self.embeddings,
                    'person_mapping': self.person_mapping,
                    'build_stats': self.build_stats,
                    'created_at': datetime.now().isoformat()
                }
                with open(self.param_cache_file, 'wb') as f:
                    pickle.dump(param_cache_data, f)
            except Exception as e:
                print(f"WARNING - Failed to save cache {self.param_cache_file}: {e}")

    def _rebuild_chunks_with_params(self, validation_data: List[Dict]):
        rebuild_start_time = time.time()

        self.chunks = []
        self.person_mapping = {}
        all_texts_for_embedding = []
        current_chunk_id = 0

        # Chunking Phase
        chunking_phase_start_time = time.time()
        desc_chunking = f"Chunking Docs (Cfg: {self.param_config.chunk_size}/{self.param_config.chunk_overlap})"
        for item in tqdm(validation_data, desc=desc_chunking, unit="doc", leave=False):
            try:
                anonymized_data = item.get('anonymized', {})
                full_text = anonymized_data.get('full_text', '')
                keywords = anonymized_data.get('keywords', [])
                title = anonymized_data.get('title', 'N/A')
                url = anonymized_data.get('url', 'N/A')
                replacements = item.get('replacements', [])

                if not full_text.strip():
                    continue

                item_person_map = self.extract_person_mappings(replacements)
                self.person_mapping.update(item_person_map)

                anonymized_persons_in_doc = [pid for pid in item_person_map.keys() if pid in full_text]

                text_chunks = self.chunk_text(full_text)

                for chunk_text_content in text_chunks:
                    chunk_anonymized_persons = [p for p in anonymized_persons_in_doc if p in chunk_text_content]
                    chunk = DocumentChunk(
                        text=chunk_text_content,
                        original_persons=keywords,
                        anonymized_persons=chunk_anonymized_persons,
                        source_url=url,
                        title=title,
                        chunk_id=current_chunk_id,
                        replacements=replacements
                    )
                    self.chunks.append(chunk)
                    all_texts_for_embedding.append(chunk_text_content)
                    current_chunk_id += 1
            except Exception as e:
                print(f"Error processing item during chunking: {e}")
                continue

        chunking_duration = time.time() - chunking_phase_start_time
        self.build_stats['chunking_time'] = chunking_duration
        self.build_stats['num_documents_processed'] = len(validation_data)
        self.build_stats['num_chunks_generated'] = len(all_texts_for_embedding)

        # Embedding Phase with GPU optimization
        if all_texts_for_embedding:
            embedding_phase_start_time = time.time()
            embeddings_list = []

            # Optimized batch size for GPU
            batch_size = 64 if DEVICE == 'cuda' else 32

            desc_embedding = f"Embedding (GPU: {DEVICE == 'cuda'})"
            for i in tqdm(range(0, len(all_texts_for_embedding), batch_size), desc=desc_embedding, unit="batch",
                          leave=False):
                batch_texts = all_texts_for_embedding[i:i + batch_size]
                try:
                    with torch.no_grad():  # Disable gradient computation for inference
                        batch_embeddings = self.encoder.encode(
                            batch_texts,
                            show_progress_bar=False,
                            batch_size=batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True  # Normalize for cosine similarity
                        )
                    embeddings_list.extend(batch_embeddings)

                    # Clear GPU cache periodically
                    if DEVICE == 'cuda' and i % (batch_size * 10) == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error encoding batch {i // batch_size}: {e}")
                    # Add zero embeddings as fallback
                    embeddings_list.extend(np.zeros((len(batch_texts), LABSE_MODEL['dimensions'])))

            if len(embeddings_list) == len(all_texts_for_embedding):
                self.embeddings = np.array(embeddings_list)
                self.is_built = True
            else:
                print(
                    f"WARNING - Embedding count mismatch. Expected {len(all_texts_for_embedding)}, got {len(embeddings_list)}")
                self.is_built = False

            embedding_duration = time.time() - embedding_phase_start_time
            self.build_stats['embedding_time'] = embedding_duration

            # Clear GPU cache after embedding
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        else:
            self.is_built = False

        self.build_stats['total_rebuild_time_internal'] = time.time() - rebuild_start_time

    def semantic_search(self, query: str) -> Tuple[List[Tuple[DocumentChunk, float]], float]:
        start_time = time.time()
        if not self.is_built or not self.chunks or self.embeddings is None or self.embeddings.shape[0] == 0:
            return [], 0.0
        if self.embeddings.shape[0] != len(self.chunks):
            print(f"Warning: Mismatch between embeddings and chunks")
            return [], 0.0

        QUERY_MAX_CHAR_LENGTH = 30000
        effective_query = query
        if len(query) > QUERY_MAX_CHAR_LENGTH:
            effective_query = query[:QUERY_MAX_CHAR_LENGTH]

        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [effective_query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        # Use numpy for CPU computation of cosine similarity (faster than sklearn for normalized vectors)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        num_candidates_to_consider = min(len(similarities), self.param_config.top_k_retrieval * 2)
        if num_candidates_to_consider == 0:
            return [], time.time() - start_time

        top_indices = np.argpartition(similarities, -num_candidates_to_consider)[-num_candidates_to_consider:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= self.param_config.similarity_threshold:
                results.append((self.chunks[idx], float(similarity_score)))
            if len(results) >= self.param_config.top_k_retrieval:
                break

        search_time = time.time() - start_time
        return results, search_time

    def hybrid_retrieval(self, query: str, target_persons: List[str]) -> Tuple[
        List[Tuple[DocumentChunk, float, Dict]], float]:
        semantic_results, search_time = self.semantic_search(query)
        if not semantic_results:
            return [], search_time

        boosted_results = []
        for chunk, similarity in semantic_results:
            boost = 0.0
            boost_details = {
                'semantic_similarity': similarity,
                'person_match_boost': 0.0,
                'anonymized_ref_boost': 0.0,
                'length_boost': 0.0
            }

            for target_person in target_persons:
                if any(target_person.lower() in op.lower() for op in chunk.original_persons):
                    person_boost = 0.20
                    boost += person_boost
                    boost_details['person_match_boost'] = max(boost_details['person_match_boost'], person_boost)

            for anon_id, orig_name in self.person_mapping.items():
                if anon_id in chunk.text:
                    if any(target_person.lower() in orig_name.lower() for target_person in target_persons):
                        anon_boost = 0.15
                        boost += anon_boost
                        boost_details['anonymized_ref_boost'] = max(boost_details['anonymized_ref_boost'], anon_boost)

            length_boost = min(len(chunk.text) / 2000.0, 0.1)
            boost += length_boost
            boost_details['length_boost'] = length_boost

            final_score = min(similarity + boost, 1.0)
            boosted_results.append((chunk, final_score, boost_details))

        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:self.param_config.top_k_retrieval], search_time

    def get_context_for_prompt(self, query: str, target_persons: List[str]) -> Tuple[str, List[Dict], float]:
        retrieved_chunks_with_scores, retrieval_time = self.hybrid_retrieval(query, target_persons)
        if not retrieved_chunks_with_scores:
            return "No relevant contextual information found in knowledge base.", [], retrieval_time

        context_parts = []
        retrieval_details_for_log = []
        for i, (chunk, relevance_score, boost_details) in enumerate(retrieved_chunks_with_scores, 1):
            original_persons_str = ", ".join(chunk.original_persons) if chunk.original_persons else "Unknown"
            anonymized_persons_str = ", ".join(chunk.anonymized_persons) if chunk.anonymized_persons else "None"
            context_parts.append(f"""
Context {i} (Relevance Score: {relevance_score:.3f}):
  - Original persons mentioned (document level): {original_persons_str}
  - Anonymized references in chunk: {anonymized_persons_str}
  - Source: {chunk.title} (URL: {chunk.source_url})
  - Text: "{chunk.text}"
""")
            retrieval_details_for_log.append({
                'context_rank': i,
                'relevance_score': relevance_score,
                'original_persons_doc': chunk.original_persons,
                'anonymized_persons_chunk': chunk.anonymized_persons,
                'source_title': chunk.title,
                'source_url': chunk.source_url,
                'text_length': len(chunk.text),
                'boost_details': boost_details
            })
        return "\n".join(context_parts), retrieval_details_for_log, retrieval_time


# Async API call for better concurrency
async def call_gemini_api_async(session: aiohttp.ClientSession, prompt_text: str, api_key: str) -> Tuple[str, Dict]:
    if not api_key:
        return "Error: No API key provided for Gemini.", {'error': 'no_api_key', 'status_code': -1}

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.1, "topK": 1, "topP": 1,
            "maxOutputTokens": 2000,
        }
    }
    url = f"{GEMINI_API_BASE_URL}?key={api_key}"

    try:
        async with session.post(url, headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=90)) as response:
            response.raise_for_status()
            result = await response.json()

            if (result.get("candidates") and
                    result["candidates"][0].get("content") and
                    result["candidates"][0]["content"].get("parts") and
                    result["candidates"][0]["content"]["parts"][0].get("text")):
                return result["candidates"][0]["content"]["parts"][0]["text"], {'status_code': response.status}
            else:
                return "Error: Unexpected API response structure from Gemini.", {
                    'error': 'unexpected_response',
                    'status_code': response.status,
                    'response_text': result
                }
    except aiohttp.ClientError as e:
        return f"Error: API call failed: {str(e)}", {'error': str(e), 'status_code': -1}
    except asyncio.TimeoutError:
        return "Error: Gemini API call timed out after 90 seconds.", {'error': 'timeout', 'status_code': -1}
    except Exception as e:
        return f"Error: An unexpected error occurred calling Gemini: {str(e)}", {'error': str(e), 'status_code': -1}


def parse_llm_response(response_text: str) -> Tuple[str, str, str]:
    identified_person = "Uncertain"
    explanation = "No explanation provided."
    confidence = "N/A"
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("1. identified person's name:") or \
                    line_lower.startswith("1. identified person:") or \
                    line_lower.startswith("atsakymas:") or \
                    line_lower.startswith("asmuo:"):
                identified_person = line.split(":", 1)[1].strip().replace("[", "").replace("]", "")
            elif line_lower.startswith("2. explanation:") or \
                    line_lower.startswith("paaiškinimas:") or \
                    line_lower.startswith("pagrindimas:"):
                explanation = line.split(":", 1)[1].strip()
            elif line_lower.startswith("3. confidence score:") or \
                    line_lower.startswith("3. confidence:") or \
                    line_lower.startswith("pasitikėjimas:") or \
                    line_lower.startswith("tikimybė:"):
                confidence = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return identified_person, explanation, confidence


async def process_single_test_item_async(
        test_item: Dict,
        target_persons: List[str],
        rag_kb: OptimizedRAGKnowledgeBase,
        api_key: str,
        item_idx: int,
        session: aiohttp.ClientSession
) -> Optional[ClassificationResult]:
    overall_item_start_time = time.time()

    try:
        anonymized_data = test_item.get('anonymized', {})
        anonymized_text = anonymized_data.get('full_text', '')
        title = anonymized_data.get('title', 'N/A')
        ground_truth_keywords = anonymized_data.get('keywords', [])
        url = anonymized_data.get('url', 'N/A')

        if not anonymized_text or not ground_truth_keywords:
            return None

        # Get RAG context
        retrieval_start_time = time.time()
        context_str, retrieval_details, _ = rag_kb.get_context_for_prompt(anonymized_text, ground_truth_keywords)
        retrieval_time = time.time() - retrieval_start_time
        retrieval_scores = [detail['relevance_score'] for detail in retrieval_details]

        prompt = f"""You are an expert AI system specialized in identifying anonymized Lithuanian political figures from news articles.

TASK: Analyze the anonymized Lithuanian text below and determine which political figure is most likely being discussed. The anonymized text may refer to one or more of the target individuals. Your goal is to identify the primary subject.

ANONYMIZED TEXT TO ANALYZE:
"{anonymized_text}"

RETRIEVED CONTEXTUAL INFORMATION (ranked by semantic relevance, may or may not be relevant):
{context_str}

LIST OF POTENTIAL TARGET INDIVIDUALS: {', '.join(target_persons)} 
(The anonymized person is one of these individuals. If uncertain after analysis, state "Uncertain".)

ANALYSIS GUIDELINES:
- The text uses placeholders like GROUPED_PERSON_X for people and GROUPED_OCCUPATION_Y for positions/roles.
- Compare patterns in roles, activities, topics, and political contexts mentioned in the anonymized text with the retrieved contextual information and your knowledge of Lithuanian politics.
- Pay attention to the relevance scores of retrieved contexts; higher scores suggest greater similarity of context.
- Base your decision on the evidence in the provided anonymized text and the supporting context. Do not invent information.
- If the text or context is insufficient to make a confident identification from the target list, select "Uncertain".

Please provide your analysis in this exact format:
1. Identified person's name: [Name from target list, or "Uncertain"]
2. Explanation: [Detailed reasoning comparing the anonymized text with retrieved contexts and known roles of the individuals. Justify your choice or uncertainty.]
3. Confidence score: [High, Medium, or Low]"""

        api_start_time = time.time()
        llm_response_text, api_stats = await call_gemini_api_async(session, prompt, api_key)
        api_time = time.time() - api_start_time

        identified_person, explanation, confidence = parse_llm_response(llm_response_text)

        true_label = ground_truth_keywords[0] if ground_truth_keywords else 'Unknown'

        result = ClassificationResult(
            item_index=item_idx,
            param_config=rag_kb.param_config.to_string(),
            true_label=true_label,
            predicted_label=identified_person,
            confidence=confidence,
            explanation=explanation,
            retrieval_scores=retrieval_scores,
            processing_time=(time.time() - overall_item_start_time),
            retrieval_time=retrieval_time,
            api_time=api_time,
            all_ground_truth=ground_truth_keywords,
            title=title,
            url=url,
            anonymized_text_length=len(anonymized_text),
            timestamp=datetime.now().isoformat(),
            chunk_size=rag_kb.param_config.chunk_size,
            chunk_overlap=rag_kb.param_config.chunk_overlap,
            top_k_retrieval=rag_kb.param_config.top_k_retrieval,
            similarity_threshold=rag_kb.param_config.similarity_threshold
        )
        return result

    except Exception as e:
        print(f"Error processing item {item_idx}: {e}")
        return None


async def process_configuration_async(
        param_config: ParameterConfiguration,
        testing_data: List[Dict],
        validation_data_for_kb: List[Dict],
        target_persons: List[str],
        api_key: str
) -> List[ClassificationResult]:
    # Build knowledge base once for this configuration
    rag_kb = OptimizedRAGKnowledgeBase(param_config)
    rag_kb.build_knowledge_base(validation_data_for_kb, use_embeddings_cache=True)

    if not rag_kb.is_built:
        print(f"KB not built for {param_config.to_string()}")
        return []

    results = []

    # Process items in batches to avoid overwhelming the API
    batch_size = 30 # Adjust based on API rate limits

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(testing_data), batch_size), desc=f"Testing {param_config.to_string()}",
                      unit="batch"):
            batch = testing_data[i:i + batch_size]
            tasks = []

            for j, test_item in enumerate(batch):
                task = process_single_test_item_async(
                    test_item, target_persons, rag_kb, api_key, i + j, session
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r is not None])

            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(testing_data):
                await asyncio.sleep(0.5)

    # Clean up
    del rag_kb
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return results


def load_keywords(filepath: str) -> List[str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]
        return keywords
    except FileNotFoundError:
        print(f"Error: Keywords file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error loading keywords from {filepath}: {e}")
        return []


def load_json_data(filepath: str) -> List[Dict]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON data file not found at {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Error loading JSON data from {filepath}: {e}")
        return []


def calculate_metrics_for_config(results: List[ClassificationResult], target_persons: List[str]) -> Dict:
    if not results:
        return {}

    y_true = []
    y_pred = []
    all_classes = sorted(list(set(target_persons + ['Uncertain'])))

    for res in results:
        true_label = res.true_label if res.true_label in all_classes else 'Uncertain'
        pred_label = res.predicted_label if res.predicted_label in all_classes else 'Uncertain'
        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        return {}

    accuracy = accuracy_score(y_true, y_pred)
    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        balanced_acc = 0.0 if len(set(y_true)) > 1 else accuracy

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_classes, average='weighted', zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=all_classes if len(set(y_true)) > 1 else None)

    valid_results = [r for r in results if not r.predicted_label.startswith("Error:")]
    avg_retrieval_time = np.mean([r.retrieval_time for r in valid_results]) if valid_results else 0
    avg_processing_time = np.mean([r.processing_time for r in valid_results]) if valid_results else 0
    avg_api_time = np.mean([r.api_time for r in valid_results]) if valid_results else 0
    avg_retrieval_score = np.mean(
        [np.mean(r.retrieval_scores) if r.retrieval_scores else 0 for r in valid_results]
    ) if valid_results else 0

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_w,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'mcc': mcc,
        'cohen_kappa': kappa,
        'correct_predictions': sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp),
        'total_predictions': len(y_true),
        'avg_retrieval_time': avg_retrieval_time,
        'avg_processing_time': avg_processing_time,
        'avg_api_time': avg_api_time,
        'avg_retrieval_score': avg_retrieval_score
    }


def create_parameter_comparison_heatmap(summary_df: pd.DataFrame, metric: str, save_path: str):
    if summary_df.empty or metric not in summary_df.columns:
        print(f"Cannot create heatmap for {metric}, data missing or empty.")
        return

    summary_df[metric] = pd.to_numeric(summary_df[metric], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    try:
        pivot1_data = summary_df.pivot_table(
            values=metric, index='chunk_size', columns='top_k_retrieval', aggfunc='mean'
        )
        if not pivot1_data.empty:
            sns.heatmap(pivot1_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[0], cbar=True)
            axes[0].set_title(f'{metric.replace("_", " ").title()} by Chunk Size vs Top K')
        else:
            axes[0].text(0.5, 0.5, 'Not enough data for Chunk Size vs Top K pivot', ha='center', va='center')
            axes[0].set_title('Chunk Size vs Top K')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Error pivoting: {e}', ha='center', va='center')
        axes[0].set_title('Chunk Size vs Top K (Error)')

    try:
        pivot2_data = summary_df.pivot_table(
            values=metric, index='chunk_overlap', columns='similarity_threshold', aggfunc='mean'
        )
        if not pivot2_data.empty:
            sns.heatmap(pivot2_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], cbar=True)
            axes[1].set_title(f'{metric.replace("_", " ").title()} by Overlap vs Similarity Threshold')
        else:
            axes[1].text(0.5, 0.5, 'Not enough data for Overlap vs Sim Threshold pivot', ha='center', va='center')
            axes[1].set_title('Overlap vs Similarity Threshold')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Error pivoting: {e}', ha='center', va='center')
        axes[1].set_title('Overlap vs Similarity Threshold (Error)')

    plt.suptitle(f'Parameter Grid Search Heatmaps for {metric.replace("_", " ").title()}', fontsize=16,
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def create_top_configurations_chart(summary_df: pd.DataFrame, save_path: str):
    if summary_df.empty:
        print("Summary DataFrame is empty. Cannot create top configurations chart.")
        return

    top_n = min(15, len(summary_df))
    if top_n == 0:
        print("No configurations to plot in top configurations chart.")
        return

    top_configs = summary_df.nlargest(top_n, 'f1_weighted').reset_index()

    top_configs['config_label'] = [
        f"Cfg {i + 1}\n({row.param_config.split('_sim')[0]})" for i, row in top_configs.iterrows()
    ]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Top {top_n} Configurations by F1 Score (Weighted)', 'Top 5 Metrics Breakdown')
    )

    fig.add_trace(
        go.Bar(
            x=top_configs['config_label'],
            y=top_configs['f1_weighted'],
            name='F1 Weighted',
            text=top_configs['f1_weighted'].apply(lambda x: f'{x:.3f}'),
            textposition='outside'
        ),
        row=1, col=1
    )

    top_5_configs = top_configs.head(min(5, top_n))
    if not top_5_configs.empty:
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'mcc', 'cohen_kappa']
        for metric in metrics_to_plot:
            if metric in top_5_configs.columns:
                fig.add_trace(
                    go.Bar(
                        name=metric.replace("_", " ").title(),
                        x=top_5_configs['config_label'],
                        y=top_5_configs[metric],
                        text=top_5_configs[metric].apply(lambda x: f'{x:.3f}'),
                        textposition='auto'
                    ),
                    row=2, col=1
                )

    fig.update_layout(
        height=800,
        title_text='RAG Parameter Optimization Analysis',
        xaxis_tickangle=-45,
        barmode='group',
        legend_title_text='Metric',
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Score", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, 1.05])

    try:
        fig.write_image(save_path, width=1200, height=800)
    except Exception as e:
        print(f"Failed to save Plotly chart as image: {e}")
        html_save_path = save_path.replace(".png", ".html")
        fig.write_html(html_save_path)
        print(f"Plotly chart saved as HTML: {html_save_path}")


async def main_async():
    parser = argparse.ArgumentParser(description='Optimized RAG Parameter Optimization with GPU Support')
    parser.add_argument('--grid', type=str, default='focused', choices=['focused', 'ultra-focused', 'quick'],
                        help='Parameter grid to use.')
    parser.add_argument('--test', action='store_true', help='Test mode (process only a subset of data)')
    parser.add_argument('--test-count', type=int, default=5, help='Number of examples in test mode')
    args = parser.parse_args()

    if args.grid == 'quick':
        param_grid, grid_name = QUICK_TEST_GRID, "QUICK TEST"
    elif args.grid == 'ultra-focused':
        param_grid, grid_name = ULTRA_FOCUSED_GRID, "ULTRA-FOCUSED"
    else:
        param_grid, grid_name = FOCUSED_PARAMETER_GRID, "FOCUSED (default)"

    max_examples_test_mode = args.test_count if args.test else None

    total_combinations = np.prod([len(v) for v in param_grid.values()])

    print("🚀 Starting Optimized RAG Parameter Optimization with GPU Support")
    print(f"🗓️ Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Parameter grid: {grid_name} ({total_combinations} raw combinations)")
    print(f"💾 Initial Memory: {get_memory_info()}")
    if args.test:
        print(f"🧪 TEST MODE: Processing up to {max_examples_test_mode} examples per config.")
    print("=" * 70)

    # Initialize global model with GPU support
    initialize_global_model()
    print(f"💾 After model load: {get_memory_info()}")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("dotenv package not found, API key must be set as environment variable.")

    overall_start_time = time.time()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("🔑 Enter your Gemini API key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return

    print("\n📂 Loading data files...")
    target_persons = load_keywords(KEYWORDS_FILE)
    validation_data_for_kb = load_json_data(TRAINING_SET_FILE)
    testing_data_for_eval = load_json_data(VALIDATION_SET_FILE)

    if not target_persons or not validation_data_for_kb or not testing_data_for_eval:
        print("❌ Critical data missing. Exiting.")
        return

    if args.test:
        testing_data_for_eval = testing_data_for_eval[:max_examples_test_mode]
        print(f"🧪 Test mode: Using {len(testing_data_for_eval)} examples for evaluation.")

    print(f"✅ Target persons: {len(target_persons)} loaded.")
    print(f"✅ Knowledge Base Source: {len(validation_data_for_kb)} documents.")
    print(f"✅ Evaluation Data: {len(testing_data_for_eval)} items.")

    param_configurations_list = []
    for p_vals in product(*param_grid.values()):
        config = ParameterConfiguration(**dict(zip(param_grid.keys(), p_vals)))
        if config.chunk_overlap < config.chunk_size:
            param_configurations_list.append(config)

    print(f"\n📊 Valid parameter configurations to test: {len(param_configurations_list)}")
    if not param_configurations_list:
        print("No valid configurations. Exiting.")
        return

    all_results_data = []
    configuration_summary_metrics = []

    for config_idx, current_param_config in enumerate(param_configurations_list):
        config_loop_start_time = time.time()
        print(f"\n{'=' * 30} Configuration {config_idx + 1}/{len(param_configurations_list)} {'=' * 30}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing: {current_param_config.to_string()}")
        print(f"💾 Memory before config: {get_memory_info()}")

        # Process configuration
        current_config_results = await process_configuration_async(
            current_param_config,
            testing_data_for_eval,
            validation_data_for_kb,
            target_persons,
            api_key
        )

        if current_config_results:
            all_results_data.extend(current_config_results)
            metrics = calculate_metrics_for_config(current_config_results, target_persons)
            metrics_entry = {
                **current_param_config.to_dict(),
                'param_config': current_param_config.to_string(),
                **metrics
            }
            configuration_summary_metrics.append(metrics_entry)

            print(f"\n📊 Config {current_param_config.to_string()} Performance:")
            print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}, F1 (w): {metrics.get('f1_weighted', 0):.3f}")
            print(f"   Avg Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
        else:
            print(f"⚠️ No results for configuration {current_param_config.to_string()}.")

        print(f"Config processing time: {(time.time() - config_loop_start_time) / 60:.2f} minutes.")
        print(f"💾 Memory after config: {get_memory_info()}")

    print("\n🏁 All configurations processed. Generating final reports...")

    # Save results
    if all_results_data:
        results_df_data = [res.__dict__ for res in all_results_data]
        results_df = pd.DataFrame(results_df_data)
        results_df.to_csv(
            os.path.join(RESULTS_DIR, f'all_classification_results_{grid_name.replace(" ", "")}.csv'),
            index=False
        )
        print(f"💾 Results saved to: {RESULTS_DIR}/")

    if configuration_summary_metrics:
        summary_df = pd.DataFrame(configuration_summary_metrics)
        summary_df = summary_df.sort_values('f1_weighted', ascending=False)
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, f'param_summary_{grid_name.replace(" ", "")}.csv'),
            index=False
        )

        print("\n📊 Creating visualizations...")
        for metric_to_plot in ['f1_weighted', 'accuracy', 'balanced_accuracy', 'mcc']:
            if metric_to_plot in summary_df.columns:
                heatmap_path = os.path.join(
                    VISUALIZATIONS_DIR,
                    f'heatmap_{metric_to_plot}_{grid_name.replace(" ", "")}.png'
                )
                create_parameter_comparison_heatmap(summary_df, metric_to_plot, heatmap_path)

        top_configs_chart_path = os.path.join(
            VISUALIZATIONS_DIR,
            f'top_configs_plot_{grid_name.replace(" ", "")}.png'
        )
        create_top_configurations_chart(summary_df, top_configs_chart_path)

        print("\n🏆 OPTIMIZATION RESULTS SUMMARY (Top 5 by F1 Weighted)")
        print("=" * 70)
        top_5_configs_summary = summary_df.head(5)
        for idx, row in top_5_configs_summary.iterrows():
            print(f"\nRank {idx + 1}: {row['param_config']}")
            print(f"  F1 (w): {row.get('f1_weighted', 0):.3f}, Acc: {row.get('accuracy', 0):.3f}")

    print(f"\n⏱️ Total Optimization Time: {(time.time() - overall_start_time) / 60:.2f} minutes.")
    print(f"💾 Final Memory: {get_memory_info()}")
    print("\n✅ Parameter optimization complete!")


def main():
    # Run the async main function
    asyncio.run(main_async())


if __name__ == '__main__':
    main()