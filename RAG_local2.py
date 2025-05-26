import json
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import pickle
from datetime import datetime
import time
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, confusion_matrix
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import gc
import psutil
from tqdm import tqdm
import requests
import subprocess
import platform
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Import required packages
try:
    from sentence_transformers import SentenceTransformer
    import ollama
    # Removed Ragas and related Langchain imports
except ImportError:
    print("Installing required packages...")
    # Simplified pip install command, removing ragas, datasets, langchain-community, openai
    pip_command = [sys.executable, "-m", "pip", "install",
                   "sentence-transformers", "ollama", "numpy", "pandas",
                   "matplotlib", "seaborn", "scikit-learn", "plotly",
                   "torch", "psutil", "tqdm", "requests"]
    subprocess.check_call(pip_command)
    from sentence_transformers import SentenceTransformer
    import ollama
    # Removed Ragas and related Langchain imports from here as well

# Configuration
KEYWORDS_FILE = 'keywords.txt'
TRAINING_SET_FILE = 'training_set.json'
VALIDATION_SET_FILE = 'validation_set.json'
RESULTS_DIR = 'RAG_local/local_model_results'
VISUALIZATIONS_DIR = 'RAG_local/local_model_visualizations'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Optimal parameters from previous experiments
OPTIMAL_PARAMS = {
    'chunk_size': 350,
    'chunk_overlap': 50,
    'top_k_retrieval': 6,
    'similarity_threshold': 0.25
}

# Best prompt template
OPTIMAL_PROMPT_TEMPLATE = """You are an expert AI system specialized in identifying anonymized Lithuanian political figures from news articles.

TASK: Analyze the anonymized Lithuanian text below and determine which political figure is most likely being discussed. The anonymized text may refer to one or more of the target individuals. Your goal is to identify the primary subject.

ANONYMIZED TEXT TO ANALYZE:
"{text}"

RETRIEVED CONTEXTUAL INFORMATION (ranked by semantic relevance, may or may not be relevant):
{context}

LIST OF POTENTIAL TARGET INDIVIDUALS: {targets}
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

# LaBSE Model Configuration
LABSE_MODEL = {
    'name': 'sentence-transformers/LaBSE',
    'dimensions': 768
}

# Global variables
GLOBAL_ENCODER = None
DEVICE = None


@dataclass
class LocalModelConfig:
    """Configuration for a local model"""
    id: str
    name: str
    ollama_name: str
    size: str
    context_length: int
    temperature: float = 0.1
    max_tokens: int = 2000


# Define local models that actually exist in Ollama
LOCAL_MODELS = [
    LocalModelConfig(
        id="gemma-2b", name="Google Gemma 2B", ollama_name="gemma:2b",
        size="2B", context_length=8192,
    ),
    LocalModelConfig(
        id="llama3-8b", name="Llama 3 8B", ollama_name="llama3:8b",
        size="8B", context_length=8192,
    ),
    LocalModelConfig(
        id="mistral-7b", name="Mistral 7B Instruct", ollama_name="mistral:7b-instruct",
        size="7B", context_length=8192,
    ),
    LocalModelConfig(
        id="phi-2", name="Microsoft Phi-2", ollama_name="phi:2.7b",
        size="2.7B", context_length=2048,
    ),
]


# Removed RAGASMetrics dataclass

@dataclass
class LocalModelTestResult:
    item_index: int
    model_id: str
    model_name: str
    true_label: str
    predicted_label: str
    confidence: str
    explanation: str
    retrieval_scores: List[float]
    generation_time: float
    total_time: float
    response_text: str
    timestamp: str
    # Removed ragas_metrics field
    prompt_tokens: int = 0
    response_tokens: int = 0
    retrieved_contexts: List[str] = field(default_factory=list)  # Kept for potential debugging
    error: Optional[str] = None


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


class OptimizedRAGKnowledgeBase:
    """Knowledge base using optimal parameters with hybrid retrieval"""

    def __init__(self):
        self.chunk_size = OPTIMAL_PARAMS['chunk_size']
        self.chunk_overlap = OPTIMAL_PARAMS['chunk_overlap']
        self.top_k_retrieval = OPTIMAL_PARAMS['top_k_retrieval']
        self.similarity_threshold = OPTIMAL_PARAMS['similarity_threshold']
        self.encoder = GLOBAL_ENCODER
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False
        self.person_mapping = {}
        self.cache_file = os.path.join(RESULTS_DIR, "kb_cache_local_models.pkl")

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
        if len(text) <= self.chunk_size:
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
            return self.fallback_chunk(text, self.chunk_size, self.chunk_overlap)
        current_chunk_text = ""
        for sentence in sentences:
            if len(current_chunk_text) + len(sentence) + 1 <= self.chunk_size:
                current_chunk_text += sentence + " "
            else:
                if current_chunk_text.strip():
                    chunks.append(current_chunk_text.strip())
                current_chunk_text = sentence + " "
        if current_chunk_text.strip():
            chunks.append(current_chunk_text.strip())
        if not chunks and text:
            return self.fallback_chunk(text, self.chunk_size, self.chunk_overlap)
        return chunks

    def fallback_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                actual_end = text.rfind(' ', max(start, end - 50), end)
                if actual_end != -1 and actual_end > start:
                    end = actual_end
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start = end - overlap
            if start >= end:
                start = end
        return chunks

    def build_knowledge_base(self, validation_data: List[Dict], use_cache: bool = True):
        if use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.chunks = cache_data['chunks']
                self.embeddings = cache_data['embeddings']
                self.person_mapping = cache_data['person_mapping']
                self.is_built = True
                print("✅ Loaded knowledge base from cache")
                return
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}. Rebuilding knowledge base.")

        print("Building knowledge base with hybrid retrieval...")
        self.chunks = []
        self.person_mapping = {}
        all_texts = []
        for item in tqdm(validation_data, desc="Processing documents"):
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
            for chunk_id_counter, chunk_text in enumerate(text_chunks):
                chunk_anonymized_persons = [p for p in anonymized_persons_in_doc if p in chunk_text]
                chunk = DocumentChunk(
                    text=chunk_text, original_persons=keywords,
                    anonymized_persons=chunk_anonymized_persons,
                    source_url=url, title=title, chunk_id=len(self.chunks),
                    replacements=replacements
                )
                self.chunks.append(chunk)
                all_texts.append(chunk_text)
        if all_texts:
            print("Generating embeddings...")
            batch_size = 64 if DEVICE == "cuda" else 32
            embeddings_list = []
            for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding chunks"):
                batch = all_texts[i:i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.encoder.encode(
                        batch, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True
                    )
                embeddings_list.extend(batch_embeddings)
            self.embeddings = np.array(embeddings_list)
            self.is_built = True
            if use_cache:
                cache_data = {
                    'chunks': self.chunks, 'embeddings': self.embeddings,
                    'person_mapping': self.person_mapping
                }
                try:
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    print("✅ Saved knowledge base to cache")
                except Exception as e:
                    print(f"⚠️ Error saving knowledge base to cache: {e}")
        else:
            print("⚠️ No texts found to build embeddings. Knowledge base is empty.")
            self.is_built = False

    def semantic_search(self, query: str) -> Tuple[List[Tuple[DocumentChunk, float]], List[float]]:
        if not self.is_built or self.embeddings is None or len(self.chunks) == 0:
            print("⚠️ Knowledge base not built or empty. Cannot perform semantic search.")
            return [], []
        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [query], show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True
            )
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        num_candidates = min(len(similarities),
                             self.top_k_retrieval * 2 if len(similarities) > self.top_k_retrieval * 2 else len(
                                 similarities))
        if num_candidates == 0: return [], []
        top_indices = np.argpartition(similarities, -num_candidates)[-num_candidates:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        results = []
        scores = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                results.append((self.chunks[idx], float(similarities[idx])))
                scores.append(float(similarities[idx]))
            if len(results) >= self.top_k_retrieval:
                break
        return results, scores

    def hybrid_retrieval(self, query: str, target_persons: List[str]) -> Tuple[
        List[Tuple[DocumentChunk, float, Dict]], List[float]]:
        semantic_results, semantic_scores = self.semantic_search(query)
        if not semantic_results:
            return [], []
        boosted_results = []
        for (chunk, similarity) in semantic_results:
            boost = 0.0
            boost_details = {
                'semantic_similarity': similarity, 'person_match_boost': 0.0,
                'anonymized_ref_boost': 0.0, 'length_boost': 0.0
            }
            if chunk.original_persons:
                for target_person in target_persons:
                    if any(target_person.lower() in op.lower() for op in chunk.original_persons):
                        person_boost = 0.20
                        boost += person_boost
                        boost_details['person_match_boost'] = max(boost_details['person_match_boost'], person_boost)
                        break
            if chunk.anonymized_persons:
                for anon_id_in_chunk in chunk.anonymized_persons:
                    original_name_mapped = self.person_mapping.get(anon_id_in_chunk)
                    if original_name_mapped:
                        if any(target_person.lower() in original_name_mapped.lower() for target_person in
                               target_persons):
                            anon_boost = 0.15
                            boost += anon_boost
                            boost_details['anonymized_ref_boost'] = max(boost_details['anonymized_ref_boost'],
                                                                        anon_boost)
                            break
            length_boost = min(len(chunk.text) / 2000.0, 0.1)
            boost += length_boost
            boost_details['length_boost'] = length_boost
            final_score = min(similarity + boost, 1.0)
            boosted_results.append({'chunk': chunk, 'score': final_score, 'boost_details': boost_details})
        boosted_results.sort(key=lambda x: x['score'], reverse=True)
        final_results_tuples = [(res['chunk'], res['score'], res['boost_details']) for res in
                                boosted_results[:self.top_k_retrieval]]
        final_scores_list = [res['score'] for res in boosted_results[:self.top_k_retrieval]]
        return final_results_tuples, final_scores_list

    def get_context_for_prompt(self, query: str, target_persons: List[str]) -> Tuple[
        str, List[float], List[DocumentChunk], List[str]]:
        results_with_details, scores = self.hybrid_retrieval(query, target_persons)
        if not results_with_details:
            return "No relevant contextual information found.", [], [], []
        context_parts = []
        retrieved_chunks_objects = []
        contexts_text_list = []
        for i, (chunk, score, boost_details) in enumerate(results_with_details, 1):
            original_persons_str = ", ".join(chunk.original_persons) if chunk.original_persons else "Unknown"
            anonymized_persons_str = ", ".join(
                chunk.anonymized_persons) if chunk.anonymized_persons else "None in this chunk"
            context_parts.append(f"""
Context {i} (Relevance Score: {score:.3f}):
  - Original persons associated with source document: {original_persons_str}
  - Anonymized references found in this specific chunk: {anonymized_persons_str}
  - Source: {chunk.title} (URL: {chunk.source_url})
  - Text: "{chunk.text}"
""")
            retrieved_chunks_objects.append(chunk)
            contexts_text_list.append(chunk.text)
        return "\n".join(context_parts), scores, retrieved_chunks_objects, contexts_text_list


def initialize_global_encoder():
    global GLOBAL_ENCODER, DEVICE
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("⚠️ CUDA not available, using CPU. Embedding generation will be slower.")
    if GLOBAL_ENCODER is None:
        print(f"Loading LaBSE encoder ({LABSE_MODEL['name']}) on {DEVICE}...")
        try:
            GLOBAL_ENCODER = SentenceTransformer(LABSE_MODEL['name'], device=DEVICE)
            GLOBAL_ENCODER.eval()
            if DEVICE == 'cuda':
                torch.backends.cudnn.benchmark = True
            print("✅ LaBSE encoder loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading SentenceTransformer model: {e}")
            sys.exit(1)


def get_memory_info():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024 ** 3
    gpu_usage_str = ""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 3
        gpu_usage_str = f", GPU: {gpu_allocated:.2f}GB (Allocated) / {gpu_reserved:.2f}GB (Reserved)"
    return f"RAM: {ram_usage:.2f}GB{gpu_usage_str}"


def check_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama service is running.")
            return True
    except requests.exceptions.ConnectionError:
        print("🔌 Ollama service not detected. Attempting to start...")
    except requests.exceptions.Timeout:
        print("🔌 Ollama service timed out. Assuming not running or unresponsive.")
        return False
    try:
        if platform.system() == "Windows":
            print("   Attempting to start 'ollama serve' on Windows. This may require manual intervention.")
            subprocess.Popen(["ollama", "serve"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Darwin":
            print("   Attempting to start 'ollama serve' on macOS.")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("   Attempting to start 'ollama serve' on Linux.")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   Waiting for Ollama to start (10 seconds)...")
        time.sleep(10)
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service started successfully after attempt.")
            return True
        else:
            print(
                f"⚠️ Ollama service did not start successfully (status: {response.status_code}). Please start it manually.")
            return False
    except FileNotFoundError:
        print("❌ 'ollama' command not found. Please ensure Ollama is installed and in your system's PATH.")
        return False
    except Exception as e:
        print(f"❌ An error occurred while trying to start or check Ollama: {e}")
        return False
    return False


def pull_model_if_needed(model_name: str) -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        if any(m["name"] == model_name for m in models):
            print(f"✅ Model '{model_name}' is already available locally.")
            return True
        print(f"📥 Model '{model_name}' not found locally. Pulling... (This may take a while)")
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace"
        )
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print(f"✅ Successfully pulled model '{model_name}'.")
            return True
        else:
            print(f"❌ Failed to pull model '{model_name}'.")
            print(f"   Ollama stdout: {stdout}")
            print(f"   Ollama stderr: {stderr}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Ollama to check/pull model '{model_name}': {e}")
        return False
    except FileNotFoundError:
        print(f"❌ 'ollama' command not found. Cannot pull model '{model_name}'.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred while pulling model '{model_name}': {e}")
        return False


def call_ollama_model(prompt: str, model_config: LocalModelConfig) -> Tuple[str, float, int, int]:
    start_time = time.time()
    estimated_prompt_tokens = int(len(prompt) / 3.5)
    try:
        response = ollama.generate(
            model=model_config.ollama_name, prompt=prompt,
            options={
                'temperature': model_config.temperature,
                'num_predict': model_config.max_tokens,
                'num_thread': psutil.cpu_count(logical=True) // 2 or 1,
            },
            stream=False
        )
        generated_text = response.get('response', '')
        generation_time = time.time() - start_time
        estimated_response_tokens = int(len(generated_text) / 3.5)
        return generated_text, generation_time, estimated_prompt_tokens, estimated_response_tokens
    except Exception as e:
        error_msg = f"Error calling Ollama model {model_config.ollama_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, time.time() - start_time, estimated_prompt_tokens, 0


def parse_model_response(response_text: str) -> Tuple[str, str, str]:
    identified_person = "Uncertain"
    explanation = "No explanation provided or failed to parse."
    confidence = "N/A"
    if response_text.startswith("Error calling Ollama model"):
        return "Error", response_text, "N/A"
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if "identified person's name:" in line_lower or "identified person:" in line_lower:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    identified_person = parts[1].strip().replace("[", "").replace("]", "")
            elif "explanation:" in line_lower:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    explanation = parts[1].strip()
            elif "confidence score:" in line_lower or "confidence:" in line_lower:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    confidence = parts[1].strip()
        if explanation == "No explanation provided or failed to parse." and response_text:
            explanation = response_text
    except Exception as e:
        print(f"⚠️ Error parsing model response: {e}. Response was: '{response_text[:200]}...'")
        if not response_text.startswith("Error"):
            explanation = response_text
    return identified_person, explanation, confidence


# Removed calculate_ragas_metrics function

def test_model_on_item(
        model_config: LocalModelConfig,
        test_item: Dict,
        context_prompt_str: str,
        retrieval_scores: List[float],
        retrieved_contexts_text_list: List[str],
        target_persons: List[str],
        item_idx: int
) -> LocalModelTestResult:
    anonymized_data = test_item.get('anonymized', {})
    anonymized_text_full = anonymized_data.get('full_text', '')
    ground_truth_label = anonymized_data.get('keywords', ['Unknown'])[0] if anonymized_data.get(
        'keywords') else 'Unknown'
    max_anom_text_chars = int(model_config.context_length * 0.5 * 3.5)
    if len(anonymized_text_full) > max_anom_text_chars:
        display_anonymized_text = anonymized_text_full[:max_anom_text_chars] + "... [TRUNCATED DUE TO LENGTH]"
    else:
        display_anonymized_text = anonymized_text_full
    prompt_to_llm = OPTIMAL_PROMPT_TEMPLATE.format(
        text=display_anonymized_text, context=context_prompt_str,
        targets=', '.join(target_persons)
    )
    if len(prompt_to_llm) > model_config.context_length * 3.5:
        available_chars_for_context = int(model_config.context_length * 3.5) - len(
            OPTIMAL_PROMPT_TEMPLATE.format(text=display_anonymized_text, context="", targets=', '.join(target_persons)))
        if available_chars_for_context > 100:
            truncated_context_prompt_str = context_prompt_str[
                                           :available_chars_for_context - 20] + "... [CONTEXT TRUNCATED]"
        else:
            truncated_context_prompt_str = "Context information was too long and had to be truncated."
        prompt_to_llm = OPTIMAL_PROMPT_TEMPLATE.format(
            text=display_anonymized_text, context=truncated_context_prompt_str,
            targets=', '.join(target_persons)
        )
    overall_start_time = time.time()
    llm_response_text, generation_time_sec, estimated_prompt_tokens, estimated_response_tokens = call_ollama_model(
        prompt_to_llm, model_config)
    overall_process_time_sec = time.time() - overall_start_time
    if llm_response_text.startswith("Error calling Ollama model"):
        return LocalModelTestResult(
            item_index=item_idx, model_id=model_config.id, model_name=model_config.name,
            true_label=ground_truth_label, predicted_label="Error", confidence="N/A",
            explanation="LLM call failed.", retrieval_scores=retrieval_scores,
            generation_time=generation_time_sec, total_time=overall_process_time_sec,
            response_text=llm_response_text, timestamp=datetime.now().isoformat(),
            prompt_tokens=estimated_prompt_tokens, response_tokens=estimated_response_tokens,
            retrieved_contexts=retrieved_contexts_text_list, error=llm_response_text
        )
    predicted_person, explanation_text, confidence_level = parse_model_response(llm_response_text)
    return LocalModelTestResult(
        item_index=item_idx, model_id=model_config.id, model_name=model_config.name,
        true_label=ground_truth_label, predicted_label=predicted_person, confidence=confidence_level,
        explanation=explanation_text, retrieval_scores=retrieval_scores,
        generation_time=generation_time_sec, total_time=overall_process_time_sec,
        response_text=llm_response_text, timestamp=datetime.now().isoformat(),
        prompt_tokens=estimated_prompt_tokens, response_tokens=estimated_response_tokens,
        retrieved_contexts=retrieved_contexts_text_list, error=None  # No RAGAS metrics
    )


def test_models(
        models_to_test: List[LocalModelConfig],
        testing_data: List[Dict],
        rag_kb: OptimizedRAGKnowledgeBase,
        target_persons: List[str],
        # Removed ragas_eval_model_name parameter
        max_items: Optional[int] = None
) -> List[LocalModelTestResult]:
    all_results: List[LocalModelTestResult] = []
    items_to_process = testing_data[:max_items] if max_items else testing_data
    num_items = len(items_to_process)
    if num_items == 0:
        print("⚠️ No items to test. Exiting test_models.")
        return []
    print(f"\n📋 Preparing retrieval contexts for {num_items} test items...")
    pre_fetched_contexts = []
    for item_idx, test_item in tqdm(enumerate(items_to_process), total=num_items, desc="Fetching contexts"):
        anonymized_text_for_query = test_item.get('anonymized', {}).get('full_text', '')
        item_true_keywords = test_item.get('anonymized', {}).get('keywords', [])
        context_prompt_str, scores, chunk_objects, contexts_text_list = rag_kb.get_context_for_prompt(
            anonymized_text_for_query, item_true_keywords
        )
        pre_fetched_contexts.append((context_prompt_str, scores, chunk_objects, contexts_text_list))

    for model_config in models_to_test:
        print(f"\n🤖 Testing Model: {model_config.name} ({model_config.ollama_name})...")
        print(f"💾 Memory before testing {model_config.name}: {get_memory_info()}")
        current_model_item_results: List[LocalModelTestResult] = []
        for item_idx, test_item in tqdm(enumerate(items_to_process), total=num_items,
                                        desc=f"Testing {model_config.name}"):
            context_prompt_str, retrieval_scores, _, retrieved_contexts_text_list = pre_fetched_contexts[item_idx]
            item_result = test_model_on_item(
                model_config, test_item, context_prompt_str, retrieval_scores,
                retrieved_contexts_text_list, target_persons, item_idx
            )
            current_model_item_results.append(item_result)

        # Removed RAGAS evaluation block
        all_results.extend(current_model_item_results)
        valid_item_results = [res for res in current_model_item_results if res.error is None]
        num_valid = len(valid_item_results)
        num_total_attempted = len(current_model_item_results)
        if num_valid > 0:
            accuracy = sum(1 for r in valid_item_results if r.predicted_label == r.true_label) / num_valid
            avg_gen_time = sum(r.generation_time for r in valid_item_results) / num_valid
            print(f"   ✅ Model {model_config.name}: {num_valid}/{num_total_attempted} valid responses.")
            print(f"      Accuracy (on valid): {accuracy:.3f}, Avg Gen Time: {avg_gen_time:.2f}s")
        else:
            print(f"   ❌ Model {model_config.name}: All {num_total_attempted} calls failed or produced errors.")
        print(f"💾 Memory after testing {model_config.name}: {get_memory_info()}")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return all_results


def calculate_model_metrics(results: List[LocalModelTestResult], target_persons: List[str]) -> pd.DataFrame:
    metrics_by_model = []
    model_ids_with_results = sorted(list(set(r.model_id for r in results)))
    for model_id in model_ids_with_results:
        model_specific_results = [r for r in results if r.model_id == model_id]
        valid_results = [r for r in model_specific_results if r.error is None and r.predicted_label != "Error"]
        if not valid_results:
            model_info_fallback = next((m for m in LOCAL_MODELS if m.id == model_id), None)
            if model_info_fallback:
                metrics_by_model.append({
                    'model_id': model_id, 'model_name': model_info_fallback.name,
                    'model_size': model_info_fallback.size, 'accuracy': 0.0, 'precision': 0.0,
                    'recall': 0.0, 'f1_score': 0.0, 'mcc': 0.0, 'cohen_kappa': 0.0,
                    'balanced_accuracy': 0.0,
                    # Removed RAGAS metrics from here
                    'avg_generation_time': 0.0, 'avg_total_time': 0.0,
                    'avg_prompt_tokens': 0.0, 'avg_response_tokens': 0.0,
                    'tokens_per_second': 0.0, 'high_confidence_pct': 0.0,
                    'medium_confidence_pct': 0.0, 'low_confidence_pct': 0.0,
                    'valid_results': 0, 'total_attempts': len(model_specific_results),
                    'success_rate': 0.0
                })
            continue
        y_true = [r.true_label for r in valid_results]
        y_pred = [r.predicted_label for r in valid_results]
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0, labels=list(set(y_true + y_pred))
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        cohen_kappa = cohen_kappa_score(y_true, y_pred, labels=list(set(y_true + y_pred)))
        try:
            balanced_acc = balanced_accuracy_score(y_true, y_pred, adjusted=False)
        except ValueError:
            balanced_acc = accuracy
        avg_generation_time = np.mean([r.generation_time for r in valid_results if r.generation_time is not None])
        avg_total_time = np.mean([r.total_time for r in valid_results if r.total_time is not None])
        avg_prompt_tokens = np.mean([r.prompt_tokens for r in valid_results if r.prompt_tokens is not None])
        avg_response_tokens = np.mean([r.response_tokens for r in valid_results if r.response_tokens is not None])
        tokens_per_second = (
                    avg_response_tokens / avg_generation_time) if avg_generation_time and avg_generation_time > 0 else 0.0

        # Removed RAGAS metrics dictionary and composite score calculation

        confidence_levels = [r.confidence for r in valid_results]
        total_valid_conf = len(confidence_levels)
        confidence_counts = pd.Series(confidence_levels).value_counts()
        model_config_info = next(m for m in LOCAL_MODELS if m.id == model_id)
        metrics_by_model.append({
            'model_id': model_id, 'model_name': model_config_info.name,
            'model_size': model_config_info.size, 'accuracy': accuracy,
            'precision': precision, 'recall': recall, 'f1_score': f1, 'mcc': mcc,
            'cohen_kappa': cohen_kappa, 'balanced_accuracy': balanced_acc,
            # Removed RAGAS individual metrics and composite score from here
            'avg_generation_time': avg_generation_time, 'avg_total_time': avg_total_time,
            'avg_prompt_tokens': avg_prompt_tokens, 'avg_response_tokens': avg_response_tokens,
            'tokens_per_second': tokens_per_second,
            'high_confidence_pct': confidence_counts.get('High', 0) / total_valid_conf if total_valid_conf > 0 else 0,
            'medium_confidence_pct': confidence_counts.get('Medium',
                                                           0) / total_valid_conf if total_valid_conf > 0 else 0,
            'low_confidence_pct': confidence_counts.get('Low', 0) / total_valid_conf if total_valid_conf > 0 else 0,
            'uncertain_pct': confidence_counts.get('Uncertain', 0) / total_valid_conf if total_valid_conf > 0 else 0,
            'na_confidence_pct': confidence_counts.get('N/A', 0) / total_valid_conf if total_valid_conf > 0 else 0,
            'valid_results': len(valid_results), 'total_attempts': len(model_specific_results),
            'success_rate': len(valid_results) / len(model_specific_results) if len(model_specific_results) > 0 else 0
        })
    return pd.DataFrame(metrics_by_model).sort_values(by='f1_score', ascending=False)


def create_model_comparison_visualizations(metrics_df: pd.DataFrame, results: List[LocalModelTestResult]):
    if metrics_df.empty:
        print("⚠️ No metrics data to create visualizations.")
        return
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    metrics_df = metrics_df.sort_values('f1_score', ascending=False)

    # Simplified visualization: F1 Score, Accuracy, Avg Generation Time
    fig = make_subplots(
        rows=2, cols=2,
        # specs adjusted as RAGAS subplot is removed
        subplot_titles=('F1 Score vs. Model', 'Accuracy vs. Avg Generation Time',
                        'Model Efficiency (Tokens/Sec)', 'Success Rate vs. Model')
    )

    # Plot 1: F1 Score (bar)
    fig.add_trace(
        go.Bar(
            x=metrics_df['model_name'], y=metrics_df['f1_score'],
            name='F1 Score (Weighted)', marker_color='cornflowerblue',
            text=metrics_df['f1_score'].apply(lambda x: f'{x:.3f}'), textposition='auto'
        ), row=1, col=1
    )

    # Plot 2: Accuracy vs. Avg Generation Time (scatter)
    fig.add_trace(
        go.Scatter(
            x=metrics_df['avg_generation_time'], y=metrics_df['accuracy'],
            mode='markers+text', text=metrics_df['model_name'], textposition='bottom right',
            marker=dict(
                size=np.maximum(5, metrics_df['tokens_per_second'] / 5),
                sizemode='diameter', sizeref=0.5,
                color=metrics_df['f1_score'], colorscale='Blues',  # Color by F1 score now
                showscale=True, colorbar=dict(title="F1 Score", x=1.05, y=0.8, len=0.4)
            ), name='Models (Size by TPS)'
        ), row=1, col=2
    )

    # Plot 3: Model Efficiency (Tokens/Sec)
    fig.add_trace(
        go.Bar(
            x=metrics_df['model_name'], y=metrics_df['tokens_per_second'],
            name='Tokens per Second', marker_color='mediumseagreen',
            text=metrics_df['tokens_per_second'].apply(lambda x: f'{x:.1f}'), textposition='auto'
        ), row=2, col=1
    )

    # Plot 4: Success Rate
    fig.add_trace(
        go.Bar(
            x=metrics_df['model_name'], y=metrics_df['success_rate'],
            name='Success Rate (Valid Results %)', marker_color='lightcoral',
            text=metrics_df['success_rate'].apply(lambda x: f'{x * 100:.1f}%'), textposition='auto'
        ), row=2, col=2
    )

    fig.update_layout(
        height=1000, width=1400, title_text="<b>Local LLM Performance Dashboard</b>",
        title_x=0.5, legend_title_text='Metrics',
        font=dict(family="Arial, sans-serif", size=12, color="RebeccaPurple")
    )
    fig.update_yaxes(title_text="F1 Score (Weighted)", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Avg. Generation Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text="Tokens / Second", row=2, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=2, col=2, range=[0, 1])
    fig.update_xaxes(tickangle=-30)  # Apply to all x-axes

    plot_path = os.path.join(VISUALIZATIONS_DIR, "local_model_performance_dashboard.html")
    fig.write_html(plot_path)
    print(f"📊 Interactive dashboard saved to: {plot_path}")

    # Removed RAGAS Metrics Heatmap


def create_detailed_report(metrics_df: pd.DataFrame, results: List[LocalModelTestResult]):
    if metrics_df.empty and not results:
        print("⚠️ No data available to generate a detailed report.")
        return
    report_content = []
    report_content.append(f"# Local LLM Evaluation Report\n")  # Title changed
    report_content.append(f"**Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.append(
        f"**Total Items Tested**: {len(results) // len(metrics_df) if not metrics_df.empty and len(metrics_df) > 0 else 'N/A'} (approx. per model)\n")

    report_content.append("## Executive Summary\n")
    if not metrics_df.empty:
        metrics_df_sorted_f1 = metrics_df.sort_values('f1_score', ascending=False)
        best_f1_model = metrics_df_sorted_f1.iloc[0]

        # Simplified efficiency score (F1 / Avg Gen Time)
        metrics_df['efficiency'] = metrics_df['f1_score'] / (metrics_df['avg_generation_time'] + 1e-6)
        best_efficiency_model = metrics_df.sort_values('efficiency', ascending=False).iloc[0]

        report_content.append(
            f"- **Top Performer (F1 Score)**: **{best_f1_model['model_name']}** (F1: {best_f1_model['f1_score']:.3f}, Accuracy: {best_f1_model['accuracy']:.3f})")
        report_content.append(
            f"- **Most Efficient (F1/Time)**: **{best_efficiency_model['model_name']}** (Efficiency: {best_efficiency_model['efficiency']:.3f}, F1: {best_efficiency_model['f1_score']:.3f}, Time: {best_efficiency_model['avg_generation_time']:.2f}s)")
    else:
        report_content.append("No model metrics available for summary.\n")
    report_content.append("\n")

    report_content.append("## Detailed Model Performance Metrics\n")
    if not metrics_df.empty:
        # Simplified key columns for the main table
        key_cols = ['model_name', 'model_size', 'f1_score', 'accuracy',
                    'avg_generation_time', 'tokens_per_second', 'success_rate']
        report_table_df = metrics_df[key_cols].copy()
        report_table_df.rename(columns={
            'model_name': 'Model', 'model_size': 'Size', 'f1_score': 'F1',
            'accuracy': 'Acc.', 'avg_generation_time': 'Avg.Time(s)',
            'tokens_per_second': 'Tok/s', 'success_rate': 'Success %'
        }, inplace=True)
        for col in ['F1', 'Acc.']:
            report_table_df[col] = report_table_df[col].apply(lambda x: f"{x:.3f}")
        report_table_df['Avg.Time(s)'] = report_table_df['Avg.Time(s)'].apply(lambda x: f"{x:.2f}")
        report_table_df['Tok/s'] = report_table_df['Tok/s'].apply(lambda x: f"{x:.1f}")
        report_table_df['Success %'] = report_table_df['Success %'].apply(lambda x: f"{x * 100:.1f}%")
        report_content.append(report_table_df.to_markdown(index=False))
    else:
        report_content.append("No model metrics data to display.\n")
    report_content.append("\n")

    report_content.append("## Conclusion & Recommendations (Example)\n")
    report_content.append("Based on the evaluation:")
    if not metrics_df.empty:
        report_content.append(
            f"- For best classification performance (F1 score), **{best_f1_model['model_name']}** is the leading candidate.")
        report_content.append(
            f"- For scenarios prioritizing speed and resource efficiency, **{best_efficiency_model['model_name']}** offers a good trade-off.")
    report_content.append(
        "- Further testing with a larger dataset or more diverse queries could provide deeper insights.\n")

    report_path = os.path.join(RESULTS_DIR, "local_model_evaluation_report.md")  # Simplified name
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        print(f"📄 Detailed Markdown report saved to: {report_path}")
    except IOError as e:
        print(f"❌ Error saving detailed report: {e}")


def save_detailed_results(results: List[LocalModelTestResult]):
    if not results:
        print("⚠️ No detailed results to save.")
        return
    detailed_data_rows = []
    for r_item in results:
        row_dict = {
            'item_index': r_item.item_index, 'model_id': r_item.model_id,
            'model_name': r_item.model_name, 'true_label': r_item.true_label,
            'predicted_label': r_item.predicted_label,
            'is_correct': r_item.true_label == r_item.predicted_label if r_item.error is None else False,
            'confidence': r_item.confidence,
            'explanation_preview': r_item.explanation[:150].replace("\n", " ") + "..." if r_item.explanation else "",
            'generation_time_s': r_item.generation_time,
            'total_processing_time_s': r_item.total_time,
            'estimated_prompt_tokens': r_item.prompt_tokens,
            'estimated_response_tokens': r_item.response_tokens,
            'error_message': r_item.error, 'timestamp': r_item.timestamp,
            # Removed RAGAS metrics columns
        }
        for i, score_val in enumerate(r_item.retrieval_scores[:3]):
            row_dict[f'retrieval_score_top_{i + 1}'] = score_val
        if r_item.retrieved_contexts:
            row_dict['retrieved_context_1_preview'] = r_item.retrieved_contexts[0][:200].replace("\n", " ") + "..."
        else:
            row_dict['retrieved_context_1_preview'] = "N/A"
        detailed_data_rows.append(row_dict)
    detailed_results_df = pd.DataFrame(detailed_data_rows)
    csv_path = os.path.join(RESULTS_DIR, "all_models_detailed_item_results_classification.csv")  # New name
    try:
        detailed_results_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 Detailed item-level results saved to: {csv_path}")
    except IOError as e:
        print(f"❌ Error saving detailed CSV results: {e}")


def load_data():
    target_persons_list = []
    try:
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            target_persons_list = [line.strip() for line in f if line.strip()]
        if not target_persons_list: print(f"⚠️ Keywords file '{KEYWORDS_FILE}' is empty or not found.")
    except Exception as e:
        print(f"❌ Error loading keywords from '{KEYWORDS_FILE}': {e}")
        return None, None, None
    training_set_data = []
    try:
        with open(TRAINING_SET_FILE, 'r', encoding='utf-8') as f:
            training_set_data = json.load(f)
        if not training_set_data: print(f"⚠️ Training set file '{TRAINING_SET_FILE}' is empty or not found.")
    except Exception as e:
        print(f"❌ Error loading training data from '{TRAINING_SET_FILE}': {e}")
        return None, None, None
    validation_set_data = []
    try:
        with open(VALIDATION_SET_FILE, 'r', encoding='utf-8') as f:
            validation_set_data = json.load(f)
        if not validation_set_data: print(f"⚠️ Validation set file '{VALIDATION_SET_FILE}' is empty or not found.")
    except Exception as e:
        print(f"❌ Error loading validation data from '{VALIDATION_SET_FILE}': {e}")
        return None, None, None
    return target_persons_list, training_set_data, validation_set_data


def main():
    parser = argparse.ArgumentParser(description='Test local LLMs with Ollama and RAG (Classification Metrics Only).')
    parser.add_argument('--test-count', type=int, default=None,
                        help='Number of validation items to test on (default: all).')
    parser.add_argument('--skip-models', nargs='+', default=[],
                        help='List of model IDs (e.g., "gemma-2b", "llama3-8b") to skip during testing.')
    # Removed --ragas-eval-model argument
    parser.add_argument('--no-cache', action='store_true',
                        help='Force rebuild of the knowledge base, ignoring any existing cache.')
    args = parser.parse_args()

    print("🚀 Starting Local LLM RAG Evaluation Pipeline (Classification Metrics Only)")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ Optimal RAG Parameters: {OPTIMAL_PARAMS}")
    # Removed RAGAS Evaluation Model print
    print("=" * 70)

    if not check_ollama_running():
        print("❌ Ollama service is not running or responding. Please ensure Ollama is installed and started.")
        return
    initialize_global_encoder()
    print(f"💾 Initial memory usage: {get_memory_info()}")
    print("\n📂 Loading datasets (keywords, training, validation)...")
    target_persons, training_data, validation_data = load_data()
    if not all([target_persons, training_data, validation_data]):
        print("❌ Critical data loading failed. Exiting pipeline.")
        return
    print(f"✅ Loaded {len(target_persons)} target persons (for prompt generation).")
    print(f"✅ Loaded {len(training_data)} documents for Knowledge Base construction.")
    print(f"✅ Loaded {len(validation_data)} items for validation/testing.")
    print("\n🏗️ Building/Loading Knowledge Base for RAG...")
    rag_knowledge_base = OptimizedRAGKnowledgeBase()
    rag_knowledge_base.build_knowledge_base(training_data, use_cache=(not args.no_cache))
    if not rag_knowledge_base.is_built:
        print("❌ Failed to build or load the RAG knowledge base. Exiting.")
        return
    print(f"✅ Knowledge Base ready with {len(rag_knowledge_base.chunks)} chunks.")
    print(f"💾 Memory after KB construction: {get_memory_info()}")
    models_for_testing_configs = [m for m in LOCAL_MODELS if m.id not in args.skip_models]
    if not models_for_testing_configs:
        print("⚠️ No models selected for testing.")
        return
    print(f"\n📋 Models selected for testing: {len(models_for_testing_configs)}")
    for model_conf in models_for_testing_configs:
        print(f"   - {model_conf.name} (Ollama: {model_conf.ollama_name}, Size: {model_conf.size})")

    # Simplified model pulling: only for models being tested
    all_required_ollama_names = list(set([m.ollama_name for m in models_for_testing_configs]))
    print("\n📥 Checking and pulling necessary Ollama models (if not present)...")
    all_models_available = True
    for ollama_model_name in all_required_ollama_names:
        if not pull_model_if_needed(ollama_model_name):
            print(f"❌ Critical: Failed to pull or confirm availability of Ollama model '{ollama_model_name}'.")
            all_models_available = False  # Continue if some models fail, skip them later

    # Filter models_for_testing_configs to only those that are confirmed available
    # This requires pull_model_if_needed to be robust or a re-check. For simplicity, we'll assume
    # if a model fails to pull, subsequent calls to it via ollama.generate will fail gracefully.

    print("\n🧪 Starting model testing (Classification Metrics Only)...")
    pipeline_start_time = time.time()
    test_run_results = test_models(
        models_for_testing_configs, validation_data, rag_knowledge_base,
        target_persons,  # Removed ragas_eval_model from arguments
        max_items=args.test_count
    )
    pipeline_duration_sec = time.time() - pipeline_start_time
    print(f"\n✅ Full testing pipeline completed in {pipeline_duration_sec / 60:.2f} minutes.")
    print(f"💾 Final memory usage: {get_memory_info()}")
    if not test_run_results:
        print("⚠️ No results were generated from the test run. Skipping metrics calculation and reporting.")
        return
    print("\n📊 Calculating aggregate performance metrics...")
    model_metrics_summary_df = calculate_model_metrics(test_run_results, target_persons)
    save_detailed_results(test_run_results)
    if not model_metrics_summary_df.empty:
        summary_csv_path = os.path.join(RESULTS_DIR, "local_models_summary_metrics_classification.csv")  # New name
        model_metrics_summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"💾 Summary metrics saved to: {summary_csv_path}")
    else:
        print("⚠️ Summary metrics DataFrame is empty. Not saved.")
    if not model_metrics_summary_df.empty:
        print("\n📈 Generating visualizations...")
        create_model_comparison_visualizations(model_metrics_summary_df, test_run_results)
        print("\n✍️ Generating detailed Markdown report...")
        create_detailed_report(model_metrics_summary_df, test_run_results)
    else:
        print("⚠️ Skipping visualizations and report generation due to empty summary metrics.")
    print("\n" + "=" * 70)
    print("🏆 PIPELINE EXECUTION SUMMARY (CLASSIFICATION METRICS ONLY) 🏆")
    print("=" * 70)
    if not model_metrics_summary_df.empty:
        print("\n### Top Performing Models (by F1 Score) ###")
        console_summary_df = model_metrics_summary_df.sort_values('f1_score', ascending=False).head(5)
        for idx, row_data in console_summary_df.iterrows():
            print(f"\nRank {idx + 1}: {row_data['model_name']} (Size: {row_data['model_size']})")
            print(f"  F1 Score: {row_data['f1_score']:.3f}")
            print(f"  Accuracy: {row_data['accuracy']:.3f}")
            # Removed RAGAS score print
            print(f"  Avg. Gen Time: {row_data['avg_generation_time']:.2f}s")
            print(f"  Success Rate: {row_data['success_rate']:.1%}")
        # Removed Best RAGAS Quality Model section
    else:
        print("No summary metrics available to display.")
    print(f"\n💾 All detailed outputs, summaries, and visualizations are in: '{RESULTS_DIR}' and '{VISUALIZATIONS_DIR}'")
    print("\n✅ Local LLM RAG Evaluation Pipeline Finished Successfully!")


if __name__ == "__main__":
    main()
