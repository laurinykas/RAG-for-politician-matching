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
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
except ImportError:
    print("Installing required packages...")
    pip_command = [sys.executable, "-m", "pip", "install",
                   "sentence-transformers", "ollama", "ragas", "datasets", "tqdm", "openai"]
    subprocess.check_call(pip_command)
    from sentence_transformers import SentenceTransformer
    import ollama
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset

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
    ollama_name: str  # Name to use with ollama
    size: str  # Approximate model size
    context_length: int
    temperature: float = 0.1
    max_tokens: int = 2000


# Define local models that actually exist in Ollama
LOCAL_MODELS = [
    LocalModelConfig(
        id="gemma-2b",
        name="Google Gemma 2B",
        ollama_name="gemma:2b",
        size="2B",
        context_length=8192,
    ),
    LocalModelConfig(
        id="gemma-7b",
        name="Google Gemma 7B",
        ollama_name="gemma:7b",
        size="7B",
        context_length=8192,
    ),
    LocalModelConfig(
        id="llama3-8b",
        name="Llama 3 8B",
        ollama_name="llama3:8b",
        size="8B",
        context_length=8192,
    ),
    LocalModelConfig(
        id="mistral-7b",
        name="Mistral 7B Instruct",
        ollama_name="mistral:7b-instruct",
        size="7B",
        context_length=8192,
    ),
    LocalModelConfig(
        id="phi-2",
        name="Microsoft Phi-2",
        ollama_name="phi:2.7b",
        size="2.7B",
        context_length=2048,
    ),
    LocalModelConfig(
        id="qwen-4b",
        name="Qwen 4B",
        ollama_name="qwen:4b",
        size="4B",
        context_length=8192,
    ),
]


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_similarity: float
    answer_correctness: float


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
    # RAGAS metrics
    ragas_metrics: Optional[RAGASMetrics] = None
    # Token counts (estimated)
    prompt_tokens: int = 0
    response_tokens: int = 0
    # Retrieved contexts
    retrieved_contexts: List[str] = field(default_factory=list)
    # Error tracking
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
                print(f"Failed to load cache: {e}")

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

            for chunk_text in text_chunks:
                chunk_anonymized_persons = [p for p in anonymized_persons_in_doc if p in chunk_text]
                chunk = DocumentChunk(
                    text=chunk_text,
                    original_persons=keywords,
                    anonymized_persons=chunk_anonymized_persons,
                    source_url=url,
                    title=title,
                    chunk_id=len(self.chunks),
                    replacements=replacements
                )
                self.chunks.append(chunk)
                all_texts.append(chunk_text)

        # Generate embeddings
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

            # Save cache
            if use_cache:
                cache_data = {
                    'chunks': self.chunks,
                    'embeddings': self.embeddings,
                    'person_mapping': self.person_mapping
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print("✅ Saved knowledge base to cache")

    def semantic_search(self, query: str) -> Tuple[List[Tuple[DocumentChunk, float]], List[float]]:
        if not self.is_built:
            return [], []

        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [query], show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True
            )

        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        num_candidates = min(len(similarities), self.top_k_retrieval * 2)
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
        semantic_results, scores = self.semantic_search(query)
        if not semantic_results:
            return [], []

        boosted_results = []
        boosted_scores = []

        for (chunk, similarity), score in zip(semantic_results, scores):
            boost = 0.0
            boost_details = {
                'semantic_similarity': similarity,
                'person_match_boost': 0.0,
                'anonymized_ref_boost': 0.0,
                'length_boost': 0.0
            }

            # Boost for matching original persons
            for target_person in target_persons:
                if any(target_person.lower() in op.lower() for op in chunk.original_persons):
                    person_boost = 0.20
                    boost += person_boost
                    boost_details['person_match_boost'] = max(boost_details['person_match_boost'], person_boost)

            # Boost for anonymized references
            for anon_id, orig_name in self.person_mapping.items():
                if anon_id in chunk.text:
                    if any(target_person.lower() in orig_name.lower() for target_person in target_persons):
                        anon_boost = 0.15
                        boost += anon_boost
                        boost_details['anonymized_ref_boost'] = max(boost_details['anonymized_ref_boost'], anon_boost)

            # Length boost
            length_boost = min(len(chunk.text) / 2000.0, 0.1)
            boost += length_boost
            boost_details['length_boost'] = length_boost

            final_score = min(similarity + boost, 1.0)
            boosted_results.append((chunk, final_score, boost_details))
            boosted_scores.append(final_score)

        # Sort by boosted score
        sorted_results = sorted(zip(boosted_results, boosted_scores), key=lambda x: x[1], reverse=True)
        boosted_results = [r[0] for r in sorted_results]
        boosted_scores = [r[1] for r in sorted_results]

        return boosted_results[:self.top_k_retrieval], boosted_scores[:self.top_k_retrieval]

    def get_context_for_prompt(self, query: str, target_persons: List[str]) -> Tuple[
        str, List[float], List[DocumentChunk], List[str]]:
        results, scores = self.hybrid_retrieval(query, target_persons)
        if not results:
            return "No relevant contextual information found.", [], [], []

        context_parts = []
        retrieved_chunks = []
        contexts_list = []

        for i, ((chunk, score, boost_details), _) in enumerate(zip(results, scores), 1):
            original_persons_str = ", ".join(chunk.original_persons) if chunk.original_persons else "Unknown"
            anonymized_persons_str = ", ".join(chunk.anonymized_persons) if chunk.anonymized_persons else "None"
            context_parts.append(f"""
Context {i} (Relevance Score: {score:.3f}):
  - Original persons mentioned (document level): {original_persons_str}
  - Anonymized references in chunk: {anonymized_persons_str}
  - Source: {chunk.title} (URL: {chunk.source_url})
  - Text: "{chunk.text}"
""")
            retrieved_chunks.append(chunk)
            contexts_list.append(chunk.text)

        return "\n".join(context_parts), scores, retrieved_chunks, contexts_list


def initialize_global_encoder():
    """Initialize the global encoder model"""
    global GLOBAL_ENCODER, DEVICE

    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("⚠️ CUDA not available, using CPU")

    if GLOBAL_ENCODER is None:
        print(f"Loading LaBSE encoder on {DEVICE}...")
        GLOBAL_ENCODER = SentenceTransformer(LABSE_MODEL['name'], device=DEVICE)
        GLOBAL_ENCODER.eval()
        if DEVICE == 'cuda':
            torch.backends.cudnn.benchmark = True


def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024 ** 3

    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return f"RAM: {ram_usage:.2f}GB, GPU: {gpu_usage:.2f}/{gpu_reserved:.2f}GB"
    return f"RAM: {ram_usage:.2f}GB"


def check_ollama_running():
    """Check if Ollama is running and start it if needed"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return True
    except:
        pass

    print("Starting Ollama service...")
    if platform.system() == "Windows":
        subprocess.Popen(["ollama", "serve"], shell=True)
    else:
        subprocess.Popen(["ollama", "serve"])

    # Wait for Ollama to start
    for _ in range(30):
        time.sleep(1)
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print("✅ Ollama service started")
                return True
        except:
            pass

    print("❌ Failed to start Ollama service")
    return False


def pull_model_if_needed(model_name: str) -> bool:
    """Pull model if not already available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if any(m["name"] == model_name for m in models):
                return True

        print(f"📥 Pulling {model_name}... This may take a while.")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        if result.stdout:
            print(result.stdout)

        return result.returncode == 0

    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")
        return False


def call_ollama_model(prompt: str, model_config: LocalModelConfig) -> Tuple[str, float, int, int]:
    """Call Ollama model synchronously"""
    start_time = time.time()

    try:
        response = ollama.generate(
            model=model_config.ollama_name,
            prompt=prompt,
            options={
                'temperature': model_config.temperature,
                'num_predict': model_config.max_tokens,
                'num_thread': 8
            }
        )

        text = response['response']
        generation_time = time.time() - start_time

        # Estimate tokens
        prompt_tokens = len(prompt.split()) * 1.3
        response_tokens = len(text.split()) * 1.3

        return text, generation_time, int(prompt_tokens), int(response_tokens)

    except Exception as e:
        error_msg = f"Error calling {model_config.ollama_name}: {str(e)}"
        return error_msg, time.time() - start_time, 0, 0


def parse_model_response(response_text: str) -> Tuple[str, str, str]:
    """Parse model response to extract person, explanation, and confidence"""
    identified_person = "Uncertain"
    explanation = "No explanation provided."
    confidence = "N/A"

    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            line_lower = line.lower().strip()

            if "identified person's name:" in line_lower or "identified person:" in line_lower:
                identified_person = line.split(":", 1)[1].strip().replace("[", "").replace("]", "")
            elif "explanation:" in line_lower:
                explanation = line.split(":", 1)[1].strip()
            elif "confidence score:" in line_lower or "confidence:" in line_lower:
                confidence = line.split(":", 1)[1].strip()
    except Exception:
        pass

    return identified_person, explanation, confidence


def calculate_ragas_metrics(
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
) -> Dict[str, float]:
    """Calculate RAGAS metrics for a batch of results"""

    # Prepare data for RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    # Define metrics to evaluate
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    ]

    # Note: RAGAS requires an LLM for evaluation. You need to set OPENAI_API_KEY
    # or configure another LLM for evaluation
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
        )

        return {
            "faithfulness": result.get('faithfulness', 0.0),
            "answer_relevancy": result.get('answer_relevancy', 0.0),
            "context_precision": result.get('context_precision', 0.0),
            "context_recall": result.get('context_recall', 0.0),
            "answer_similarity": result.get('answer_similarity', 0.0),
            "answer_correctness": result.get('answer_correctness', 0.0)
        }
    except Exception as e:
        print(f"Warning: RAGAS evaluation failed: {e}")
        print("Make sure OPENAI_API_KEY is set for RAGAS evaluation")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_similarity": 0.0,
            "answer_correctness": 0.0
        }


def test_model_on_item(
        model_config: LocalModelConfig,
        test_item: Dict,
        context: str,
        retrieval_scores: List[float],
        retrieved_chunks: List[DocumentChunk],
        contexts_list: List[str],
        target_persons: List[str],
        item_idx: int
) -> LocalModelTestResult:
    """Test a single model on a single item"""

    try:
        anonymized_data = test_item.get('anonymized', {})
        anonymized_text = anonymized_data.get('full_text', '')
        ground_truth = anonymized_data.get('keywords', [])[0] if anonymized_data.get('keywords') else 'Unknown'

        # Format prompt
        prompt = OPTIMAL_PROMPT_TEMPLATE.format(
            text=anonymized_text,
            context=context,
            targets=', '.join(target_persons)
        )

        # Truncate if needed
        if len(prompt) > model_config.context_length * 4:
            max_text_length = model_config.context_length * 3
            truncated_text = anonymized_text[:max_text_length] + "... [TRUNCATED]"
            prompt = OPTIMAL_PROMPT_TEMPLATE.format(
                text=truncated_text,
                context=context,
                targets=', '.join(target_persons)
            )

        # Call model
        start_time = time.time()
        response_text, generation_time, prompt_tokens, response_tokens = call_ollama_model(prompt, model_config)
        total_time = time.time() - start_time

        # Check for errors
        if response_text.startswith("Error"):
            return LocalModelTestResult(
                item_index=item_idx,
                model_id=model_config.id,
                model_name=model_config.name,
                true_label=ground_truth,
                predicted_label="Error",
                confidence="N/A",
                explanation="Model call failed",
                retrieval_scores=retrieval_scores,
                generation_time=generation_time,
                total_time=total_time,
                response_text=response_text,
                timestamp=datetime.now().isoformat(),
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                retrieved_contexts=contexts_list,
                error=response_text
            )

        # Parse response
        identified_person, explanation, confidence = parse_model_response(response_text)

        return LocalModelTestResult(
            item_index=item_idx,
            model_id=model_config.id,
            model_name=model_config.name,
            true_label=ground_truth,
            predicted_label=identified_person,
            confidence=confidence,
            explanation=explanation,
            retrieval_scores=retrieval_scores,
            generation_time=generation_time,
            total_time=total_time,
            response_text=response_text,
            timestamp=datetime.now().isoformat(),
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            retrieved_contexts=contexts_list,
            ragas_metrics=None  # Will be calculated in batch
        )

    except Exception as e:
        return LocalModelTestResult(
            item_index=item_idx,
            model_id=model_config.id,
            model_name=model_config.name,
            true_label=test_item.get('anonymized', {}).get('keywords', ['Unknown'])[0],
            predicted_label="Error",
            confidence="N/A",
            explanation=f"Exception: {str(e)}",
            retrieval_scores=retrieval_scores,
            generation_time=0,
            total_time=0,
            response_text="",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )


def test_models(
        models_to_test: List[LocalModelConfig],
        testing_data: List[Dict],
        rag_kb: OptimizedRAGKnowledgeBase,
        target_persons: List[str],
        max_items: Optional[int] = None
) -> List[LocalModelTestResult]:
    """Test all models on all items"""

    results = []
    items_to_test = testing_data[:max_items] if max_items else testing_data

    # First, get all retrieval results
    print("\n📋 Preparing test data and retrievals...")
    test_contexts = []

    for item in tqdm(items_to_test, desc="Getting retrievals"):
        anonymized_text = item.get('anonymized', {}).get('full_text', '')
        ground_truth_keywords = item.get('anonymized', {}).get('keywords', [])
        context, scores, chunks, contexts_list = rag_kb.get_context_for_prompt(anonymized_text, ground_truth_keywords)
        test_contexts.append((context, scores, chunks, contexts_list))

    # Test each model
    for model_config in models_to_test:
        print(f"\n🤖 Testing {model_config.name}...")
        print(f"💾 Memory before: {get_memory_info()}")

        # Check if model is available
        if not pull_model_if_needed(model_config.ollama_name):
            print(f"❌ Failed to pull {model_config.ollama_name}. Skipping.")
            continue

        # Test on all items with progress bar
        model_results = []

        for idx, (item, (context, scores, chunks, contexts_list)) in tqdm(
                enumerate(zip(items_to_test, test_contexts)),
                total=len(items_to_test),
                desc=f"Testing {model_config.name}"
        ):
            result = test_model_on_item(
                model_config, item, context, scores, chunks, contexts_list,
                target_persons, idx
            )
            model_results.append(result)

        # Calculate RAGAS metrics in batch for valid results
        valid_results = [r for r in model_results if r.error is None]

        if valid_results and os.getenv("OPENAI_API_KEY"):
            print(f"   📊 Calculating RAGAS metrics for {len(valid_results)} results...")

            # Prepare batch data
            questions = [items_to_test[r.item_index].get('anonymized', {}).get('full_text', '') for r in valid_results]
            answers = [r.predicted_label for r in valid_results]
            contexts = [r.retrieved_contexts for r in valid_results]
            ground_truths = [r.true_label for r in valid_results]

            # Calculate RAGAS metrics
            ragas_scores = calculate_ragas_metrics(questions, answers, contexts, ground_truths)

            # Assign metrics to results
            for result in valid_results:
                result.ragas_metrics = RAGASMetrics(**ragas_scores)

        results.extend(model_results)

        # Quick stats
        if valid_results:
            accuracy = sum(1 for r in valid_results if r.predicted_label == r.true_label) / len(valid_results)
            avg_time = sum(r.generation_time for r in valid_results) / len(valid_results)
            print(f"   ✅ Valid results: {len(valid_results)}/{len(model_results)}")
            print(f"   📊 Accuracy: {accuracy:.3f}, Avg generation time: {avg_time:.2f}s")

            if valid_results[0].ragas_metrics:
                print(f"   🎯 RAGAS - Faithfulness: {ragas_scores['faithfulness']:.3f}, "
                      f"Answer Correctness: {ragas_scores['answer_correctness']:.3f}")
        else:
            print(f"   ❌ All calls failed")

        print(f"💾 Memory after: {get_memory_info()}")

        # Clear cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return results


def calculate_model_metrics(results: List[LocalModelTestResult], target_persons: List[str]) -> pd.DataFrame:
    """Calculate comprehensive metrics for each model including RAGAS metrics"""

    metrics_by_model = []

    for model_id in set(r.model_id for r in results):
        model_results = [r for r in results if r.model_id == model_id]
        valid_results = [r for r in model_results if r.error is None]

        if not valid_results:
            continue

        y_true = [r.true_label for r in valid_results]
        y_pred = [r.predicted_label for r in valid_results]

        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Timing metrics
        avg_generation_time = np.mean([r.generation_time for r in valid_results])
        avg_total_time = np.mean([r.total_time for r in valid_results])
        avg_prompt_tokens = np.mean([r.prompt_tokens for r in valid_results])
        avg_response_tokens = np.mean([r.response_tokens for r in valid_results])

        # RAGAS metrics (if available)
        ragas_metrics_dict = {
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'answer_similarity': 0.0,
            'answer_correctness': 0.0
        }

        if valid_results[0].ragas_metrics:
            ragas_metrics_dict = {
                'faithfulness': valid_results[0].ragas_metrics.faithfulness,
                'answer_relevancy': valid_results[0].ragas_metrics.answer_relevancy,
                'context_precision': valid_results[0].ragas_metrics.context_precision,
                'context_recall': valid_results[0].ragas_metrics.context_recall,
                'answer_similarity': valid_results[0].ragas_metrics.answer_similarity,
                'answer_correctness': valid_results[0].ragas_metrics.answer_correctness
            }

        # Composite RAGAS score
        ragas_score = (
                ragas_metrics_dict['faithfulness'] * 0.3 +
                ragas_metrics_dict['answer_correctness'] * 0.3 +
                ragas_metrics_dict['context_precision'] * 0.2 +
                ragas_metrics_dict['answer_relevancy'] * 0.2
        )

        # Confidence distribution
        confidence_counts = pd.Series([r.confidence for r in valid_results]).value_counts()
        total_conf = len(valid_results)

        # Model info
        model_info = next(r for r in model_results)

        metrics_by_model.append({
            'model_id': model_id,
            'model_name': model_info.model_name,
            'model_size': next(m.size for m in LOCAL_MODELS if m.id == model_id),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            # RAGAS metrics
            'ragas_score': ragas_score,
            'faithfulness': ragas_metrics_dict['faithfulness'],
            'answer_relevancy': ragas_metrics_dict['answer_relevancy'],
            'context_precision': ragas_metrics_dict['context_precision'],
            'context_recall': ragas_metrics_dict['context_recall'],
            'answer_similarity': ragas_metrics_dict['answer_similarity'],
            'answer_correctness': ragas_metrics_dict['answer_correctness'],
            # Performance metrics
            'avg_generation_time': avg_generation_time,
            'avg_total_time': avg_total_time,
            'avg_prompt_tokens': avg_prompt_tokens,
            'avg_response_tokens': avg_response_tokens,
            'tokens_per_second': avg_response_tokens / avg_generation_time if avg_generation_time > 0 else 0,
            # Confidence
            'high_confidence_pct': confidence_counts.get('High', 0) / total_conf,
            'medium_confidence_pct': confidence_counts.get('Medium', 0) / total_conf,
            'low_confidence_pct': confidence_counts.get('Low', 0) / total_conf,
            # Success metrics
            'valid_results': len(valid_results),
            'total_attempts': len(model_results),
            'success_rate': len(valid_results) / len(model_results)
        })

    return pd.DataFrame(metrics_by_model).sort_values('f1_score', ascending=False)


def create_model_comparison_visualizations(metrics_df: pd.DataFrame, results: List[LocalModelTestResult]):
    """Create comprehensive visualizations including RAGAS metrics"""

    # 1. Main performance comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Classification vs RAGAS Performance', 'Speed vs Accuracy',
                        'RAGAS Metrics Breakdown', 'Model Efficiency Analysis')
    )

    # Classification vs RAGAS score
    fig.add_trace(
        go.Bar(
            x=metrics_df['model_name'],
            y=metrics_df['f1_score'],
            name='F1 Score',
            marker_color='blue',
            yaxis='y'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_df['model_name'],
            y=metrics_df['ragas_score'],
            name='RAGAS Score',
            mode='lines+markers',
            marker_color='red',
            yaxis='y2'
        ),
        row=1, col=1
    )

    # Speed vs Accuracy
    fig.add_trace(
        go.Scatter(
            x=metrics_df['avg_generation_time'],
            y=metrics_df['accuracy'],
            mode='markers+text',
            text=metrics_df['model_name'],
            textposition='top center',
            marker=dict(
                size=metrics_df['tokens_per_second'] / 10,
                color=metrics_df['ragas_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="RAGAS Score", x=1.1)
            ),
            name='Models'
        ),
        row=1, col=2
    )

    # RAGAS Metrics Breakdown
    ragas_metrics = ['faithfulness', 'answer_correctness', 'context_precision', 'answer_relevancy']
    for metric in ragas_metrics:
        fig.add_trace(
            go.Bar(
                x=metrics_df['model_name'],
                y=metrics_df[metric],
                name=metric.replace('_', ' ').title(),
                text=metrics_df[metric].apply(lambda x: f'{x:.3f}'),
                textposition='auto'
            ),
            row=2, col=1
        )

    # Efficiency Score
    metrics_df['efficiency_score'] = (metrics_df['f1_score'] * metrics_df['ragas_score']) / metrics_df[
        'avg_generation_time']

    fig.add_trace(
        go.Bar(
            x=metrics_df['model_name'],
            y=metrics_df['efficiency_score'],
            text=metrics_df['efficiency_score'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
            name='Efficiency Score'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1000,
        title_text="Local Model Performance Comparison with RAGAS Metrics",
        showlegend=True,
        yaxis=dict(title="F1 Score", side="left"),
        yaxis2=dict(title="RAGAS Score", side="right", overlaying="y", range=[0, 1])
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Avg Generation Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency Score", row=2, col=2)

    fig.write_html(os.path.join(VISUALIZATIONS_DIR, "local_model_ragas_comparison.html"))

    # 2. RAGAS Metrics Heatmap
    plt.figure(figsize=(12, 8))

    ragas_columns = ['faithfulness', 'answer_relevancy', 'context_precision',
                     'context_recall', 'answer_similarity', 'answer_correctness']
    heatmap_data = metrics_df.set_index('model_name')[ragas_columns].T

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'})
    plt.title('RAGAS Metrics Heatmap by Model')
    plt.xlabel('Model')
    plt.ylabel('RAGAS Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "ragas_metrics_heatmap.png"), dpi=300)
    plt.close()


def create_detailed_report(metrics_df: pd.DataFrame, results: List[LocalModelTestResult]):
    """Create a detailed markdown report with RAGAS analysis"""

    report = []
    report.append("# Local Model RAGAS Evaluation Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    if not metrics_df.empty:
        best_f1 = metrics_df.loc[metrics_df['f1_score'].idxmax()]
        best_ragas = metrics_df.loc[metrics_df['ragas_score'].idxmax()]
        best_efficiency = metrics_df.loc[metrics_df['efficiency_score'].idxmax()]

        report.append(f"**Best F1 Score**: {best_f1['model_name']} ({best_f1['f1_score']:.3f})")
        report.append(f"**Best RAGAS Score**: {best_ragas['model_name']} ({best_ragas['ragas_score']:.3f})")
        report.append(
            f"**Most Efficient**: {best_efficiency['model_name']} (score: {best_efficiency['efficiency_score']:.3f})\n")

    # Model Rankings
    report.append("## Model Rankings\n")
    report.append("| Rank | Model | Size | F1 Score | RAGAS Score | Accuracy | Avg Time (s) | Success Rate |")
    report.append("|------|-------|------|----------|-------------|----------|--------------|--------------|")

    for idx, row in metrics_df.iterrows():
        report.append(f"| {idx + 1} | {row['model_name']} | {row['model_size']} | "
                      f"{row['f1_score']:.3f} | {row['ragas_score']:.3f} | "
                      f"{row['accuracy']:.3f} | {row['avg_generation_time']:.2f} | "
                      f"{row['success_rate']:.1%} |")

    # RAGAS Metrics Analysis
    report.append("\n## RAGAS Metrics Analysis\n")
    report.append(
        "| Model | Faithfulness | Answer Correctness | Context Precision | Answer Relevancy | Answer Similarity |")
    report.append(
        "|-------|--------------|-------------------|-------------------|------------------|-------------------|")

    for _, row in metrics_df.iterrows():
        report.append(f"| {row['model_name']} | {row['faithfulness']:.3f} | "
                      f"{row['answer_correctness']:.3f} | {row['context_precision']:.3f} | "
                      f"{row['answer_relevancy']:.3f} | {row['answer_similarity']:.3f} |")

    # Save report
    report_path = os.path.join(RESULTS_DIR, "local_model_ragas_evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"📄 Detailed report saved to: {report_path}")


def save_detailed_results(results: List[LocalModelTestResult]):
    """Save detailed results including RAGAS metrics to CSV"""

    detailed_data = []

    for r in results:
        row = {
            'item_index': r.item_index,
            'model_id': r.model_id,
            'model_name': r.model_name,
            'true_label': r.true_label,
            'predicted_label': r.predicted_label,
            'correct': r.true_label == r.predicted_label,
            'confidence': r.confidence,
            'generation_time': r.generation_time,
            'total_time': r.total_time,
            'prompt_tokens': r.prompt_tokens,
            'response_tokens': r.response_tokens,
            'error': r.error,
            'timestamp': r.timestamp
        }

        # Add RAGAS metrics if available
        if r.ragas_metrics:
            row.update({
                'faithfulness': r.ragas_metrics.faithfulness,
                'answer_relevancy': r.ragas_metrics.answer_relevancy,
                'context_precision': r.ragas_metrics.context_precision,
                'context_recall': r.ragas_metrics.context_recall,
                'answer_similarity': r.ragas_metrics.answer_similarity,
                'answer_correctness': r.ragas_metrics.answer_correctness
            })

        # Add retrieval scores
        for i, score in enumerate(r.retrieval_scores[:3]):
            row[f'retrieval_score_{i + 1}'] = score

        detailed_data.append(row)

    # Save to CSV
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(RESULTS_DIR, "local_model_detailed_results.csv"), index=False)


def load_data():
    """Load all required data files"""
    target_persons = []
    try:
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            target_persons = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading keywords: {e}")
        return None, None, None

    try:
        with open(TRAINING_SET_FILE, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None, None, None

    try:
        with open(VALIDATION_SET_FILE, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None, None, None

    return target_persons, training_data, validation_data


def main():
    parser = argparse.ArgumentParser(description='Test local models with Ollama and RAGAS metrics')
    parser.add_argument('--test-count', type=int, default=None,
                        help='Number of items to test (default: all)')
    parser.add_argument('--skip-models', nargs='+', default=[],
                        help='Model IDs to skip')
    args = parser.parse_args()

    print("🚀 Starting Local Model Testing with Ollama and RAGAS Metrics")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Using optimal RAG parameters: {OPTIMAL_PARAMS}")
    print("=" * 70)

    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set. RAGAS evaluation will be limited.")
        print("   Set OPENAI_API_KEY to enable full RAGAS metrics evaluation.")

    # Check Ollama
    if not check_ollama_running():
        print("❌ Ollama is not running. Please install Ollama from https://ollama.ai")
        return

    # Initialize encoder
    initialize_global_encoder()
    print(f"💾 Initial memory: {get_memory_info()}")

    # Load data
    print("\n📂 Loading data...")
    target_persons, training_data, validation_data = load_data()

    if not all([target_persons, training_data, validation_data]):
        print("❌ Failed to load required data. Exiting.")
        return

    print(f"✅ Loaded {len(target_persons)} target persons")
    print(f"✅ Loaded {len(training_data)} training documents")
    print(f"✅ Loaded {len(validation_data)} validation items")

    # Build knowledge base
    print("\n🏗️ Building knowledge base with hybrid retrieval...")
    rag_kb = OptimizedRAGKnowledgeBase()
    rag_kb.build_knowledge_base(training_data, use_cache=True)

    if not rag_kb.is_built:
        print("❌ Failed to build knowledge base. Exiting.")
        return

    print(f"✅ Knowledge base ready with {len(rag_kb.chunks)} chunks")
    print(f"💾 Memory after KB: {get_memory_info()}")

    # Select models to test
    models_to_test = [m for m in LOCAL_MODELS if m.id not in args.skip_models]

    print(f"\n📋 Local models to test: {len(models_to_test)}")
    for model in models_to_test:
        print(f"   - {model.name} ({model.size})")

    # Run tests
    print("\n🧪 Starting model testing...")
    start_time = time.time()

    results = test_models(
        models_to_test,
        validation_data,
        rag_kb,
        target_persons,
        max_items=args.test_count
    )

    total_time = time.time() - start_time
    print(f"\n✅ Testing complete in {total_time / 60:.2f} minutes")
    print(f"💾 Final memory: {get_memory_info()}")

    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics_df = calculate_model_metrics(results, target_persons)

    # Save results
    save_detailed_results(results)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "local_model_metrics_summary.csv"), index=False)

    # Create visualizations
    print("📈 Creating visualizations...")
    create_model_comparison_visualizations(metrics_df, results)

    # Create report
    create_detailed_report(metrics_df, results)

    # Print summary
    print("\n" + "=" * 70)
    print("🏆 LOCAL MODEL TESTING RESULTS WITH RAGAS")
    print("=" * 70)

    if not metrics_df.empty:
        print("\n### Top Models by F1 Score ###")
        for idx, row in metrics_df.head(3).iterrows():
            print(f"\nRank {idx + 1}: {row['model_name']} ({row['model_size']})")
            print(f"  F1 Score: {row['f1_score']:.3f}")
            print(f"  RAGAS Score: {row['ragas_score']:.3f}")
            print(f"  Accuracy: {row['accuracy']:.3f}")
            print(f"  Avg Time: {row['avg_generation_time']:.2f}s")

        print("\n### Best RAGAS Performance ###")
        best_ragas = metrics_df.loc[metrics_df['ragas_score'].idxmax()]
        print(f"{best_ragas['model_name']} ({best_ragas['model_size']})")
        print(f"  RAGAS Score: {best_ragas['ragas_score']:.3f}")
        print(f"  Faithfulness: {best_ragas['faithfulness']:.3f}")
        print(f"  Answer Correctness: {best_ragas['answer_correctness']:.3f}")
        print(f"  Context Precision: {best_ragas['context_precision']:.3f}")

    print(f"\n💾 All results saved to: {RESULTS_DIR}/")
    print(f"📊 Visualizations saved to: {VISUALIZATIONS_DIR}/")
    print("\n✅ Local model testing complete!")


if __name__ == "__main__":
    main()