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
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import torch
import gc
import psutil
from tqdm import tqdm
import subprocess
import ollama
from scipy.stats import pearsonr, spearmanr
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Import required packages
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing required packages...")
    pip_command = [sys.executable, "-m", "pip", "install", "sentence-transformers", "ollama", "tqdm"]
    subprocess.check_call(pip_command)
    from sentence_transformers import SentenceTransformer

# Configuration
KEYWORDS_FILE = 'keywords.txt'
TRAINING_SET_FILE = 'training_set.json'
TEST_SET_FILE = 'testing_set.json'  # Using test set for final evaluation
RESULTS_DIR = 'final_evaluation_results'
VISUALIZATIONS_DIR = 'final_evaluation_visualizations'

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

# Llama3-8B configuration
LLAMA3_CONFIG = {
    'model_name': 'llama3:8b',
    'temperature': 0.1,
    'max_tokens': 2000,
    'context_length': 8192
}

# Global variables
GLOBAL_ENCODER = None
DEVICE = None


@dataclass
class EvaluationResult:
    item_index: int
    true_label: str
    predicted_label: str
    confidence: str
    confidence_numeric: float  # Mapped confidence value
    explanation: str
    retrieval_scores: List[float]
    processing_time: float
    timestamp: str
    title: str
    url: str
    correct: bool


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

        self.cache_file = os.path.join(RESULTS_DIR, "kb_cache_final_eval.pkl")

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

    def build_knowledge_base(self, training_data: List[Dict], use_cache: bool = True):
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

        print("Building knowledge base...")
        self.chunks = []
        self.person_mapping = {}
        all_texts = []

        for item in tqdm(training_data, desc="Processing documents"):
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
            boosted_scores.append(final_score)

        sorted_results = sorted(zip(boosted_results, boosted_scores), key=lambda x: x[1], reverse=True)
        boosted_results = [r[0] for r in sorted_results]
        boosted_scores = [r[1] for r in sorted_results]

        return boosted_results[:self.top_k_retrieval], boosted_scores[:self.top_k_retrieval]

    def get_context_for_prompt(self, query: str, target_persons: List[str]) -> Tuple[str, List[float]]:
        results, scores = self.hybrid_retrieval(query, target_persons)
        if not results:
            return "No relevant contextual information found.", []

        context_parts = []
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

        return "\n".join(context_parts), scores


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
        GLOBAL_ENCODER = SentenceTransformer('sentence-transformers/LaBSE', device=DEVICE)
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


import json
import subprocess
import ollama # Ensure ollama is imported
import logging # Ensure logging is imported
# ... (other imports like LLAMA3_CONFIG from your script)

def check_ollama_and_model():
    """Check if Ollama is running and the required model is available."""
    global LLAMA3_CONFIG
    try:
        logging.info("ℹ️ Checking Ollama status and models...")
        response = ollama.list()

        # Debugging (optional, keep commented unless needed)
        # if hasattr(response, 'model_dump_json'):
        #     logging.debug(f"Raw response from ollama.list(): {response.model_dump_json(indent=2)}")
        # elif hasattr(response, 'dict'):
        #     logging.debug(f"Raw response from ollama.list(): {json.dumps(response.dict(), indent=2)}")
        # else:
        #     logging.debug(f"Raw response object from ollama.list(): {response}")

        if not hasattr(response, 'models') or not isinstance(response.models, list):
            logging.error("❌ Error: Unexpected response structure from ollama.list(). 'models' attribute missing or not a list.")
            logging.error(f"   Received response object: {response}")
            return False

        available_model_names = []
        malformed_model_entries = 0
        for model_info in response.models:
            # *** CHANGED HERE: Use model_info.model instead of model_info.name ***
            if hasattr(model_info, 'model') and isinstance(model_info.model, str) and model_info.model:
                available_model_names.append(model_info.model)
            else:
                malformed_model_entries += 1
                # Update warning message to reflect checking for '.model'
                logging.warning(f"⚠️ Warning: Encountered a malformed model entry (missing 'model' attribute, 'model' is not a string, or empty): {model_info}")

        if malformed_model_entries > 0:
            logging.info(f"ℹ️ {malformed_model_entries} out of {len(response.models)} model entries were malformed or incomplete based on '.model' attribute.")

        required_model = LLAMA3_CONFIG['model_name']
        if required_model not in available_model_names:
            logging.info(f"ℹ️ Model '{required_model}' not found in available models: {available_model_names}")
            logging.info(f"📥 Attempting to pull '{required_model}'... This may take a while.")
            try:
                current_status = ""
                for progress in ollama.pull(required_model, stream=True):
                    if 'status' in progress:
                        status_detail = progress['status']
                        if 'total' in progress and 'completed' in progress and progress.get('total', 0) > 0:
                            percentage = (progress['completed'] / progress['total']) * 100
                            current_status = f"{status_detail} {percentage:.2f}%"
                        else:
                            current_status = status_detail
                        print(f"\r   Pulling status: {current_status.ljust(60)}", end="")
                    if 'error' in progress:
                        print()
                        logging.error(f"\n❌ Error during model pull: {progress['error']}")
                        return False
                print()
                logging.info(f"✅ Model pull stream finished for '{required_model}'. Verifying...")

                updated_response = ollama.list()
                if hasattr(updated_response, 'models') and isinstance(updated_response.models, list):
                    # *** CHANGED HERE: Use m.model in list comprehension ***
                    updated_model_names = [m.model for m in updated_response.models if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
                    if required_model in updated_model_names:
                        logging.info(f"✅ Model '{required_model}' is now available.")
                    else:
                        # This case should be less likely now if the pull was truly successful and parsing is correct
                        logging.error(f"❌ Model '{required_model}' still not found after pull attempt and re-check.")
                        logging.error(f"   Available models now (using .model attribute): {updated_model_names}")
                        return False
                else:
                    logging.error("❌ Could not verify models after pull attempt due to unexpected response from ollama.list().")
                    return False

            except Exception as e_pull:
                print()
                logging.error(f"\n❌ Error occurred while trying to pull '{required_model}': {e_pull}")
                return False
        else:
            logging.info(f"✅ Model '{required_model}' is available.")

        return True

    except ollama.ResponseError as e_res:
        logging.error(f"❌ Ollama API Response Error: Status Code {e_res.status_code}, Error: {e_res.error}")
        logging.error(f"   Please ensure Ollama (http://127.0.0.1:11434) is installed, running, and accessible.")
        return False
    except ConnectionRefusedError:
        logging.error(f"❌ Connection Error: Could not connect to Ollama at http://127.0.0.1:11434.")
        logging.error(f"   Please ensure Ollama is installed and running.")
        return False
    except Exception as e_gen:
        import traceback
        logging.error(f"❌ An unexpected error occurred in check_ollama_and_model: {type(e_gen).__name__} - {e_gen}")
        logging.error("   Full traceback:")
        logging.error(traceback.format_exc())
        logging.error("   This might be due to Ollama not running, a network issue, or an unexpected API response format.")
        return False


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


def map_confidence_to_numeric(confidence: str) -> float:
    """Map confidence strings to numeric values"""
    confidence_map = {
        'high': 0.9,
        'medium': 0.6,
        'low': 0.3,
        'n/a': 0.5
    }
    return confidence_map.get(confidence.lower(), 0.5)


def evaluate_model_on_test_set(
        test_data: List[Dict],
        rag_kb: OptimizedRAGKnowledgeBase,
        target_persons: List[str]
) -> List[EvaluationResult]:
    """Evaluate llama3:8b on the test set"""

    results = []

    print("\n🧪 Starting evaluation on test set...")

    for idx, test_item in tqdm(enumerate(test_data), total=len(test_data), desc="Evaluating"):
        try:
            anonymized_data = test_item.get('anonymized', {})
            anonymized_text = anonymized_data.get('full_text', '')
            ground_truth = anonymized_data.get('keywords', [])[0] if anonymized_data.get('keywords') else 'Unknown'
            title = anonymized_data.get('title', 'N/A')
            url = anonymized_data.get('url', 'N/A')

            if not anonymized_text:
                continue

            # Get context
            context, retrieval_scores = rag_kb.get_context_for_prompt(
                anonymized_text,
                anonymized_data.get('keywords', [])
            )

            # Format prompt
            prompt = OPTIMAL_PROMPT_TEMPLATE.format(
                text=anonymized_text,
                context=context,
                targets=', '.join(target_persons)
            )

            # Truncate if needed
            if len(prompt) > LLAMA3_CONFIG['context_length'] * 4:
                max_text_length = LLAMA3_CONFIG['context_length'] * 3
                truncated_text = anonymized_text[:max_text_length] + "... [TRUNCATED]"
                prompt = OPTIMAL_PROMPT_TEMPLATE.format(
                    text=truncated_text,
                    context=context,
                    targets=', '.join(target_persons)
                )

            # Call model
            start_time = time.time()
            response = ollama.generate(
                model=LLAMA3_CONFIG['model_name'],
                prompt=prompt,
                options={
                    'temperature': LLAMA3_CONFIG['temperature'],
                    'num_predict': LLAMA3_CONFIG['max_tokens'],
                }
            )
            processing_time = time.time() - start_time

            # Parse response
            response_text = response['response']
            identified_person, explanation, confidence = parse_model_response(response_text)

            # Create result
            result = EvaluationResult(
                item_index=idx,
                true_label=ground_truth,
                predicted_label=identified_person,
                confidence=confidence,
                confidence_numeric=map_confidence_to_numeric(confidence),
                explanation=explanation,
                retrieval_scores=retrieval_scores,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                title=title,
                url=url,
                correct=(ground_truth.lower() == identified_person.lower())
            )
            results.append(result)

        except Exception as e:
            print(f"\nError processing item {idx}: {e}")
            continue

    return results


def calculate_comprehensive_metrics(results: List[EvaluationResult], target_persons: List[str]) -> Dict:
    """Calculate all classification metrics"""

    # Prepare data
    y_true = [r.true_label for r in results]
    y_pred = [r.predicted_label for r in results]

    # Get all unique labels
    all_labels = sorted(list(set(y_true + y_pred + target_persons + ['Uncertain'])))

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Macro and weighted averages
    precision_macro = precision_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    # Other metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Confidence analysis
    confidence_levels = ['High', 'Medium', 'Low']
    confidence_accuracy = {}

    for level in confidence_levels:
        level_results = [r for r in results if r.confidence.lower() == level.lower()]
        if level_results:
            level_correct = sum(1 for r in level_results if r.correct)
            confidence_accuracy[level] = level_correct / len(level_results)
        else:
            confidence_accuracy[level] = 0.0

    # Confidence correlation
    confidence_numeric = [r.confidence_numeric for r in results]
    correct_binary = [1 if r.correct else 0 for r in results]

    if len(set(confidence_numeric)) > 1:  # Need variation for correlation
        pearson_corr, pearson_p = pearsonr(confidence_numeric, correct_binary)
        spearman_corr, spearman_p = spearmanr(confidence_numeric, correct_binary)
    else:
        pearson_corr = pearson_p = spearman_corr = spearman_p = 0.0

    # Processing time stats
    processing_times = [r.processing_time for r in results]

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'cohen_kappa': kappa,
        'total_predictions': len(results),
        'correct_predictions': sum(1 for r in results if r.correct),
        'confusion_matrix': cm,
        'labels': all_labels,
        'per_class_precision': dict(zip(all_labels, precision_per_class)),
        'per_class_recall': dict(zip(all_labels, recall_per_class)),
        'per_class_f1': dict(zip(all_labels, f1_per_class)),
        'per_class_support': dict(zip(all_labels, support_per_class)),
        'confidence_accuracy': confidence_accuracy,
        'confidence_correlation_pearson': pearson_corr,
        'confidence_correlation_pearson_p': pearson_p,
        'confidence_correlation_spearman': spearman_corr,
        'confidence_correlation_spearman_p': spearman_p,
        'avg_processing_time': np.mean(processing_times),
        'std_processing_time': np.std(processing_times),
        'min_processing_time': np.min(processing_times),
        'max_processing_time': np.max(processing_times)
    }


def create_visualizations(metrics: Dict, results: List[EvaluationResult]):
    """Create all visualization plots"""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    cm = metrics['confusion_matrix']
    labels = metrics['labels']

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

    # Create annotations with both count and percentage
    annotations = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            if count > 0:
                row.append(f'{count}\n({percentage:.1f}%)')
            else:
                row.append('0\n(0.0%)')
        annotations.append(row)

    sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized Frequency'})
    plt.title('Confusion Matrix - Llama3-8B Final Evaluation', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # 2. Per-Class Metrics Bar Chart
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Prepare data
    labels_plot = metrics['labels']
    precision_values = [metrics['per_class_precision'][label] for label in labels_plot]
    recall_values = [metrics['per_class_recall'][label] for label in labels_plot]
    f1_values = [metrics['per_class_f1'][label] for label in labels_plot]
    support_values = [metrics['per_class_support'][label] for label in labels_plot]

    # Precision plot
    ax1 = axes[0]
    bars1 = ax1.bar(labels_plot, precision_values, color='skyblue', edgecolor='black')
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Per-Class Precision', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=metrics['precision_macro'], color='red', linestyle='--',
                label=f'Macro Avg: {metrics["precision_macro"]:.3f}')
    ax1.axhline(y=metrics['precision_weighted'], color='green', linestyle='--',
                label=f'Weighted Avg: {metrics["precision_weighted"]:.3f}')
    ax1.legend()

    # Add value labels
    for bar, value in zip(bars1, precision_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # Recall plot
    ax2 = axes[1]
    bars2 = ax2.bar(labels_plot, recall_values, color='lightcoral', edgecolor='black')
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Per-Class Recall', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=metrics['recall_macro'], color='red', linestyle='--',
                label=f'Macro Avg: {metrics["recall_macro"]:.3f}')
    ax2.axhline(y=metrics['recall_weighted'], color='green', linestyle='--',
                label=f'Weighted Avg: {metrics["recall_weighted"]:.3f}')
    ax2.legend()

    for bar, value in zip(bars2, recall_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # F1 plot
    ax3 = axes[2]
    bars3 = ax3.bar(labels_plot, f1_values, color='lightgreen', edgecolor='black')
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_title('Per-Class F1 Score', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.axhline(y=metrics['f1_macro'], color='red', linestyle='--', label=f'Macro Avg: {metrics["f1_macro"]:.3f}')
    ax3.axhline(y=metrics['f1_weighted'], color='green', linestyle='--',
                label=f'Weighted Avg: {metrics["f1_weighted"]:.3f}')
    ax3.legend()

    for bar, value, support in zip(bars3, f1_values, support_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}\n(n={support})', ha='center', va='bottom', fontsize=9)

    # Rotate x labels for all subplots
    for ax in axes:
        ax.set_xticklabels(labels_plot, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'per_class_metrics.png'), bbox_inches='tight')
    plt.close()

    # 3. Confidence Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confidence level accuracy
    confidence_levels = list(metrics['confidence_accuracy'].keys())
    confidence_acc_values = list(metrics['confidence_accuracy'].values())

    bars = ax1.bar(confidence_levels, confidence_acc_values,
                   color=['green', 'orange', 'red'], edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Confidence Level', fontsize=12)
    ax1.set_title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=metrics['accuracy'], color='blue', linestyle='--',
                label=f'Overall Accuracy: {metrics["accuracy"]:.3f}')
    ax1.legend()

    # Add value labels
    for bar, value in zip(bars, confidence_acc_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=12)

    # Confidence distribution
    confidence_counts = pd.Series([r.confidence for r in results]).value_counts()
    ax2.pie(confidence_counts.values, labels=confidence_counts.index,
            autopct='%1.1f%%', startangle=90, colors=['green', 'orange', 'red'])
    ax2.set_title('Distribution of Confidence Levels', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'confidence_analysis.png'), bbox_inches='tight')
    plt.close()

    # 4. Label Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # True label distribution
    true_labels = [r.true_label for r in results]
    true_label_counts = pd.Series(true_labels).value_counts()

    ax1.bar(true_label_counts.index, true_label_counts.values, color='skyblue', edgecolor='black')
    ax1.set_xlabel('True Label', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of True Labels', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(true_label_counts.index, rotation=45, ha='right')

    # Add count labels
    for i, (label, count) in enumerate(true_label_counts.items()):
        ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')

    # Predicted label distribution
    pred_labels = [r.predicted_label for r in results]
    pred_label_counts = pd.Series(pred_labels).value_counts()

    ax2.bar(pred_label_counts.index, pred_label_counts.values, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predicted Labels', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(pred_label_counts.index, rotation=45, ha='right')

    # Add count labels
    for i, (label, count) in enumerate(pred_label_counts.items()):
        ax2.text(i, count + 0.5, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'label_distribution.png'), bbox_inches='tight')
    plt.close()

    # 5. Overall Performance Summary
    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare summary metrics
    summary_metrics = {
        'Accuracy': metrics['accuracy'],
        'Balanced Accuracy': metrics['balanced_accuracy'],
        'F1 (Macro)': metrics['f1_macro'],
        'F1 (Weighted)': metrics['f1_weighted'],
        'Precision (Macro)': metrics['precision_macro'],
        'Precision (Weighted)': metrics['precision_weighted'],
        'Recall (Macro)': metrics['recall_macro'],
        'Recall (Weighted)': metrics['recall_weighted'],
        'MCC': metrics['mcc'],
        "Cohen's Kappa": metrics['cohen_kappa']
    }

    # Create horizontal bar chart
    y_pos = np.arange(len(summary_metrics))
    values = list(summary_metrics.values())

    bars = ax.barh(y_pos, values, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(summary_metrics.keys()))
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Overall Performance Metrics - Llama3-8B', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)

    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{value:.3f}', ha='left', va='center', fontsize=11)

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'overall_performance_summary.png'), bbox_inches='tight')
    plt.close()

    print("✅ All visualizations created successfully!")


def save_results_to_csv(results: List[EvaluationResult], metrics: Dict):
    """Save detailed results and summary to CSV files"""

    # 1. Detailed results CSV
    detailed_data = []
    for r in results:
        detailed_data.append({
            'item_index': r.item_index,
            'title': r.title,
            'url': r.url,
            'true_label': r.true_label,
            'predicted_label': r.predicted_label,
            'correct': r.correct,
            'confidence': r.confidence,
            'confidence_numeric': r.confidence_numeric,
            'processing_time': r.processing_time,
            'retrieval_score_1': r.retrieval_scores[0] if len(r.retrieval_scores) > 0 else 0,
            'retrieval_score_2': r.retrieval_scores[1] if len(r.retrieval_scores) > 1 else 0,
            'retrieval_score_3': r.retrieval_scores[2] if len(r.retrieval_scores) > 2 else 0,
            'timestamp': r.timestamp
        })

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(RESULTS_DIR, 'llama3_8b_detailed_results.csv'), index=False)

    # 2. Summary metrics CSV
    summary_data = {
        'Model': 'Llama3-8B',
        'Total Predictions': metrics['total_predictions'],
        'Correct Predictions': metrics['correct_predictions'],
        'Accuracy': metrics['accuracy'],
        'Balanced Accuracy': metrics['balanced_accuracy'],
        'Precision (Macro)': metrics['precision_macro'],
        'Recall (Macro)': metrics['recall_macro'],
        'F1 Score (Macro)': metrics['f1_macro'],
        'Precision (Weighted)': metrics['precision_weighted'],
        'Recall (Weighted)': metrics['recall_weighted'],
        'F1 Score (Weighted)': metrics['f1_weighted'],
        'Matthews Correlation Coefficient': metrics['mcc'],
        "Cohen's Kappa": metrics['cohen_kappa'],
        'Confidence Accuracy (High)': metrics['confidence_accuracy'].get('High', 0),
        'Confidence Accuracy (Medium)': metrics['confidence_accuracy'].get('Medium', 0),
        'Confidence Accuracy (Low)': metrics['confidence_accuracy'].get('Low', 0),
        'Confidence Correlation (Pearson)': metrics['confidence_correlation_pearson'],
        'Confidence Correlation (Spearman)': metrics['confidence_correlation_spearman'],
        'Avg Processing Time (s)': metrics['avg_processing_time'],
        'Std Processing Time (s)': metrics['std_processing_time'],
        'Min Processing Time (s)': metrics['min_processing_time'],
        'Max Processing Time (s)': metrics['max_processing_time']
    }

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'llama3_8b_summary_metrics.csv'), index=False)

    # 3. Per-class metrics CSV
    per_class_data = []
    for label in metrics['labels']:
        per_class_data.append({
            'Class': label,
            'Precision': metrics['per_class_precision'][label],
            'Recall': metrics['per_class_recall'][label],
            'F1 Score': metrics['per_class_f1'][label],
            'Support': metrics['per_class_support'][label]
        })

    per_class_df = pd.DataFrame(per_class_data)
    per_class_df.to_csv(os.path.join(RESULTS_DIR, 'llama3_8b_per_class_metrics.csv'), index=False)

    # 4. Confusion matrix CSV
    cm_df = pd.DataFrame(metrics['confusion_matrix'],
                         columns=metrics['labels'],
                         index=metrics['labels'])
    cm_df.to_csv(os.path.join(RESULTS_DIR, 'llama3_8b_confusion_matrix.csv'))

    print("✅ All CSV files saved successfully!")


def create_evaluation_report(metrics: Dict, results: List[EvaluationResult]):
    """Create a comprehensive evaluation report"""

    report = []
    report.append("# Llama3-8B Final Evaluation Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"**Model**: Llama3-8B")
    report.append(f"**Test Set Size**: {metrics['total_predictions']} samples")
    report.append(f"**Correct Predictions**: {metrics['correct_predictions']} ({metrics['accuracy']:.1%})")
    report.append(f"**F1 Score (Weighted)**: {metrics['f1_weighted']:.3f}")
    report.append(f"**Matthews Correlation Coefficient**: {metrics['mcc']:.3f}")
    report.append(f"**Cohen's Kappa**: {metrics['cohen_kappa']:.3f}\n")

    # Overall Performance Metrics
    report.append("## Overall Performance Metrics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Accuracy | {metrics['accuracy']:.3f} |")
    report.append(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.3f} |")
    report.append(f"| Precision (Macro) | {metrics['precision_macro']:.3f} |")
    report.append(f"| Recall (Macro) | {metrics['recall_macro']:.3f} |")
    report.append(f"| F1 Score (Macro) | {metrics['f1_macro']:.3f} |")
    report.append(f"| Precision (Weighted) | {metrics['precision_weighted']:.3f} |")
    report.append(f"| Recall (Weighted) | {metrics['recall_weighted']:.3f} |")
    report.append(f"| F1 Score (Weighted) | {metrics['f1_weighted']:.3f} |")
    report.append(f"| MCC | {metrics['mcc']:.3f} |")
    report.append(f"| Cohen's Kappa | {metrics['cohen_kappa']:.3f} |\n")

    # Confidence Analysis
    report.append("## Confidence Analysis\n")
    report.append("### Accuracy by Confidence Level\n")
    report.append("| Confidence Level | Accuracy | Samples |")
    report.append("|-----------------|----------|---------|")

    for level in ['High', 'Medium', 'Low']:
        level_results = [r for r in results if r.confidence.lower() == level.lower()]
        count = len(level_results)
        acc = metrics['confidence_accuracy'].get(level, 0)
        report.append(f"| {level} | {acc:.3f} | {count} |")

    report.append(f"\n**Confidence Correlation with Accuracy**:")
    report.append(
        f"- Pearson: {metrics['confidence_correlation_pearson']:.3f} (p={metrics['confidence_correlation_pearson_p']:.3f})")
    report.append(
        f"- Spearman: {metrics['confidence_correlation_spearman']:.3f} (p={metrics['confidence_correlation_spearman_p']:.3f})\n")

    # Per-Class Performance
    report.append("## Per-Class Performance\n")
    report.append("| Class | Precision | Recall | F1 Score | Support |")
    report.append("|-------|-----------|--------|----------|---------|")

    for label in metrics['labels']:
        report.append(f"| {label} | {metrics['per_class_precision'][label]:.3f} | "
                      f"{metrics['per_class_recall'][label]:.3f} | "
                      f"{metrics['per_class_f1'][label]:.3f} | "
                      f"{metrics['per_class_support'][label]} |")

    # Processing Time Statistics
    report.append("\n## Processing Time Statistics\n")
    report.append(f"- Average: {metrics['avg_processing_time']:.2f}s")
    report.append(f"- Std Dev: {metrics['std_processing_time']:.2f}s")
    report.append(f"- Minimum: {metrics['min_processing_time']:.2f}s")
    report.append(f"- Maximum: {metrics['max_processing_time']:.2f}s")

    # Error Analysis
    report.append("\n## Error Analysis\n")

    # Most confused pairs
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    confused_pairs = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((true_label, pred_label, cm[i][j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    report.append("### Most Confused Pairs\n")
    report.append("| True Label | Predicted As | Count |")
    report.append("|------------|--------------|-------|")

    for true_label, pred_label, count in confused_pairs[:10]:
        report.append(f"| {true_label} | {pred_label} | {count} |")

    # Save report
    report_path = os.path.join(RESULTS_DIR, 'llama3_8b_evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"📄 Evaluation report saved to: {report_path}")


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
        with open(TEST_SET_FILE, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None, None

    return target_persons, training_data, test_data


def main():
    print("🚀 Starting Final Evaluation of Llama3-8B")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check Ollama and model
    if not check_ollama_and_model():
        print("❌ Ollama setup failed. Exiting.")
        return

    # Initialize encoder
    initialize_global_encoder()
    print(f"💾 Initial memory: {get_memory_info()}")

    # Load data
    print("\n📂 Loading data...")
    target_persons, training_data, test_data = load_data()

    if not all([target_persons, training_data, test_data]):
        print("❌ Failed to load required data. Exiting.")
        return

    print(f"✅ Loaded {len(target_persons)} target persons")
    print(f"✅ Loaded {len(training_data)} training documents")
    print(f"✅ Loaded {len(test_data)} test samples")

    # Build knowledge base
    print("\n🏗️ Building knowledge base...")
    rag_kb = OptimizedRAGKnowledgeBase()
    rag_kb.build_knowledge_base(training_data, use_cache=True)

    if not rag_kb.is_built:
        print("❌ Failed to build knowledge base. Exiting.")
        return

    print(f"✅ Knowledge base ready with {len(rag_kb.chunks)} chunks")
    print(f"💾 Memory after KB: {get_memory_info()}")

    # Evaluate model
    start_time = time.time()
    results = evaluate_model_on_test_set(test_data, rag_kb, target_persons)
    evaluation_time = time.time() - start_time

    print(f"\n✅ Evaluation complete in {evaluation_time / 60:.2f} minutes")
    print(f"💾 Final memory: {get_memory_info()}")

    # Calculate metrics
    print("\n📊 Calculating comprehensive metrics...")
    metrics = calculate_comprehensive_metrics(results, target_persons)

    # Create visualizations
    print("📈 Creating visualizations...")
    create_visualizations(metrics, results)

    # Save results
    print("💾 Saving results to CSV...")
    save_results_to_csv(results, metrics)

    # Create report
    print("📄 Creating evaluation report...")
    create_evaluation_report(metrics, results)

    # Print summary
    print("\n" + "=" * 70)
    print("🏆 LLAMA3-8B FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nTest Set Size: {metrics['total_predictions']} samples")
    print(f"Correct Predictions: {metrics['correct_predictions']} ({metrics['accuracy']:.1%})")
    print(f"\nKey Metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.3f}")
    print(f"  - Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  - F1 Score (Weighted): {metrics['f1_weighted']:.3f}")
    print(f"  - F1 Score (Macro): {metrics['f1_macro']:.3f}")
    print(f"  - MCC: {metrics['mcc']:.3f}")
    print(f"  - Cohen's Kappa: {metrics['cohen_kappa']:.3f}")

    print(f"\nConfidence Analysis:")
    for level, acc in metrics['confidence_accuracy'].items():
        print(f"  - {level} Confidence Accuracy: {acc:.3f}")

    print(f"\nProcessing Time:")
    print(f"  - Average: {metrics['avg_processing_time']:.2f}s per sample")
    print(f"  - Total: {evaluation_time:.2f}s")

    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print(f"Visualizations saved to: {VISUALIZATIONS_DIR}/")
    print("\n✅ Final evaluation complete!")


if __name__ == "__main__":
    main()