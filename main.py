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

warnings.filterwarnings('ignore')

# Try to import required packages, install if missing
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Installing required packages for advanced RAG...")
    os.system("pip install sentence-transformers scikit-learn pandas matplotlib seaborn plotly")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

# Configuration
KEYWORDS_FILE = 'keywords.txt'
TRAINING_SET_FILE = 'training_set.json'
VALIDATION_SET_FILE = 'validation_set.json'
EMBEDDINGS_CACHE_FILE_PREFIX = 'embeddings_cache'
RESULTS_DIR = 'rag_results'
VISUALIZATIONS_DIR = 'rag_visualizations'

# RAG Parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.25

# Gemini API Configuration
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Embedding Models Configuration
EMBEDDING_MODELS = {
    'multilingual-minilm': {
        'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'type': 'sentence-transformers',
        'description': 'Lightweight multilingual model (12 layers)',
        'dimensions': 384
    },
    'distiluse': {
        'name': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
        'type': 'sentence-transformers',
        'description': 'Fast and lightweight multilingual model supporting Lithuanian',
        'dimensions': 512
    },
    'multilingual-mpnet': {
        'name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'type': 'sentence-transformers',
        'description': 'Larger multilingual model with better performance',
        'dimensions': 768
    },
    'labse': {
        'name': 'sentence-transformers/LaBSE',
        'type': 'sentence-transformers',
        'description': 'Language-agnostic BERT for 109+ languages including Lithuanian',
        'dimensions': 768
    },
    'gte-base': {
        'name': 'thenlper/gte-base',
        'type': 'sentence-transformers',
        'description': 'Balanced multilingual embedding model optimized for general-purpose similarity tasks',
        'dimensions': 768
    },
    'e5-large': {
        'name': 'intfloat/multilingual-e5-large',
        'type': 'sentence-transformers',
        'description': 'High-dimensional multilingual embedding model (1024d) optimized for semantic search and entity linking',
        'dimensions': 1024
    }
}


@dataclass
class DocumentChunk:
    """Represents a chunk of anonymized text with metadata."""
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
    """Stores detailed classification result for a single test item."""
    item_index: int
    model_id: str
    true_label: str  # Main ground truth person
    predicted_label: str
    confidence: str
    explanation: str
    retrieval_scores: List[float]
    processing_time: float
    retrieval_time: float
    api_time: float
    all_ground_truth: List[str]  # All ground truth persons
    title: str
    url: str
    anonymized_text_length: int
    timestamp: str


class MultiEmbeddingRAGKnowledgeBase:
    """Advanced RAG system with multiple embedding model support."""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.model_type = model_config['type']
        self.model_description = model_config['description']

        print(f"\n🔧 Initializing {self.model_id}: {self.model_description}")
        print(f"   Model: {self.model_name}")

        # Load the embedding model
        if self.model_type == 'sentence-transformers':
            self.encoder = SentenceTransformer(self.model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False
        self.person_mapping = {}
        self.build_stats = {}
        self.cache_file = f"{EMBEDDINGS_CACHE_FILE_PREFIX}_{self.model_id}.pkl"

    def extract_person_mappings(self, replacements: List[Dict]) -> Dict[str, str]:
        """Extract person mappings from replacements data."""
        person_map = {}
        if replacements:
            for replacement in replacements:
                if replacement.get('type') == 'PERSON':
                    original = replacement.get('original', '')
                    replacement_id = replacement.get('replacement', '')
                    if original and replacement_id:
                        person_map[replacement_id] = original
        return person_map

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split Lithuanian text into overlapping chunks with sentence-aware boundaries."""
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

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks or max(len(chunk) for chunk in chunks) > chunk_size * 1.5:
            return self.fallback_chunk(text, chunk_size, overlap)

        return chunks

    def fallback_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback character-based chunking for very long texts."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for i in range(min(50, len(text) - end)):
                    if text[end - i] in ' \n\t':
                        end = end - i
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def build_knowledge_base(self, training_data: List[Dict], use_cache: bool = True):
        """Build advanced knowledge base from anonymized Lithuanian data."""
        start_time = time.time()
        print(f"📚 Building knowledge base for {self.model_id}...")

        # Check for cached embeddings
        if use_cache and os.path.exists(self.cache_file):
            try:
                print(f"   Loading cached embeddings from {self.cache_file}...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.chunks = cache_data['chunks']
                    self.embeddings = cache_data['embeddings']
                    self.person_mapping = cache_data.get('person_mapping', {})
                    self.build_stats = cache_data.get('build_stats', {})
                    self.is_built = True
                print(f"   ✅ Loaded {len(self.chunks)} cached document chunks")
                return
            except Exception as e:
                print(f"   Failed to load cache: {e}. Building from scratch...")

        # Build knowledge base from scratch
        all_texts = []
        chunk_id = 0
        processed_docs = 0
        skipped_docs = 0

        for item in training_data:
            try:
                anonymized_data = item.get('anonymized', {})
                full_text = anonymized_data.get('full_text', '')
                keywords = anonymized_data.get('keywords', [])
                title = anonymized_data.get('title', 'N/A')
                url = anonymized_data.get('url', 'N/A')
                replacements = item.get('replacements', [])

                if not full_text.strip():
                    skipped_docs += 1
                    continue

                # Extract person mappings for this document
                person_map = self.extract_person_mappings(replacements)
                self.person_mapping.update(person_map)

                # Find anonymized person references in text
                anonymized_persons = []
                for replacement_id in person_map.keys():
                    if replacement_id in full_text:
                        anonymized_persons.append(replacement_id)

                # Create chunks from the anonymized full text
                text_chunks = self.chunk_text(full_text)

                for chunk_text in text_chunks:
                    chunk_anonymized_persons = [p for p in anonymized_persons if p in chunk_text]

                    chunk = DocumentChunk(
                        text=chunk_text,
                        original_persons=keywords,
                        anonymized_persons=chunk_anonymized_persons,
                        source_url=url,
                        title=title,
                        chunk_id=chunk_id,
                        replacements=replacements
                    )
                    self.chunks.append(chunk)
                    all_texts.append(chunk_text)
                    chunk_id += 1

                processed_docs += 1

            except Exception as e:
                print(f"   Error processing training item: {e}")
                skipped_docs += 1
                continue

        if not all_texts:
            print("   ❌ Warning: No text chunks created. Knowledge base will be empty.")
            return

        # Create embeddings
        print(f"   Creating embeddings for {len(all_texts)} text chunks...")
        embeddings_start = time.time()

        # Batch encoding for efficiency
        batch_size = 32
        embeddings_list = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch, show_progress_bar=False)
            embeddings_list.extend(batch_embeddings)
            if i % (batch_size * 10) == 0 and i > 0:
                print(f"   Processed {i}/{len(all_texts)} embeddings...")

        self.embeddings = np.array(embeddings_list)
        embeddings_time = time.time() - embeddings_start

        # Build statistics
        build_time = time.time() - start_time
        self.build_stats = {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'total_build_time': build_time,
            'embeddings_time': embeddings_time,
            'processed_documents': processed_docs,
            'skipped_documents': skipped_docs,
            'total_chunks': len(self.chunks),
            'person_mappings': len(self.person_mapping),
            'avg_chunk_length': np.mean([len(chunk.text) for chunk in self.chunks]),
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'build_timestamp': datetime.now().isoformat()
        }

        # Cache the results
        if use_cache:
            try:
                cache_data = {
                    'chunks': self.chunks,
                    'embeddings': self.embeddings,
                    'person_mapping': self.person_mapping,
                    'build_stats': self.build_stats,
                    'created_at': datetime.now().isoformat()
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"   ✅ Cached embeddings to {self.cache_file}")
            except Exception as e:
                print(f"   Failed to cache embeddings: {e}")

        self.is_built = True
        print(f"   ✅ Knowledge base built with {len(self.chunks)} chunks in {build_time:.2f}s")

    def semantic_search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Tuple[
        List[Tuple[DocumentChunk, float]], float]:
        """Perform semantic search and return results with timing."""
        start_time = time.time()

        if not self.is_built or not self.chunks or self.embeddings is None:
            return [], 0.0

        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices[:top_k * 2]:
            similarity = similarities[idx]
            if similarity >= SIMILARITY_THRESHOLD:
                results.append((self.chunks[idx], similarity))

        search_time = time.time() - start_time
        return results[:top_k], search_time

    def hybrid_retrieval(self, query: str, target_persons: List[str],
                         top_k: int = TOP_K_RETRIEVAL) -> Tuple[List[Tuple[DocumentChunk, float, Dict]], float]:
        """Enhanced hybrid retrieval with detailed scoring and timing."""
        semantic_results, search_time = self.semantic_search(query, top_k * 2)

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

            # Boost if chunk mentions target persons (original names)
            for target_person in target_persons:
                for original_person in chunk.original_persons:
                    if target_person.lower() in original_person.lower():
                        person_boost = 0.2
                        boost += person_boost
                        boost_details['person_match_boost'] += person_boost
                        break

                # Also check reverse mapping
                for anon_id, orig_name in self.person_mapping.items():
                    if (target_person.lower() in orig_name.lower() and
                            anon_id in chunk.text):
                        anon_boost = 0.15
                        boost += anon_boost
                        boost_details['anonymized_ref_boost'] += anon_boost
                        break

            # Length boost
            length_boost = min(len(chunk.text) / 2000, 0.1)
            boost += length_boost
            boost_details['length_boost'] = length_boost

            final_score = min(similarity + boost, 1.0)
            boosted_results.append((chunk, final_score, boost_details))

        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k], search_time

    def get_context_for_prompt(self, query: str, target_persons: List[str]) -> Tuple[str, List[Dict], float]:
        """Generate formatted context string with retrieval details and timing."""
        retrieved_chunks, retrieval_time = self.hybrid_retrieval(query, target_persons)

        if not retrieved_chunks:
            return "No relevant contextual information found in knowledge base.", [], retrieval_time

        context_parts = []
        retrieval_details = []

        for i, (chunk, relevance_score, boost_details) in enumerate(retrieved_chunks, 1):
            original_persons_str = ", ".join(chunk.original_persons) if chunk.original_persons else "Unknown"
            anonymized_persons_str = ", ".join(chunk.anonymized_persons) if chunk.anonymized_persons else "None"

            context_parts.append(f"""
Context {i} (Relevance Score: {relevance_score:.3f}):
  - Original persons mentioned: {original_persons_str}
  - Anonymized references: {anonymized_persons_str}
  - Source: {chunk.title}
  - Text: {chunk.text}
""")

            retrieval_details.append({
                'context_rank': i,
                'relevance_score': relevance_score,
                'original_persons': chunk.original_persons,
                'anonymized_persons': chunk.anonymized_persons,
                'source_title': chunk.title,
                'source_url': chunk.source_url,
                'text_length': len(chunk.text),
                'boost_details': boost_details
            })

        return "\n".join(context_parts), retrieval_details, retrieval_time


class EnhancedMetricsCalculator:
    """Calculate comprehensive metrics for multi-class classification."""

    def __init__(self, results: List[ClassificationResult], target_persons: List[str]):
        self.results = results
        self.target_persons = sorted(target_persons)
        self.all_classes = sorted(set(target_persons + ['Uncertain']))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.all_classes)

    def prepare_labels(self):
        """Prepare true and predicted labels for metric calculation."""
        y_true = []
        y_pred = []

        for result in self.results:
            # Use primary ground truth person as true label
            true_label = result.true_label if result.true_label in self.all_classes else 'Uncertain'
            pred_label = result.predicted_label if result.predicted_label in self.all_classes else 'Uncertain'

            y_true.append(true_label)
            y_pred.append(pred_label)

        return y_true, y_pred

    def calculate_confusion_matrix(self):
        """Calculate confusion matrix."""
        y_true, y_pred = self.prepare_labels()
        cm = confusion_matrix(y_true, y_pred, labels=self.all_classes)
        return cm, self.all_classes

    def calculate_metrics(self):
        """Calculate comprehensive metrics."""
        y_true, y_pred = self.prepare_labels()

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.all_classes, average=None, zero_division=0
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.all_classes, average='weighted', zero_division=0
        )

        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.all_classes, average='macro', zero_division=0
        )

        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confidence analysis
        confidence_scores = self._analyze_confidence()

        # Timing analysis
        timing_stats = self._analyze_timing()

        # Create comprehensive metrics dictionary
        metrics = {
            'overall': {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'matthews_correlation_coefficient': mcc,
                'cohen_kappa': kappa,
                'total_predictions': len(y_true),
                'correct_predictions': sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp),
                'uncertain_predictions': sum(1 for yp in y_pred if yp == 'Uncertain')
            },
            'per_class': {
                cls: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1_score': f1[i],
                    'support': int(support[i]),
                    'true_positives': int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)),
                    'false_positives': int(sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)),
                    'false_negatives': int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls))
                }
                for i, cls in enumerate(self.all_classes)
            },
            'confidence_analysis': confidence_scores,
            'timing_analysis': timing_stats,
            'class_distribution': {
                'true_distribution': dict(Counter(y_true)),
                'predicted_distribution': dict(Counter(y_pred))
            }
        }

        return metrics

    def _analyze_confidence(self):
        """Analyze confidence scores and their correlation with accuracy."""
        confidence_mapping = {'High': 3, 'Medium': 2, 'Low': 1, 'N/A': 0}

        confidence_correct = defaultdict(list)
        for result in self.results:
            is_correct = result.true_label == result.predicted_label
            conf_score = confidence_mapping.get(result.confidence, 0)
            confidence_correct[result.confidence].append(is_correct)

        confidence_accuracy = {}
        for conf, correct_list in confidence_correct.items():
            if correct_list:
                confidence_accuracy[conf] = {
                    'accuracy': sum(correct_list) / len(correct_list),
                    'count': len(correct_list)
                }

        # Calculate correlation
        conf_scores = [confidence_mapping.get(r.confidence, 0) for r in self.results]
        accuracies = [1 if r.true_label == r.predicted_label else 0 for r in self.results]

        if len(set(conf_scores)) > 1:
            correlation = np.corrcoef(conf_scores, accuracies)[0, 1]
        else:
            correlation = 0.0

        return {
            'accuracy_by_confidence': confidence_accuracy,
            'confidence_accuracy_correlation': correlation
        }

    def _analyze_timing(self):
        """Analyze processing times."""
        processing_times = [r.processing_time for r in self.results]
        retrieval_times = [r.retrieval_time for r in self.results]
        api_times = [r.api_time for r in self.results]

        return {
            'processing_time': {
                'mean': np.mean(processing_times),
                'std': np.std(processing_times),
                'min': np.min(processing_times),
                'max': np.max(processing_times),
                'median': np.median(processing_times)
            },
            'retrieval_time': {
                'mean': np.mean(retrieval_times),
                'std': np.std(retrieval_times),
                'min': np.min(retrieval_times),
                'max': np.max(retrieval_times),
                'median': np.median(retrieval_times)
            },
            'api_time': {
                'mean': np.mean(api_times),
                'std': np.std(api_times),
                'min': np.min(api_times),
                'max': np.max(api_times),
                'median': np.median(api_times)
            }
        }

    def export_classification_results_csv(self, output_path: str):
        """Export classification results to CSV."""
        rows = []

        for result in self.results:
            row = {
                'item_index': result.item_index,
                'model_id': result.model_id,
                'title': result.title,
                'url': result.url,
                'true_label': result.true_label,
                'all_ground_truth': '|'.join(result.all_ground_truth),
                'predicted_label': result.predicted_label,
                'is_correct': result.true_label == result.predicted_label,
                'confidence': result.confidence,
                'avg_retrieval_score': np.mean(result.retrieval_scores) if result.retrieval_scores else 0,
                'max_retrieval_score': max(result.retrieval_scores) if result.retrieval_scores else 0,
                'processing_time': result.processing_time,
                'retrieval_time': result.retrieval_time,
                'api_time': result.api_time,
                'text_length': result.anonymized_text_length,
                'timestamp': result.timestamp
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        return df


class VisualizationGenerator:
    """Generate comprehensive visualizations for model performance."""

    def __init__(self, results_dir: str, viz_dir: str):
        self.results_dir = results_dir
        self.viz_dir = viz_dir
        os.makedirs(viz_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def create_confusion_matrix_heatmap(self, cm: np.ndarray, labels: List[str],
                                        model_id: str, save_path: str):
        """Create and save confusion matrix heatmap."""
        plt.figure(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # Create annotation text
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100
                if count > 0:
                    text = f'{count}\n({percentage:.1f}%)'
                else:
                    text = '0'
                row.append(text)
            annotations.append(row)

        # Create heatmap
        sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Normalized Frequency'})

        plt.title(f'Confusion Matrix - {model_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_interactive_confusion_matrix(self, cm: np.ndarray, labels: List[str],
                                            model_id: str, save_path: str):
        """Create interactive confusion matrix using plotly."""
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # Create hover text
        hover_text = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                text = f'True: {labels[i]}<br>Predicted: {labels[j]}<br>Count: {cm[i, j]}<br>Percentage: {cm_normalized[i, j] * 100:.1f}%'
                row.append(text)
            hover_text.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            text=cm,
            texttemplate='%{text}',
            hovertext=hover_text,
            hoverinfo='text',
            colorscale='Blues'
        ))

        fig.update_layout(
            title=f'Interactive Confusion Matrix - {model_id}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=800,
            height=700
        )

        fig.write_html(save_path)

    def create_per_class_metrics_chart(self, metrics: Dict, model_id: str, save_path: str):
        """Create bar chart showing per-class metrics."""
        per_class = metrics['per_class']
        classes = list(per_class.keys())

        precision_values = [per_class[cls]['precision'] for cls in classes]
        recall_values = [per_class[cls]['recall'] for cls in classes]
        f1_values = [per_class[cls]['f1_score'] for cls in classes]
        support_values = [per_class[cls]['support'] for cls in classes]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Metrics comparison
        x = np.arange(len(classes))
        width = 0.25

        bars1 = ax1.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
        bars2 = ax1.bar(x, recall_values, width, label='Recall', alpha=0.8)
        bars3 = ax1.bar(x + width, f1_values, width, label='F1-Score', alpha=0.8)

        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Per-Class Metrics - {model_id}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{height:.2f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom',
                                 fontsize=8)

        # Support distribution
        ax2.bar(x, support_values, alpha=0.8, color='skyblue')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Support (Count)')
        ax2.set_title('Class Distribution in Test Set', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add count labels
        for i, v in enumerate(support_values):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_model_comparison_chart(self, all_metrics: Dict[str, Dict], save_path: str):
        """Create comparison chart across all models."""
        models = list(all_metrics.keys())

        # Extract key metrics
        metrics_to_compare = [
            'accuracy', 'balanced_accuracy', 'f1_weighted',
            'precision_weighted', 'recall_weighted', 'cohen_kappa'
        ]

        # Create subplot for each metric
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics_to_compare,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for idx, metric in enumerate(metrics_to_compare):
            row = idx // 3 + 1
            col = idx % 3 + 1

            values = [all_metrics[model]['overall'][metric] for model in models]

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    name=metric
                ),
                row=row, col=col
            )

            fig.update_xaxes(tickangle=45, row=row, col=col)
            fig.update_yaxes(range=[0, 1.1], row=row, col=col)

        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=800,
            width=1200
        )

        fig.write_html(save_path)

    def create_confidence_analysis_chart(self, all_metrics: Dict[str, Dict], save_path: str):
        """Create chart showing confidence vs accuracy relationship."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (model_id, metrics) in enumerate(all_metrics.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            conf_analysis = metrics['confidence_analysis']['accuracy_by_confidence']

            if conf_analysis:
                confidences = list(conf_analysis.keys())
                accuracies = [conf_analysis[c]['accuracy'] for c in confidences]
                counts = [conf_analysis[c]['count'] for c in confidences]

                # Create bar chart
                bars = ax.bar(confidences, accuracies, alpha=0.7)

                # Add count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'n={count}', ha='center', va='bottom')

                ax.set_ylim(0, 1.1)
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Confidence Level')
                ax.set_title(f'{model_id}', fontsize=12)
                ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(all_metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Confidence Level vs Accuracy Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_timing_analysis_chart(self, all_metrics: Dict[str, Dict], save_path: str):
        """Create chart comparing processing times across models."""
        models = list(all_metrics.keys())

        # Extract timing data
        retrieval_times = []
        api_times = []
        total_times = []

        for model in models:
            timing = all_metrics[model]['timing_analysis']
            retrieval_times.append(timing['retrieval_time']['mean'])
            api_times.append(timing['api_time']['mean'])
            total_times.append(timing['processing_time']['mean'])

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(models))
        width = 0.6

        p1 = ax.bar(x, retrieval_times, width, label='Retrieval Time', alpha=0.8)
        p2 = ax.bar(x, api_times, width, bottom=retrieval_times, label='API Time', alpha=0.8)

        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Model')
        ax.set_title('Average Processing Time Breakdown by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add total time labels
        for i, (total, bar) in enumerate(zip(total_times, p2)):
            ax.text(bar.get_x() + bar.get_width() / 2., total + 0.02,
                    f'{total:.2f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# Helper functions
def load_keywords(filepath: str) -> List[str]:
    """Load target keywords from your file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]
        print(f"✅ Successfully loaded {len(keywords)} target person keywords from {filepath}")
        return keywords
    except FileNotFoundError:
        print(f"❌ Keywords file {filepath} not found!")
        print("Please create keywords.txt with your target person names (one per line)")
        return []


def load_json_data(filepath: str) -> List[Dict]:
    """Load your JSON data files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ Data file {filepath} not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {filepath}: {e}")
        return []


async def call_gemini_api(prompt_text: str, api_key: str) -> Tuple[str, Dict]:
    """Enhanced API call with detailed timing and response info."""
    if not api_key:
        return "Error: No API key provided.", {'error': 'no_api_key'}

    start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 2000,
        }
    }

    url = f"{GEMINI_API_BASE_URL}?key={api_key}"
    api_stats = {
        'prompt_length': len(prompt_text),
        'api_call_time': 0,
        'status_code': 0,
        'response_length': 0,
        'error': None
    }

    try:
        timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                api_stats['status_code'] = response.status
                api_stats['api_call_time'] = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    if (result.get("candidates") and
                            result["candidates"][0].get("content") and
                            result["candidates"][0]["content"].get("parts") and
                            result["candidates"][0]["content"]["parts"][0].get("text")):

                        response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        api_stats['response_length'] = len(response_text)
                        return response_text, api_stats
                    else:
                        api_stats['error'] = 'unexpected_response_structure'
                        return "Error: Unexpected API response structure.", api_stats
                else:
                    error_text = await response.text()
                    api_stats['error'] = f'api_error_{response.status}'
                    print(f"⚠️ API Error {response.status}: {error_text[:200]}...")
                    return f"Error: API call failed with status {response.status}", api_stats

    except asyncio.TimeoutError:
        api_stats['api_call_time'] = time.time() - start_time
        api_stats['error'] = 'timeout'
        return "Error: API call timed out", api_stats
    except Exception as e:
        api_stats['api_call_time'] = time.time() - start_time
        api_stats['error'] = str(e)
        return f"Error: {str(e)}", api_stats


def parse_llm_response(response_text: str) -> Tuple[str, str, str, Dict]:
    """Enhanced response parsing with parsing confidence."""
    identified_person = "Uncertain"
    explanation = "No explanation provided."
    confidence = "N/A"

    parsing_info = {
        'raw_response_length': len(response_text),
        'parsing_successful': False,
        'found_sections': []
    }

    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()

            if any(marker in line.lower() for marker in
                   ["identified person", "person's name", "1.", "atsakymas:", "asmuo:"]):
                if ":" in line:
                    identified_person = line.split(":", 1)[1].strip()
                    identified_person = identified_person.replace("[", "").replace("]", "")
                    parsing_info['found_sections'].append('person')

            elif any(marker in line.lower() for marker in ["explanation", "2.", "paaiškinimas:", "pagrindimas:"]):
                if ":" in line:
                    explanation = line.split(":", 1)[1].strip()
                    parsing_info['found_sections'].append('explanation')

            elif any(marker in line.lower() for marker in ["confidence", "3.", "pasitikėjimas:", "tikimybė:"]):
                if ":" in line:
                    confidence = line.split(":", 1)[1].strip()
                    parsing_info['found_sections'].append('confidence')

        if response_text.strip().lower().startswith(("uncertain", "neaišku", "nežinau")):
            identified_person = "Uncertain"
            explanation = response_text.strip()
            parsing_info['found_sections'].append('direct_uncertain')

        parsing_info['parsing_successful'] = len(parsing_info['found_sections']) >= 2

    except Exception as e:
        identified_person = "Parsing Error"
        explanation = response_text
        confidence = "Error"
        parsing_info['error'] = str(e)

    return identified_person, explanation, confidence, parsing_info


async def process_test_item_enhanced(test_item: Dict, target_persons: List[str],
                                     rag_kb: MultiEmbeddingRAGKnowledgeBase,
                                     api_key: str, item_index: int) -> Optional[ClassificationResult]:
    """Enhanced test item processing with proper result formatting."""
    start_time = time.time()

    try:
        anonymized_data = test_item.get('anonymized', {})
        anonymized_text = anonymized_data.get('full_text', '')
        title = anonymized_data.get('title', 'N/A')
        ground_truth = anonymized_data.get('keywords', [])
        url = anonymized_data.get('url', 'N/A')

        if not anonymized_text or not ground_truth:
            return None

        # Get RAG context with timing
        retrieval_start = time.time()
        context_str, retrieval_details, _ = rag_kb.get_context_for_prompt(anonymized_text, target_persons)
        retrieval_time = time.time() - retrieval_start

        # Extract retrieval scores
        retrieval_scores = [detail['relevance_score'] for detail in retrieval_details]

        # Create prompt
        prompt = f"""You are an expert AI system specialized in identifying anonymized Lithuanian political figures from news articles.

TASK: Analyze the anonymized Lithuanian text below and determine which political figure is most likely being discussed.

ANONYMIZED TEXT TO ANALYZE:
"{anonymized_text}"

RETRIEVED CONTEXTUAL INFORMATION (ranked by semantic relevance):
{context_str}

TARGET INDIVIDUALS: {', '.join(target_persons)}

ANALYSIS GUIDELINES:
- The text uses placeholders like GROUPED_PERSON_X for people and GROUPED_OCCUPATION_Y for positions
- Compare patterns in roles, activities, topics, and political contexts
- Use both the anonymized text content and retrieved contextual information
- Pay attention to relevance scores - higher scores indicate more similar contexts
- Consider Lithuanian political context and typical roles of these figures

Please provide your analysis in this exact format:
1. Identified person's name: [Name from target list, or "Uncertain"]
2. Explanation: [Detailed reasoning comparing the anonymized text with retrieved contexts]
3. Confidence score: [High, Medium, or Low]"""

        # Call API
        api_start = time.time()
        llm_response, api_stats = await call_gemini_api(prompt, api_key)
        api_time = time.time() - api_start

        # Parse response
        identified_person, explanation, confidence, _ = parse_llm_response(llm_response)

        # Determine primary ground truth label (first person in the list)
        true_label = ground_truth[0] if ground_truth else 'Unknown'

        # Create classification result
        result = ClassificationResult(
            item_index=item_index,
            model_id=rag_kb.model_id,
            true_label=true_label,
            predicted_label=identified_person,
            confidence=confidence,
            explanation=explanation,
            retrieval_scores=retrieval_scores,
            processing_time=time.time() - start_time,
            retrieval_time=retrieval_time,
            api_time=api_time,
            all_ground_truth=ground_truth,
            title=title,
            url=url,
            anonymized_text_length=len(anonymized_text),
            timestamp=datetime.now().isoformat()
        )

        return result

    except Exception as e:
        print(f"Error processing item {item_index}: {e}")
        return None


async def main():
    """Enhanced main execution with proper metrics and visualizations."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Embedding RAG Lithuanian Person Identification System')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only 5 examples)')
    parser.add_argument('--test-count', type=int, default=5, help='Number of examples to process in test mode')
    parser.add_argument('--models', nargs='+', choices=list(EMBEDDING_MODELS.keys()),
                        default=list(EMBEDDING_MODELS.keys()),
                        help='Embedding models to compare')
    args = parser.parse_args()

    test_mode = args.test
    max_examples = args.test_count if test_mode else None
    selected_models = args.models

    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    # Setup environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    start_time = time.time()
    print("🚀 Starting Enhanced Multi-Embedding RAG Comparison System")
    print(f"📊 Models to compare: {', '.join(selected_models)}")

    if test_mode:
        print(f"🧪 Running in TEST MODE - Processing {max_examples} examples only")
    else:
        print("📊 Processing ALL test data with comprehensive comparison")

    print("=" * 70)

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n🔑 Gemini API Key Required:")
        api_key = input("Enter your Gemini API key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return

    # Load data
    print("\n📂 Loading your data files...")
    target_persons = load_keywords(KEYWORDS_FILE)
    training_data = load_json_data(TRAINING_SET_FILE)
    validation_set_data = load_json_data(VALIDATION_SET_FILE)

    if not target_persons or not training_data or not validation_set_data:
        print("❌ Missing required data files. Please check your file paths.")
        return

    # Limit validation data if in test mode
    if test_mode:
        original_count = len(validation_set_data)
        validation_set_data = validation_set_data[:max_examples]
        print(f"✅ Test mode: Using {len(validation_set_data)} out of {original_count} test examples")

    print(f"✅ Target persons: {', '.join(target_persons)}")
    print(f"✅ Training examples: {len(training_data)}")
    print(f"✅ Validation examples: {len(validation_set_data)}")

    # Check for existing cache files
    print("\n🔍 Checking for existing embedding cache files...")
    for model_id in selected_models:
        cache_file = f"{EMBEDDINGS_CACHE_FILE_PREFIX}_{model_id}.pkl"
        if os.path.exists(cache_file):
            print(f"   ✅ Found cache: {cache_file}")
        else:
            print(f"   ⚠️  No cache found for {model_id}, will build from scratch")

    # Initialize storage
    all_model_results = {}
    all_model_metrics = {}
    model_knowledge_bases = {}

    # Build knowledge bases for each model
    print("\n🏗️ Building Knowledge Bases for Each Model...")
    print("=" * 70)

    for model_id in selected_models:
        model_config = EMBEDDING_MODELS[model_id].copy()
        model_config['id'] = model_id

        rag_kb = MultiEmbeddingRAGKnowledgeBase(model_config)
        rag_kb.build_knowledge_base(training_data, use_cache=True)

        if not rag_kb.is_built:
            print(f"❌ Failed to build knowledge base for {model_id}")
            continue

        model_knowledge_bases[model_id] = rag_kb
        all_model_results[model_id] = []

    if not model_knowledge_bases:
        print("❌ No knowledge bases built successfully. Exiting.")
        return

    # Process test examples
    print(f"\n🔍 Processing {len(validation_set_data)} Test Examples with {len(model_knowledge_bases)} Models...")
    print("=" * 70)

    for i, test_item in enumerate(validation_set_data):
        print(f"\n--- Processing Item {i + 1}/{len(validation_set_data)} ---")

        for model_id, rag_kb in model_knowledge_bases.items():
            print(f"  🤖 Validating with {model_id}...")

            result = await process_test_item_enhanced(
                test_item, target_persons, rag_kb, api_key, i
            )

            if result:
                all_model_results[model_id].append(result)

    # Calculate metrics and generate outputs for each model
    print("\n📊 Calculating Metrics and Generating Outputs...")
    viz_generator = VisualizationGenerator(RESULTS_DIR, VISUALIZATIONS_DIR)

    for model_id, results in all_model_results.items():
        if not results:
            continue

        print(f"\n📈 Processing results for {model_id}...")

        # Calculate metrics
        metrics_calculator = EnhancedMetricsCalculator(results, target_persons)
        metrics = metrics_calculator.calculate_metrics()
        all_model_metrics[model_id] = metrics

        # Export classification results to CSV
        csv_path = os.path.join(RESULTS_DIR, f'{model_id}_classification_results.csv')
        metrics_calculator.export_classification_results_csv(csv_path)
        print(f"   ✅ Saved classification results to {csv_path}")

        # Generate confusion matrix
        cm, labels = metrics_calculator.calculate_confusion_matrix()

        # Static confusion matrix
        cm_path = os.path.join(VISUALIZATIONS_DIR, f'{model_id}_confusion_matrix.png')
        viz_generator.create_confusion_matrix_heatmap(cm, labels, model_id, cm_path)
        print(f"   ✅ Saved confusion matrix to {cm_path}")

        # Interactive confusion matrix
        cm_interactive_path = os.path.join(VISUALIZATIONS_DIR, f'{model_id}_confusion_matrix_interactive.html')
        viz_generator.create_interactive_confusion_matrix(cm, labels, model_id, cm_interactive_path)

        # Per-class metrics chart
        metrics_chart_path = os.path.join(VISUALIZATIONS_DIR, f'{model_id}_per_class_metrics.png')
        viz_generator.create_per_class_metrics_chart(metrics, model_id, metrics_chart_path)

        # Save detailed metrics report
        report_path = os.path.join(RESULTS_DIR, f'{model_id}_performance_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Saved performance report to {report_path}")

        # Create metrics summary text file
        summary_path = os.path.join(RESULTS_DIR, f'{model_id}_metrics_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Performance Metrics Summary for {model_id}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy: {metrics['overall']['accuracy']:.3f}\n")
            f.write(f"  Balanced Accuracy: {metrics['overall']['balanced_accuracy']:.3f}\n")
            f.write(f"  F1 Score (weighted): {metrics['overall']['f1_weighted']:.3f}\n")
            f.write(f"  Precision (weighted): {metrics['overall']['precision_weighted']:.3f}\n")
            f.write(f"  Recall (weighted): {metrics['overall']['recall_weighted']:.3f}\n")
            f.write(
                f"  Matthews Correlation Coefficient: {metrics['overall']['matthews_correlation_coefficient']:.3f}\n")
            f.write(f"  Cohen's Kappa: {metrics['overall']['cohen_kappa']:.3f}\n")
            f.write(f"  Total Predictions: {metrics['overall']['total_predictions']}\n")
            f.write(f"  Correct Predictions: {metrics['overall']['correct_predictions']}\n")
            f.write(f"  Uncertain Predictions: {metrics['overall']['uncertain_predictions']}\n")

            f.write("\n\nPer-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")

            for cls, cls_metrics in metrics['per_class'].items():
                f.write(f"{cls:<20} {cls_metrics['precision']:<10.3f} {cls_metrics['recall']:<10.3f} ")
                f.write(f"{cls_metrics['f1_score']:<10.3f} {cls_metrics['support']:<10}\n")

            f.write("\n\nTiming Analysis:\n")
            timing = metrics['timing_analysis']
            f.write(
                f"  Average Processing Time: {timing['processing_time']['mean']:.2f}s ± {timing['processing_time']['std']:.2f}s\n")
            f.write(
                f"  Average Retrieval Time: {timing['retrieval_time']['mean']:.3f}s ± {timing['retrieval_time']['std']:.3f}s\n")
            f.write(f"  Average API Time: {timing['api_time']['mean']:.2f}s ± {timing['api_time']['std']:.2f}s\n")

        print(f"   ✅ Saved metrics summary to {summary_path}")

    # Generate comparison visualizations
    if len(all_model_metrics) > 1:
        print("\n📊 Generating Model Comparison Visualizations...")

        # Model comparison chart
        comparison_path = os.path.join(VISUALIZATIONS_DIR, 'model_comparison.html')
        viz_generator.create_model_comparison_chart(all_model_metrics, comparison_path)
        print(f"   ✅ Saved model comparison to {comparison_path}")

        # Confidence analysis
        confidence_path = os.path.join(VISUALIZATIONS_DIR, 'confidence_analysis.png')
        viz_generator.create_confidence_analysis_chart(all_model_metrics, confidence_path)
        print(f"   ✅ Saved confidence analysis to {confidence_path}")

        # Timing analysis
        timing_path = os.path.join(VISUALIZATIONS_DIR, 'timing_analysis.png')
        viz_generator.create_timing_analysis_chart(all_model_metrics, timing_path)
        print(f"   ✅ Saved timing analysis to {timing_path}")

    # Create summary comparison CSV
    comparison_rows = []
    for model_id, metrics in all_model_metrics.items():
        row = {
            'model_id': model_id,
            'accuracy': metrics['overall']['accuracy'],
            'balanced_accuracy': metrics['overall']['balanced_accuracy'],
            'f1_weighted': metrics['overall']['f1_weighted'],
            'precision_weighted': metrics['overall']['precision_weighted'],
            'recall_weighted': metrics['overall']['recall_weighted'],
            'mcc': metrics['overall']['matthews_correlation_coefficient'],
            'cohen_kappa': metrics['overall']['cohen_kappa'],
            'avg_processing_time': metrics['timing_analysis']['processing_time']['mean'],
            'total_predictions': metrics['overall']['total_predictions'],
            'correct_predictions': metrics['overall']['correct_predictions']
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_summary.csv'), index=False)

    # Print final summary
    print("\n🏆 FINAL RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 Model Performance Ranking (by F1 Score):")
    ranked_models = sorted(all_model_metrics.items(),
                           key=lambda x: x[1]['overall']['f1_weighted'],
                           reverse=True)

    for rank, (model_id, metrics) in enumerate(ranked_models, 1):
        print(f"{rank}. {model_id}:")
        print(f"   - F1 Score: {metrics['overall']['f1_weighted']:.3f}")
        print(f"   - Accuracy: {metrics['overall']['accuracy']:.3f}")
        print(f"   - Balanced Accuracy: {metrics['overall']['balanced_accuracy']:.3f}")

    print(f"\n⏱️ Total Execution Time: {time.time() - start_time:.2f} seconds")
    print(f"\n💾 All results saved to:")
    print(f"   • Results: {RESULTS_DIR}/")
    print(f"   • Visualizations: {VISUALIZATIONS_DIR}/")

    print("\n✅ Analysis complete! Check the output directories for detailed results.")


if __name__ == '__main__':
    # Install dependencies if needed
    required_packages = [
        'sentence-transformers', 'scikit-learn', 'aiohttp', 'numpy',
        'pandas', 'matplotlib', 'seaborn', 'plotly'
    ]

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            try:
                import nest_asyncio

                nest_asyncio.apply()
                asyncio.run(main())
            except ImportError:
                print("Installing nest-asyncio for Jupyter compatibility...")
                os.system("pip install nest-asyncio")
                import nest_asyncio

                nest_asyncio.apply()
                asyncio.run(main())
        else:
            raise