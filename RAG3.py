import json
import asyncio
import aiohttp
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
from collections import defaultdict
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

warnings.filterwarnings('ignore')

# Import required packages
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Installing required packages...")
    pip_command = [sys.executable, "-m", "pip", "install", "sentence-transformers", "scikit-learn", "pandas",
                   "matplotlib", "seaborn", "plotly", "tqdm", "torch", "psutil", "aiohttp"]
    import subprocess

    subprocess.check_call(pip_command)
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

# Configuration
KEYWORDS_FILE = 'keywords.txt'
TRAINING_SET_FILE = 'training_set.json'
VALIDATION_SET_FILE = 'validation_set.json'
RESULTS_DIR = 'RAG3/prompt_optimization_results'
VISUALIZATIONS_DIR = 'RAG3/prompt_optimization_visualizations'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Optimal parameters from previous optimization
OPTIMAL_PARAMS = {
    'chunk_size': 350,
    'chunk_overlap': 50,
    'top_k_retrieval': 6,
    'similarity_threshold': 0.25
}

# Gemini API Configuration
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# LaBSE Model Configuration
LABSE_MODEL = {
    'id': 'labse',
    'name': 'sentence-transformers/LaBSE',
    'dimensions': 768
}

# Global model instance
GLOBAL_MODEL = None
DEVICE = None


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata"""
    id: str
    name: str
    description: str
    template: str
    category: str  # e.g., "structured", "cot", "multilingual", "evidence-focused"
    expected_format: Dict[str, str] = field(default_factory=dict)


# Define different prompt strategies
PROMPT_TEMPLATES = [
    PromptTemplate(
        id="baseline",
        name="Baseline Structured",
        description="Original structured prompt with clear guidelines",
        category="structured",
        template="""You are an expert AI system specialized in identifying anonymized Lithuanian political figures from news articles.

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
3. Confidence score: [High, Medium, or Low]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="cot",
        name="Chain of Thought",
        description="Step-by-step reasoning approach",
        category="cot",
        template="""You are an expert AI system specialized in identifying anonymized Lithuanian political figures.

ANONYMIZED TEXT:
"{text}"

CONTEXTUAL INFORMATION FROM KNOWLEDGE BASE:
{context}

POTENTIAL CANDIDATES: {targets}

Think through this step-by-step:

Step 1: Identify key roles, positions, and activities mentioned in the anonymized text.
Step 2: Match these patterns with the contextual information provided.
Step 3: Consider which candidate from the list best matches these patterns.
Step 4: Evaluate the strength of the evidence.

Based on your step-by-step analysis:
1. Identified person's name: [Name from list or "Uncertain"]
2. Explanation: [Your step-by-step reasoning]
3. Confidence score: [High/Medium/Low]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="evidence_first",
        name="Evidence-First Analysis",
        description="Emphasizes evidence evaluation before identification",
        category="evidence-focused",
        template="""As an expert in Lithuanian politics, analyze this anonymized news article to identify the political figure being discussed.

ANONYMIZED ARTICLE:
"{text}"

REFERENCE CONTEXTS (similarity scores indicate relevance):
{context}

CANDIDATE LIST: {targets}

INSTRUCTIONS:
1. First, list all evidence from the text (roles, actions, contexts)
2. Then match evidence to the reference contexts
3. Finally, determine which candidate best fits the evidence

Your analysis must follow this format:
1. Identified person's name: [Select from candidates or state "Uncertain"]
2. Explanation: [Evidence found → Context matches → Final reasoning]
3. Confidence score: [High if strong evidence, Medium if partial, Low if weak]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="concise",
        name="Concise Direct",
        description="Minimal prompt focusing on key information",
        category="minimal",
        template="""Identify the anonymized Lithuanian political figure in this text.

TEXT: "{text}"

CONTEXT: {context}

CANDIDATES: {targets}

Analyze the text patterns and context to identify the person.

Format:
1. Identified person's name: [Name or "Uncertain"]
2. Explanation: [Brief reasoning]
3. Confidence score: [High/Medium/Low]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="bilingual",
        name="Bilingual Lithuanian-English",
        description="Instructions in both Lithuanian and English",
        category="multilingual",
        template="""You are an expert AI system specialized in identifying anonymized Lithuanian political figures.
Jūs esate ekspertinė AI sistema, specializuojanti anoniminių Lietuvos politinių veikėjų identifikavime.

TASK/UŽDUOTIS: Analyze the anonymized text below / Išanalizuokite žemiau pateiktą anoniminį tekstą

ANONYMIZED TEXT / ANONIMINIS TEKSTAS:
"{text}"

CONTEXTUAL INFORMATION / KONTEKSTINĖ INFORMACIJA:
{context}

CANDIDATES / KANDIDATAI: {targets}

Please identify the political figure using evidence from the text and context.
Prašome identifikuoti politinį veikėją naudojant teksto ir konteksto įrodymus.

Provide your answer in English following this format / Pateikite atsakymą anglų kalba pagal šį formatą:
1. Identified person's name: [Name/Vardas or "Uncertain"/"Neaišku"]
2. Explanation: [Reasoning/Pagrindimas]
3. Confidence score: [High/Medium/Low - Aukštas/Vidutinis/Žemas]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="comparative",
        name="Comparative Analysis",
        description="Explicitly compares each candidate",
        category="comparative",
        template="""Expert task: Identify which Lithuanian political figure is discussed in this anonymized article.

ANONYMIZED ARTICLE:
"{text}"

KNOWLEDGE BASE CONTEXTS:
{context}

Your task is to compare how well each candidate matches the text:
CANDIDATES TO COMPARE: {targets}

For each candidate, evaluate:
- How well their known roles match the anonymized positions
- How well their typical topics match the article content
- How well the contexts support this identification

After comparing all candidates:
1. Identified person's name: [Most likely candidate or "Uncertain"]
2. Explanation: [Comparative analysis showing why this person fits best]
3. Confidence score: [High/Medium/Low based on evidence strength]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="role_focused",
        name="Role and Position Focused",
        description="Emphasizes matching roles and positions",
        category="role-based",
        template="""You are analyzing an anonymized Lithuanian political news article where person names are replaced with GROUPED_PERSON_X and roles with GROUPED_OCCUPATION_Y.

ANONYMIZED TEXT:
"{text}"

SUPPORTING CONTEXTS FROM DATABASE:
{context}

POSSIBLE IDENTITIES: {targets}

Focus on matching:
1. Political positions and roles mentioned (even if anonymized)
2. Policy areas and topics discussed
3. Relationships and interactions described

Based on role and position analysis:
1. Identified person's name: [Choose from list or "Uncertain"]
2. Explanation: [How roles and positions led to this identification]
3. Confidence score: [High/Medium/Low]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    ),

    PromptTemplate(
        id="negative_space",
        name="Elimination Method",
        description="Identifies by eliminating unlikely candidates",
        category="elimination",
        template="""Identify the anonymized Lithuanian political figure by process of elimination.

ANONYMIZED TEXT TO ANALYZE:
"{text}"

CONTEXTUAL CLUES:
{context}

COMPLETE CANDIDATE LIST: {targets}

Approach:
1. Note characteristics that DON'T match each candidate
2. Eliminate candidates who clearly don't fit
3. From remaining candidates, select the best match

Important: The person IS one of the listed candidates. Choose "Uncertain" only if multiple candidates remain equally likely after elimination.

Your conclusion:
1. Identified person's name: [Final choice or "Uncertain"]
2. Explanation: [Who was eliminated and why, then why the chosen person fits]
3. Confidence score: [High if others clearly eliminated, Medium if some ambiguity, Low if many remain possible]""",
        expected_format={
            "person_line": "1. identified person's name:",
            "explanation_line": "2. explanation:",
            "confidence_line": "3. confidence score:"
        }
    )
]


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
class PromptTestResult:
    item_index: int
    prompt_id: str
    prompt_name: str
    true_label: str
    predicted_label: str
    confidence: str
    explanation: str
    retrieval_scores: List[float]
    processing_time: float
    api_time: float
    prompt_length: int
    response_length: int
    timestamp: str


def initialize_global_model():
    """Initialize the global model with GPU support if available."""
    global GLOBAL_MODEL, DEVICE

    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        DEVICE = 'cpu'
        print("⚠️ CUDA not available, using CPU")

    if GLOBAL_MODEL is None:
        print(f"Loading LaBSE model on {DEVICE}...")
        GLOBAL_MODEL = SentenceTransformer(LABSE_MODEL['name'], device=DEVICE)
        GLOBAL_MODEL.eval()
        if DEVICE == 'cuda':
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


class OptimizedRAGKnowledgeBase:
    """Simplified KB class using optimal parameters"""

    def __init__(self):
        self.chunk_size = OPTIMAL_PARAMS['chunk_size']
        self.chunk_overlap = OPTIMAL_PARAMS['chunk_overlap']
        self.top_k_retrieval = OPTIMAL_PARAMS['top_k_retrieval']
        self.similarity_threshold = OPTIMAL_PARAMS['similarity_threshold']

        self.encoder = GLOBAL_MODEL
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False
        self.person_mapping = {}

        self.cache_file = os.path.join(RESULTS_DIR, "optimal_kb_cache.pkl")

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
            # Fallback to character-based chunking
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                start = end - self.chunk_overlap if end < len(text) else end
            return chunks

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

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

        print("Building knowledge base with optimal parameters...")
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

            text_chunks = self.chunk_text(full_text)

            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    original_persons=keywords,
                    anonymized_persons=[p for p in item_person_map.keys() if p in chunk_text],
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
            batch_size = 64 if DEVICE == 'cuda' else 32
            embeddings_list = []

            for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
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

    def semantic_search(self, query: str) -> List[Tuple[DocumentChunk, float]]:
        if not self.is_built:
            return []

        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [query], show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True
            )

        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-self.top_k_retrieval * 2:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                results.append((self.chunks[idx], float(similarities[idx])))
            if len(results) >= self.top_k_retrieval:
                break

        return results

    def get_context_for_prompt(self, query: str) -> Tuple[str, List[float]]:
        results = self.semantic_search(query)
        if not results:
            return "No relevant contextual information found.", []

        context_parts = []
        scores = []

        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(f"""
Context {i} (Relevance Score: {score:.3f}):
  - Source: {chunk.title}
  - Text: "{chunk.text}"
""")
            scores.append(score)

        return "\n".join(context_parts), scores


async def call_gemini_api_async(session: aiohttp.ClientSession, prompt_text: str, api_key: str) -> Tuple[str, float]:
    """Call Gemini API and return response with timing"""
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

    try:
        async with session.post(url, headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=90)) as response:
            response.raise_for_status()
            result = await response.json()

            api_time = time.time() - start_time

            if (result.get("candidates") and
                    result["candidates"][0].get("content") and
                    result["candidates"][0]["content"].get("parts") and
                    result["candidates"][0]["content"]["parts"][0].get("text")):
                return result["candidates"][0]["content"]["parts"][0]["text"], api_time
            else:
                return "Error: Unexpected API response structure", api_time

    except Exception as e:
        api_time = time.time() - start_time
        return f"Error: {str(e)}", api_time


def parse_llm_response(response_text: str, prompt_template: PromptTemplate) -> Tuple[str, str, str]:
    """Parse LLM response according to expected format"""
    identified_person = "Uncertain"
    explanation = "No explanation provided."
    confidence = "N/A"

    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            line_lower = line.lower().strip()

            # Check for person identification
            if prompt_template.expected_format["person_line"].lower() in line_lower:
                identified_person = line.split(":", 1)[1].strip().replace("[", "").replace("]", "")

            # Check for explanation
            elif prompt_template.expected_format["explanation_line"].lower() in line_lower:
                explanation = line.split(":", 1)[1].strip()

            # Check for confidence
            elif prompt_template.expected_format["confidence_line"].lower() in line_lower:
                confidence = line.split(":", 1)[1].strip()

    except Exception as e:
        print(f"Error parsing response: {e}")

    return identified_person, explanation, confidence


async def test_prompt_on_item(
        session: aiohttp.ClientSession,
        prompt_template: PromptTemplate,
        test_item: Dict,
        rag_kb: OptimizedRAGKnowledgeBase,
        target_persons: List[str],
        api_key: str,
        item_idx: int
) -> Optional[PromptTestResult]:
    """Test a single prompt template on a single item"""

    try:
        anonymized_data = test_item.get('anonymized', {})
        anonymized_text = anonymized_data.get('full_text', '')
        ground_truth = anonymized_data.get('keywords', [])[0] if anonymized_data.get('keywords') else 'Unknown'

        if not anonymized_text:
            return None

        # Get context
        context_str, retrieval_scores = rag_kb.get_context_for_prompt(anonymized_text)

        # Format prompt
        prompt = prompt_template.template.format(
            text=anonymized_text,
            context=context_str,
            targets=', '.join(target_persons)
        )

        # Call API
        start_time = time.time()
        response_text, api_time = await call_gemini_api_async(session, prompt, api_key)
        total_time = time.time() - start_time

        # Parse response
        identified_person, explanation, confidence = parse_llm_response(response_text, prompt_template)

        return PromptTestResult(
            item_index=item_idx,
            prompt_id=prompt_template.id,
            prompt_name=prompt_template.name,
            true_label=ground_truth,
            predicted_label=identified_person,
            confidence=confidence,
            explanation=explanation,
            retrieval_scores=retrieval_scores,
            processing_time=total_time,
            api_time=api_time,
            prompt_length=len(prompt),
            response_length=len(response_text),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Error testing prompt {prompt_template.id} on item {item_idx}: {e}")
        return None


async def test_all_prompts(
        testing_data: List[Dict],
        rag_kb: OptimizedRAGKnowledgeBase,
        target_persons: List[str],
        api_key: str,
        max_items: Optional[int] = None
) -> List[PromptTestResult]:
    """Test all prompt templates on all items"""

    results = []
    items_to_test = testing_data[:max_items] if max_items else testing_data

    async with aiohttp.ClientSession() as session:
        # Test each prompt template
        for prompt_template in PROMPT_TEMPLATES:
            print(f"\n📝 Testing prompt: {prompt_template.name} ({prompt_template.category})")
            print(f"   Description: {prompt_template.description}")

            prompt_results = []

            # Process items in batches
            batch_size = 20  # Small batches to avoid rate limiting

            for i in tqdm(range(0, len(items_to_test), batch_size),
                          desc=f"Testing {prompt_template.name}"):
                batch = items_to_test[i:i + batch_size]
                tasks = []

                for j, item in enumerate(batch):
                    task = test_prompt_on_item(
                        session, prompt_template, item, rag_kb,
                        target_persons, api_key, i + j
                    )
                    tasks.append(task)

                batch_results = await asyncio.gather(*tasks)
                prompt_results.extend([r for r in batch_results if r is not None])

                # Small delay between batches
                if i + batch_size < len(items_to_test):
                    await asyncio.sleep(0.5)

            results.extend(prompt_results)

            # Quick performance summary
            if prompt_results:
                accuracy = sum(1 for r in prompt_results if r.predicted_label == r.true_label) / len(prompt_results)
                avg_time = sum(r.api_time for r in prompt_results) / len(prompt_results)
                print(f"   Quick stats - Accuracy: {accuracy:.3f}, Avg API time: {avg_time:.2f}s")

    return results


def calculate_prompt_metrics(results: List[PromptTestResult], target_persons: List[str]) -> pd.DataFrame:
    """Calculate metrics for each prompt template"""

    metrics_by_prompt = []

    for prompt_id in set(r.prompt_id for r in results):
        prompt_results = [r for r in results if r.prompt_id == prompt_id]

        if not prompt_results:
            continue

        y_true = [r.true_label for r in prompt_results]
        y_pred = [r.predicted_label for r in prompt_results]

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Calculate confidence distribution
        confidence_dist = pd.Series([r.confidence for r in prompt_results]).value_counts().to_dict()

        # Calculate timing statistics
        avg_api_time = np.mean([r.api_time for r in prompt_results])
        avg_response_length = np.mean([r.response_length for r in prompt_results])
        avg_prompt_length = np.mean([r.prompt_length for r in prompt_results])

        # Get prompt metadata
        prompt_template = next(p for p in PROMPT_TEMPLATES if p.id == prompt_id)

        metrics_by_prompt.append({
            'prompt_id': prompt_id,
            'prompt_name': prompt_template.name,
            'category': prompt_template.category,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'avg_api_time': avg_api_time,
            'avg_response_length': avg_response_length,
            'avg_prompt_length': avg_prompt_length,
            'high_confidence_pct': confidence_dist.get('High', 0) / len(prompt_results),
            'medium_confidence_pct': confidence_dist.get('Medium', 0) / len(prompt_results),
            'low_confidence_pct': confidence_dist.get('Low', 0) / len(prompt_results),
            'total_samples': len(prompt_results)
        })

    return pd.DataFrame(metrics_by_prompt).sort_values('f1_score', ascending=False)


def create_prompt_comparison_visualizations(metrics_df: pd.DataFrame, results: List[PromptTestResult]):
    """Create comprehensive visualizations comparing prompts"""

    # 1. Main metrics comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('F1 Score by Prompt', 'Accuracy vs Response Time',
                        'Confidence Distribution', 'Category Performance'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # F1 Score comparison
    fig.add_trace(
        go.Bar(
            x=metrics_df['prompt_name'],
            y=metrics_df['f1_score'],
            text=metrics_df['f1_score'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
            name='F1 Score'
        ),
        row=1, col=1
    )

    # Accuracy vs Response Time scatter
    fig.add_trace(
        go.Scatter(
            x=metrics_df['avg_api_time'],
            y=metrics_df['accuracy'],
            mode='markers+text',
            text=metrics_df['prompt_name'],
            textposition='top center',
            marker=dict(size=10),
            name='Prompts'
        ),
        row=1, col=2
    )

    # Confidence distribution
    confidence_data = []
    for _, row in metrics_df.iterrows():
        confidence_data.extend([
            {'prompt': row['prompt_name'], 'confidence': 'High', 'percentage': row['high_confidence_pct']},
            {'prompt': row['prompt_name'], 'confidence': 'Medium', 'percentage': row['medium_confidence_pct']},
            {'prompt': row['prompt_name'], 'confidence': 'Low', 'percentage': row['low_confidence_pct']}
        ])
    confidence_df = pd.DataFrame(confidence_data)

    for conf_level in ['High', 'Medium', 'Low']:
        data = confidence_df[confidence_df['confidence'] == conf_level]
        fig.add_trace(
            go.Bar(
                x=data['prompt'],
                y=data['percentage'],
                name=conf_level,
                text=data['percentage'].apply(lambda x: f'{x:.2%}'),
                textposition='auto'
            ),
            row=2, col=1
        )

    # Category performance
    category_metrics = metrics_df.groupby('category').agg({
        'f1_score': 'mean',
        'accuracy': 'mean',
        'avg_api_time': 'mean'
    }).reset_index()

    fig.add_trace(
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['f1_score'],
            text=category_metrics['f1_score'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
            name='Avg F1 by Category'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1000,
        title_text="Prompt Template Performance Comparison",
        barmode='stack',
        showlegend=True
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="F1 Score", row=1, col=1)
    fig.update_xaxes(title_text="Avg API Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    fig.update_yaxes(title_text="Avg F1 Score", row=2, col=2)

    fig.write_html(os.path.join(VISUALIZATIONS_DIR, "prompt_comparison_comprehensive.html"))

    # 2. Confusion matrix heatmap for best prompt
    best_prompt_id = metrics_df.iloc[0]['prompt_id']
    best_prompt_results = [r for r in results if r.prompt_id == best_prompt_id]

    y_true = [r.true_label for r in best_prompt_results]
    y_pred = [r.predicted_label for r in best_prompt_results]

    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - Best Prompt: {metrics_df.iloc[0]["prompt_name"]}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "best_prompt_confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. Prompt length vs performance
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['avg_prompt_length'], metrics_df['f1_score'], s=100)
    for idx, row in metrics_df.iterrows():
        plt.annotate(row['prompt_name'],
                     (row['avg_prompt_length'], row['f1_score']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Average Prompt Length (characters)')
    plt.ylabel('F1 Score')
    plt.title('Prompt Length vs Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "prompt_length_vs_performance.png"), dpi=300)
    plt.close()


def create_detailed_report(metrics_df: pd.DataFrame, results: List[PromptTestResult]):
    """Create a detailed markdown report"""

    report = []
    report.append("# Prompt Optimization Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    best_prompt = metrics_df.iloc[0]
    report.append(f"**Best Performing Prompt**: {best_prompt['prompt_name']} ({best_prompt['category']})")
    report.append(f"- F1 Score: {best_prompt['f1_score']:.3f}")
    report.append(f"- Accuracy: {best_prompt['accuracy']:.3f}")
    report.append(f"- Average API Time: {best_prompt['avg_api_time']:.2f}s\n")

    # Optimal Parameters Used
    report.append("## Optimal RAG Parameters Used\n")
    for param, value in OPTIMAL_PARAMS.items():
        report.append(f"- **{param}**: {value}")
    report.append("")

    # Prompt Rankings
    report.append("## Prompt Template Rankings\n")
    report.append("| Rank | Prompt Name | Category | F1 Score | Accuracy | Avg Time (s) |")
    report.append("|------|-------------|----------|----------|----------|--------------|")

    for idx, row in metrics_df.iterrows():
        report.append(f"| {idx + 1} | {row['prompt_name']} | {row['category']} | "
                      f"{row['f1_score']:.3f} | {row['accuracy']:.3f} | {row['avg_api_time']:.2f} |")

    # Category Analysis
    report.append("\n## Performance by Category\n")
    category_stats = metrics_df.groupby('category').agg({
        'f1_score': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'avg_api_time': 'mean'
    }).round(3)

    report.append("| Category | Avg F1 | F1 Std | Avg Accuracy | Acc Std | Avg Time |")
    report.append("|----------|--------|--------|--------------|---------|----------|")

    for category in category_stats.index:
        report.append(f"| {category} | {category_stats.loc[category, ('f1_score', 'mean')]:.3f} | "
                      f"{category_stats.loc[category, ('f1_score', 'std')]:.3f} | "
                      f"{category_stats.loc[category, ('accuracy', 'mean')]:.3f} | "
                      f"{category_stats.loc[category, ('accuracy', 'std')]:.3f} | "
                      f"{category_stats.loc[category, ('avg_api_time', 'mean')]:.2f} |")

    # Detailed Prompt Analysis
    report.append("\n## Detailed Prompt Analysis\n")

    for _, prompt in metrics_df.head(3).iterrows():
        report.append(f"### {prompt['prompt_name']}\n")
        report.append(f"**Category**: {prompt['category']}")
        report.append(f"**Performance Metrics**:")
        report.append(f"- F1 Score: {prompt['f1_score']:.3f}")
        report.append(f"- Precision: {prompt['precision']:.3f}")
        report.append(f"- Recall: {prompt['recall']:.3f}")
        report.append(f"- MCC: {prompt['mcc']:.3f}")
        report.append(f"- Cohen's Kappa: {prompt['cohen_kappa']:.3f}")
        report.append(f"\n**Confidence Distribution**:")
        report.append(f"- High: {prompt['high_confidence_pct']:.1%}")
        report.append(f"- Medium: {prompt['medium_confidence_pct']:.1%}")
        report.append(f"- Low: {prompt['low_confidence_pct']:.1%}")
        report.append("")

    # Save report
    report_path = os.path.join(RESULTS_DIR, "prompt_optimization_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"📄 Detailed report saved to: {report_path}")


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


async def main():
    parser = argparse.ArgumentParser(description='Test different prompts with optimal RAG parameters')
    parser.add_argument('--test-count', type=int, default=None,
                        help='Number of items to test (default: all)')
    parser.add_argument('--skip-prompts', nargs='+', default=[],
                        help='Prompt IDs to skip')
    args = parser.parse_args()

    print("🚀 Starting Prompt Optimization Testing")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Using optimal parameters: {OPTIMAL_PARAMS}")
    print("=" * 70)

    # Initialize model
    initialize_global_model()
    print(f"💾 Initial memory: {get_memory_info()}")

    # Load API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("🔑 Enter your Gemini API key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return

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
    print("\n🏗️ Building knowledge base with optimal parameters...")
    rag_kb = OptimizedRAGKnowledgeBase()
    rag_kb.build_knowledge_base(training_data, use_cache=True)

    if not rag_kb.is_built:
        print("❌ Failed to build knowledge base. Exiting.")
        return

    print(f"✅ Knowledge base ready with {len(rag_kb.chunks)} chunks")
    print(f"💾 Memory after KB build: {get_memory_info()}")

    # Filter prompts if requested
    global PROMPT_TEMPLATES
    if args.skip_prompts:
        PROMPT_TEMPLATES = [p for p in PROMPT_TEMPLATES if p.id not in args.skip_prompts]
        print(f"📝 Testing {len(PROMPT_TEMPLATES)} prompts (skipped: {args.skip_prompts})")
    else:
        print(f"📝 Testing all {len(PROMPT_TEMPLATES)} prompt templates")

    # Run tests
    print("\n🧪 Starting prompt testing...")
    start_time = time.time()

    results = await test_all_prompts(
        validation_data,
        rag_kb,
        target_persons,
        api_key,
        max_items=args.test_count
    )

    total_time = time.time() - start_time
    print(f"\n✅ Testing complete in {total_time / 60:.2f} minutes")
    print(f"💾 Final memory: {get_memory_info()}")

    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics_df = calculate_prompt_metrics(results, target_persons)

    # Save results
    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df.to_csv(os.path.join(RESULTS_DIR, "prompt_test_results.csv"), index=False)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "prompt_metrics_summary.csv"), index=False)

    # Create visualizations
    print("📈 Creating visualizations...")
    create_prompt_comparison_visualizations(metrics_df, results)

    # Create report
    create_detailed_report(metrics_df, results)

    # Print summary
    print("\n" + "=" * 70)
    print("🏆 TOP 3 PERFORMING PROMPTS")
    print("=" * 70)

    for idx, row in metrics_df.head(3).iterrows():
        print(f"\nRank {idx + 1}: {row['prompt_name']} ({row['category']})")
        print(f"  F1 Score: {row['f1_score']:.3f}")
        print(f"  Accuracy: {row['accuracy']:.3f}")
        print(f"  MCC: {row['mcc']:.3f}")
        print(f"  Avg Response Time: {row['avg_api_time']:.2f}s")
        print(f"  High Confidence: {row['high_confidence_pct']:.1%}")

    print(f"\n💾 All results saved to: {RESULTS_DIR}/")
    print(f"📊 Visualizations saved to: {VISUALIZATIONS_DIR}/")
    print("\n✅ Prompt optimization testing complete!")


if __name__ == "__main__":
    asyncio.run(main())