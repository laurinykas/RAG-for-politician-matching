import json
import re
import os
import time
import asyncio
import hashlib
import argparse
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from tqdm.asyncio import tqdm
import aiohttp
from typing import List, Dict, Any, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("first_step_anon/gemini_anonymization_single_key.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 2000  # Characters per chunk
MAX_CONCURRENT_REQUESTS = 25  # Max concurrent calls to Gemini API
BATCH_SIZE = 50  # Number of articles to process before saving
INPUT_FILE = "combined_lrt_articles.json"
OUTPUT_FILE = "first_step_anon/anonymized_output_single_key.json"
REQUEST_TIMEOUT = 30  # Seconds for API request
MAX_RETRIES = 3
CACHE_ENABLED = True
WATCHDOG_TIMEOUT = 600  # 10 minutes
PROGRESS_CHECK_INTERVAL = 60  # Check progress every minute

# Precompiled regex patterns for faster matching
NAME_PATTERN = re.compile(r'[A-ZĖĄĮŲŪČŠŽŻ][a-zėąįųūčšžę]+(?:\s+[A-ZĖĄĮŲŪČŠŽŻ][a-zėąįųūčšžę]+){0,2}')
INITIAL_PATTERN = re.compile(r'[A-ZĖĄĮŲŪČŠŽŻ]\.\s+[A-ZĖĄĮŲŪČŠŽŻ][a-zėąįųūčšžę]+')
JSON_PATTERN = re.compile(r'\[\s*{.*}\s*\]', re.DOTALL)


class SingleApiKeyRateLimiter:
    """Manages a single API key with rate limiting for Gemini API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.lock = asyncio.Lock()
        self.request_timestamps = deque()
        self.RPM_LIMIT = 4000  # Gemini 2.0 Flash-Lite RPM limit
        # RPD is not specified for Flash-Lite, TPM is 4,000,000 (not actively limited here yet)

        self.daily_request_count = 0
        self.total_tokens_session = 0  # Tokens used in the current session/day
        self.last_reset_day = datetime.now().day

        logger.info(f"Initialized SingleApiKeyRateLimiter. RPM Limit: {self.RPM_LIMIT}.")

    async def wait_to_make_request(self) -> str:
        """
        Waits if necessary to comply with RPM limit, then returns the API key.
        Also handles daily counter reset.
        """
        async with self.lock:
            # Reset daily counters if the day has changed
            current_time_obj = datetime.now()
            if current_time_obj.day != self.last_reset_day:
                logger.info(
                    f"New day detected. Resetting daily request count (was {self.daily_request_count}) and token count (was {self.total_tokens_session}).")
                self.daily_request_count = 0
                self.total_tokens_session = 0
                self.last_reset_day = current_time_obj.day

            while True:
                current_timestamp = time.time()
                # Remove timestamps older than 60 seconds from the deque
                while self.request_timestamps and self.request_timestamps[0] < current_timestamp - 60:
                    self.request_timestamps.popleft()

                if len(self.request_timestamps) < self.RPM_LIMIT:
                    self.request_timestamps.append(current_timestamp)
                    self.daily_request_count += 1
                    # logger.debug(f"Request slot acquired. RPM: {len(self.request_timestamps)}/60s. Daily: {self.daily_request_count}")
                    return self.api_key
                else:
                    # Calculate time to wait for the oldest request to fall out of the 60s window
                    wait_time = (self.request_timestamps[0] + 60.0) - current_timestamp + 0.001  # small epsilon
                    if wait_time <= 0:  # Should not happen if logic is correct, but as a safeguard
                        wait_time = 0.001
                    logger.warning(f"RPM limit ({self.RPM_LIMIT}) reached. "
                                   f"Currently {len(self.request_timestamps)} requests in the last minute. "
                                   f"Waiting for {wait_time:.3f} seconds.")
                    # Temporarily release lock while sleeping to allow other tasks to proceed if they can
                    # (e.g., other parts of the application not waiting for this specific lock)
                    # However, since we are in a while True loop under the same lock, this specific task will re-acquire
                    # the lock after sleeping.
                    await asyncio.sleep(wait_time)
                    # Loop will continue, re-evaluate conditions with the lock re-acquired.

    def record_token_usage(self, tokens: int):
        """Records token usage for the API key."""
        # This method might be called from multiple coroutines,
        # but += on int is generally atomic in Python's CPython implementation for single operations.
        # For more complex updates or if strict atomicity across different interpreters is needed, a lock might be used.
        self.total_tokens_session += tokens

    def get_usage_stats(self) -> Dict:
        """Gets current API key usage statistics."""
        # To get a consistent view of request_timestamps length, ideally lock, but for logging it's often acceptable.
        return {
            "api_key_masked": self.api_key[:4] + "..." + self.api_key[-4:],
            "requests_in_last_minute_approx": len(self.request_timestamps),
            "daily_requests_today": self.daily_request_count,
            "total_tokens_session": self.total_tokens_session,
            "rpm_limit_target": self.RPM_LIMIT
        }


class WatchdogTimer:
    """Class for monitoring progress and detecting stuck processing"""

    def __init__(self, timeout=WATCHDOG_TIMEOUT):
        self.timeout = timeout
        self.last_progress_time = time.time()
        self.running = False
        self.progress_data = {"items_processed": 0, "last_item_id": None}
        self.task = None

    async def start(self):
        """Start the watchdog timer"""
        self.running = True
        self.last_progress_time = time.time()
        self.task = asyncio.create_task(self._monitor())
        logger.info(f"Watchdog timer started with timeout of {self.timeout} seconds")

    async def _monitor(self):
        """Monitor progress and detect if processing is stuck"""
        while self.running:
            await asyncio.sleep(PROGRESS_CHECK_INTERVAL)
            current_time = time.time()
            elapsed = current_time - self.last_progress_time

            if elapsed > self.timeout:
                logger.error(f"WATCHDOG ALERT: No progress detected for {elapsed:.1f} seconds!")
                logger.error(f"Last progress: {self.progress_data}")
            else:
                logger.debug(f"Watchdog: Last progress was {elapsed:.1f} seconds ago")

    def update_progress(self, item_id, items_processed=None):
        """Update progress data"""
        self.last_progress_time = time.time()
        self.progress_data["last_item_id"] = item_id
        if items_processed is not None:
            self.progress_data["items_processed"] = items_processed

    async def stop(self):
        """Stop the watchdog timer"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Watchdog timer stopped")


class EntityCache:
    """Cache for storing and retrieving extracted entities"""

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, text_hash):
        if text_hash in self.cache:
            self.hits += 1
            return self.cache[text_hash]
        self.misses += 1
        return None

    def set(self, text_hash, entities):
        self.cache[text_hash] = entities

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.2f}%"
        }


entity_cache = EntityCache()


def parse_json_from_text(text):
    """Extract JSON from text more efficiently"""
    start = text.find('[')
    end = text.rfind(']')

    if start != -1 and end != -1 and start < end:
        json_str = text[start:end + 1].replace("'", '"')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_match = JSON_PATTERN.search(text)
            if json_match:
                json_str_regex = json_match.group(0).replace("'", '"')
                try:
                    return json.loads(json_str_regex)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON with direct find and regex: {json_str_regex[:100]}...")
                    pass
    logger.debug(f"No valid JSON array found in text: {text[:100]}...")
    return []


async def extract_entities_with_gemini(text: str, rate_limiter: SingleApiKeyRateLimiter):
    """Extract entities using Gemini API with rate limiting and caching."""
    text_hash = hashlib.md5(text.encode()).hexdigest()

    if CACHE_ENABLED:
        cached_entities = entity_cache.get(text_hash)
        if cached_entities is not None:
            logger.debug(f"Cache hit for text hash: {text_hash}")
            return cached_entities

    potential_entities = set()
    for match in NAME_PATTERN.finditer(text):
        potential_entities.add(match.group(0))
    for match in INITIAL_PATTERN.finditer(text):
        potential_entities.add(match.group(0))

    if not potential_entities:
        logger.debug("No potential entities found by regex, skipping API call.")
        if CACHE_ENABLED:
            entity_cache.set(text_hash, [])
        return []

    # Add a small random delay for jitter before acquiring rate limit slot
    # This helps in making traffic pattern slightly less predictable.
    # The main rate limiting is handled by wait_to_make_request().
    jitter_delay = random.uniform(0.05, 0.2)  # Small jitter
    await asyncio.sleep(jitter_delay)

    api_key = await rate_limiter.wait_to_make_request()  # Handles RPM limiting

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    prompt_payload = {
        "contents": [{
            "parts": [{"text": f"""
                Extract all PERSON NAMES, JOBS, OCCUPATIONS and POSITIONS from the following Lithuanian text.
                Find ALL variations of names (full names, last names, initials with last names).
                DO NOT extract locations or organizations. Return ONLY a JSON list with the following format:
                [
                    {{"text": "extracted text", "type": "PERSON"}},
                    {{"text": "extracted text", "type": "OCCUPATION"}}
                ]

                TEXT:
                {text}
                """
                       }]
        }],
        "generationConfig": {
            "temperature": round(random.uniform(0.1, 0.3), 2),  # Slight variation
            "topP": round(random.uniform(0.7, 0.9), 2),  # Slight variation
            "topK": random.randint(35, 45),  # Slight variation
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json"
        }
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    for attempt in range(MAX_RETRIES):
        try:
            request_start_time = time.time()
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES} to call Gemini API.")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=prompt_payload, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                    response_duration = time.time() - request_start_time
                    if response.status == 200:
                        result = await response.json()
                        if "usageMetadata" in result:
                            token_usage = result["usageMetadata"].get("totalTokens", 0)
                            rate_limiter.record_token_usage(token_usage)

                        if result.get("candidates") and result["candidates"][0].get("content", {}).get("parts"):
                            response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                            entities = parse_json_from_text(response_text)
                            if CACHE_ENABLED:
                                entity_cache.set(text_hash, entities)
                            logger.debug(
                                f"Gemini API call successful. Duration: {response_duration:.2f}s. Entities found: {len(entities)}")
                            return entities
                        else:
                            logger.error(
                                f"Gemini API response malformed: Missing candidates or parts. Status: {response.status}. Response: {str(result)[:200]}")
                            # Treat as error, will retry or return empty

                    else:  # Error status codes
                        error_text = await response.text()
                        logger.error(
                            f"Error from Gemini API (Attempt {attempt + 1}): Status {response.status}, Text: {error_text[:200]}. Duration: {response_duration:.2f}s")
                        if response.status == 429:  # Rate limit explicitly hit
                            # The rate limiter should prevent this, but if it happens, wait longer.
                            # The wait in wait_to_make_request should handle this primarily.
                            # This is an additional safety.
                            wait_time = random.uniform(5, 10) * (attempt + 1)
                            logger.warning(
                                f"Explicit 429 from API. Waiting {wait_time:.2f}s before retry {attempt + 2}.")
                            await asyncio.sleep(wait_time)
                        elif response.status >= 500:  # Server-side errors
                            await asyncio.sleep(
                                random.uniform(2, 5) * (attempt + 1))  # Exponential backoff for server errors
                        else:  # Other client-side errors (4xx)
                            await asyncio.sleep(random.uniform(1, 3) * (attempt + 1))
                            if response.status in [400, 401,
                                                   403]:  # Bad request, unauth - unlikely to be fixed by retry
                                logger.error(
                                    f"Critical client error {response.status}. Aborting retries for this chunk.")
                                return []


        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error (Attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(random.uniform(2, 5) * (attempt + 1))
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to Gemini API (Attempt {attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(random.uniform(3, 6) * (attempt + 1))
        except Exception as e:
            logger.error(f"Unexpected error during Gemini API call (Attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}",
                         exc_info=True)
            await asyncio.sleep(random.uniform(2, 5) * (attempt + 1))

    logger.error(f"All {MAX_RETRIES} attempts to connect to Gemini API failed for text hash {text_hash}.")
    return []


class NameMapper:
    """Efficient data structure for name mapping"""

    def __init__(self):
        self.full_name_map = {}
        self.last_name_map = {}
        self.initial_map = {}
        self.first_name_map = {}

    def add(self, name, replacement):
        self.full_name_map[name] = replacement
        parts = name.split()
        if len(parts) > 1:
            last_name = parts[-1]
            first_name = parts[0]
            if last_name not in self.last_name_map: self.last_name_map[last_name] = name
            if first_name not in self.first_name_map: self.first_name_map[first_name] = name
            if '.' in first_name:
                initial = first_name.rstrip('.')
                self.initial_map[f"{initial}. {last_name}"] = name

    def find(self, name):
        if name in self.full_name_map: return self.full_name_map[name]
        clean_name = name.replace('.', '').strip()
        if clean_name in self.full_name_map: return self.full_name_map[clean_name]
        parts = name.split()
        if len(parts) > 0:
            last_name = parts[-1]
            if last_name in self.last_name_map: return self.full_name_map[self.last_name_map[last_name]]
            if len(parts) > 1 and '.' in parts[0]:
                initial = parts[0].rstrip('.')
                key = f"{initial}. {last_name}"
                if key in self.initial_map: return self.full_name_map[self.initial_map[key]]
        if len(parts) == 1 and parts[0] in self.first_name_map:
            return self.full_name_map[self.first_name_map[parts[0]]]
        return None

    def get_all_mappings(self):
        return self.full_name_map.copy()


class Anonymizer:
    """Main anonymization class"""

    def __init__(self, rate_limiter: SingleApiKeyRateLimiter):
        self.rate_limiter = rate_limiter
        self.name_mapper = NameMapper()
        self.occupation_map = {}
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def anonymize_chunk(self, chunk_id: int, chunk: str):
        async with self.semaphore:
            logger.debug(f"Processing chunk {chunk_id} ({len(chunk)} chars) with semaphore.")
            if not chunk.strip():
                return chunk, []
            try:
                entities = await extract_entities_with_gemini(chunk, self.rate_limiter)
                if not entities:
                    return chunk, []

                entities.sort(key=lambda x: len(x['text']), reverse=True)
                replacements_made = []
                anonymized_chunk = chunk
                processed_entities = set()

                for entity in entities:
                    entity_text = entity['text']
                    entity_type = entity['type']
                    if entity_text in processed_entities: continue
                    processed_entities.add(entity_text)

                    replacement = None
                    if entity_type == "PERSON":
                        replacement = self.name_mapper.find(entity_text)
                        if not replacement:
                            replacement = f"PERSON_{len(self.name_mapper.get_all_mappings()) + 1}"
                            self.name_mapper.add(entity_text, replacement)
                            logger.debug(f"New person map: '{entity_text}' -> {replacement}")
                    elif entity_type in ("OCCUPATION", "POSITION", "JOB"):
                        replacement = self.occupation_map.get(entity_text)
                        if not replacement:
                            replacement = f"OCCUPATION_{len(self.occupation_map) + 1}"
                            self.occupation_map[entity_text] = replacement
                            logger.debug(f"New occupation map: '{entity_text}' -> {replacement}")

                    if replacement:
                        replacements_made.append(
                            {"original": entity_text, "replacement": replacement, "type": entity_type})
                        # Use regex for robust replacement of whole words/phrases
                        # Ensure to escape special characters in entity_text for regex
                        pattern = r'\b' + re.escape(entity_text) + r'\b'
                        anonymized_chunk = re.sub(pattern, replacement, anonymized_chunk)

                return anonymized_chunk, replacements_made
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}", exc_info=True)
                return chunk, []  # Return original chunk on error

    async def anonymize_text(self, text: str):
        if not text or len(text.strip()) == 0:
            return text, []

        # Using global CHUNK_SIZE
        if len(text) < CHUNK_SIZE:
            return await self.anonymize_chunk(0, text)

        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para_with_sep = para + "\n\n"
            if len(current_chunk) + len(para_with_sep) <= CHUNK_SIZE or not current_chunk:
                current_chunk += para_with_sep
            else:
                chunks.append(current_chunk.rstrip("\n\n"))  # Store without trailing newlines from split
                current_chunk = para_with_sep

        if current_chunk.strip():  # Add any remaining part
            chunks.append(current_chunk.rstrip("\n\n"))

        if not chunks:  # If text was only newlines or very short
            if current_chunk.strip():
                chunks.append(current_chunk.rstrip("\n\n"))
            elif text.strip():
                chunks = [text]  # Fallback for edge cases
            else:
                return text, []

        tasks = [self.anonymize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        anonymized_parts = []
        all_replacements = []
        for anon_chunk, replacements in results:
            anonymized_parts.append(anon_chunk)
            all_replacements.extend(replacements)

        # Reconstruct text carefully, original split was '\n\n'
        anonymized_text = "\n\n".join(anonymized_parts)

        unique_replacements = []
        seen = set()
        for r in all_replacements:
            key = (r["original"], r["replacement"], r["type"])
            if key not in seen:
                seen.add(key)
                unique_replacements.append(r)
        return anonymized_text, unique_replacements

    def get_name_map(self):
        return self.name_mapper.get_all_mappings()

    def get_occupation_map(self):
        return self.occupation_map.copy()


async def process_item(anonymizer: Anonymizer, item: Dict, item_index: int, batch_index: int):
    overall_index = batch_index * BATCH_SIZE + item_index
    logger.debug(f"Processing item {overall_index}")
    start_time = time.time()
    output_entry = {"anonymized": {}, "replacements": []}

    for key, value in item.items():
        if key not in ["title", "full_text"]:
            output_entry["anonymized"][key] = value

    if "title" in item and item["title"]:
        anonymized_title, title_replacements = await anonymizer.anonymize_text(item["title"])
        output_entry["anonymized"]["title"] = anonymized_title
        output_entry["replacements"].extend(title_replacements)

    if "full_text" in item and item["full_text"]:
        anonymized_text, text_replacements = await anonymizer.anonymize_text(item["full_text"])
        output_entry["anonymized"]["full_text"] = anonymized_text
        output_entry["replacements"].extend(text_replacements)

    unique_replacements = []
    seen = set()
    for r in output_entry["replacements"]:
        key = (r["original"], r["replacement"], r["type"])
        if key not in seen:
            seen.add(key)
            unique_replacements.append(r)
    output_entry["replacements"] = unique_replacements

    logger.debug(
        f"Item {overall_index} processed in {time.time() - start_time:.2f} seconds. Replacements: {len(unique_replacements)}")
    return output_entry


async def process_data_batch(anonymizer: Anonymizer, data_batch: List[Dict], output_data: List[Dict], batch_index: int,
                             watchdog: WatchdogTimer = None):
    tasks = [process_item(anonymizer, item, i, batch_index) for i, item in enumerate(data_batch)]
    batch_results = await asyncio.gather(*tasks)

    for i, output_entry in enumerate(batch_results):
        output_data.append(output_entry)
        if watchdog:
            watchdog.update_progress(f"item_{batch_index * BATCH_SIZE + i}", len(output_data))
        if (i + 1) % 5 == 0 or i == len(batch_results) - 1:
            logger.info(
                f"Batch {batch_index + 1} progress: {i + 1}/{len(batch_results)} items processed. Total items: {len(output_data)}")
    return output_data


def save_output(output_file: str, data: List[Dict]):
    temp_file = f"{output_file}.temp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, output_file)
        logger.info(f"Data successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving output to {output_file}: {e}", exc_info=True)


def save_mapping_dictionaries(name_map: Dict, occupation_map: Dict, output_prefix: str):
    name_map_file = f"{output_prefix}_name_map.json"
    occupation_map_file = f"{output_prefix}_occupation_map.json"
    try:
        with open(f"{name_map_file}.temp", 'w', encoding='utf-8') as f:
            json.dump(name_map, f, ensure_ascii=False, indent=2)
        os.replace(f"{name_map_file}.temp", name_map_file)

        with open(f"{occupation_map_file}.temp", 'w', encoding='utf-8') as f:
            json.dump(occupation_map, f, ensure_ascii=False, indent=2)
        os.replace(f"{occupation_map_file}.temp", occupation_map_file)
        logger.info(f"Mapping dictionaries saved to {name_map_file} and {occupation_map_file}")
    except Exception as e:
        logger.error(f"Error saving mapping dictionaries: {e}", exc_info=True)


def load_mapping_dictionaries(output_prefix: str):
    name_map, occupation_map = {}, {}
    name_map_file = f"{output_prefix}_name_map.json"
    occupation_map_file = f"{output_prefix}_occupation_map.json"

    if os.path.exists(name_map_file):
        try:
            with open(name_map_file, 'r', encoding='utf-8') as f:
                name_map = json.load(f)
            logger.info(f"Loaded {len(name_map)} existing name mappings from {name_map_file}")
        except Exception as e:
            logger.error(f"Error loading name map {name_map_file}: {e}")

    if os.path.exists(occupation_map_file):
        try:
            with open(occupation_map_file, 'r', encoding='utf-8') as f:
                occupation_map = json.load(f)
            logger.info(f"Loaded {len(occupation_map)} existing occupation mappings from {occupation_map_file}")
        except Exception as e:
            logger.error(f"Error loading occupation map {occupation_map_file}: {e}")

    return name_map, occupation_map


async def anonymize_json_file(rate_limiter: SingleApiKeyRateLimiter, input_file: str, output_file: str,
                              continue_from_previous: bool):
    watchdog = WatchdogTimer()
    await watchdog.start()

    logger.info(f"Starting Gemini API anonymization process with single API key.")
    logger.info(f"Input file: {input_file}, Output file: {output_file}")

    process_start_time = time.time()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_list = [data] if isinstance(data, dict) else data
        logger.info(f"Loaded {len(data_list)} items to process.")
        watchdog.update_progress("data_loaded")
    except Exception as e:
        logger.error(f"Failed to load or parse input file {input_file}: {e}", exc_info=True)
        await watchdog.stop()
        return False

    output_prefix = os.path.splitext(output_file)[0]
    anonymizer = Anonymizer(rate_limiter)

    if continue_from_previous:
        name_map, occupation_map = load_mapping_dictionaries(output_prefix)
        for name, replacement in name_map.items(): anonymizer.name_mapper.add(name, replacement)
        anonymizer.occupation_map = occupation_map
        logger.info(
            f"Loaded mappings: Names {len(anonymizer.name_mapper.get_all_mappings())}, Occupations {len(anonymizer.occupation_map)}")

    output_data = []
    processed_items_count = 0
    if continue_from_previous and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            processed_items_count = len(output_data)
            logger.info(f"Loaded {processed_items_count} existing entries from {output_file}.")
            watchdog.update_progress("output_loaded", processed_items_count)
            if processed_items_count >= len(data_list):
                logger.info("All items already processed.")
                await watchdog.stop()
                return True
            data_list = data_list[processed_items_count:]  # Slice to process remaining
            logger.info(f"Resuming. {len(data_list)} items remaining.")
        except Exception as e:
            logger.error(f"Error loading existing output file {output_file}: {e}. Starting fresh for output.",
                         exc_info=True)
            output_data = []  # Reset if error
            processed_items_count = 0

    total_remaining_articles = len(data_list)
    if total_remaining_articles == 0:
        logger.info("No new articles to process.")
        await watchdog.stop()
        return True

    total_batches = (total_remaining_articles + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Estimated {total_batches} batches for {total_remaining_articles} remaining articles.")

    with tqdm(total=total_remaining_articles, desc="Articles Anonymized", unit="article") as progress_bar:
        for batch_idx in range(total_batches):
            batch_start_idx = batch_idx * BATCH_SIZE
            batch_end_idx = min(batch_start_idx + BATCH_SIZE, total_remaining_articles)
            current_batch_data = data_list[batch_start_idx:batch_end_idx]

            logger.info(f"Processing Batch {batch_idx + 1}/{total_batches} ({len(current_batch_data)} items). "
                        f"Overall progress: {processed_items_count + batch_start_idx}/{processed_items_count + total_remaining_articles}")
            logger.info(f"Current API Usage: {rate_limiter.get_usage_stats()}")
            watchdog.update_progress(f"batch_{batch_idx}_start", processed_items_count + batch_start_idx)

            batch_process_start_time = time.time()
            try:
                # output_data is appended to in process_data_batch
                await process_data_batch(anonymizer, current_batch_data, output_data, batch_idx, watchdog)
            except Exception as e:
                logger.error(f"Critical error processing Batch {batch_idx + 1}: {e}", exc_info=True)
                # Decide if to continue or stop. For now, log and attempt to save then continue.

            batch_duration = time.time() - batch_process_start_time
            progress_bar.update(len(current_batch_data))
            watchdog.update_progress(f"batch_{batch_idx}_complete", len(output_data))

            try:
                save_output(output_file, output_data)
                save_mapping_dictionaries(anonymizer.get_name_map(), anonymizer.get_occupation_map(), output_prefix)
            except Exception as e:
                logger.error(f"Error saving data after Batch {batch_idx + 1}: {e}", exc_info=True)

            logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.2f}s. "
                        f"Avg time/article: {batch_duration / len(current_batch_data) if current_batch_data else 0:.2f}s.")

            if batch_idx < total_batches - 1:  # Avoid sleep after last batch
                await asyncio.sleep(1)  # Small pause between batches

    total_duration = time.time() - process_start_time
    logger.info(f"Anonymization complete. Total time: {timedelta(seconds=total_duration)}")
    logger.info(f"Final Cache Stats: {entity_cache.stats()}")
    logger.info(f"Final API Usage: {rate_limiter.get_usage_stats()}")
    await watchdog.stop()
    return True


async def main():
    global INPUT_FILE, OUTPUT_FILE, BATCH_SIZE, MAX_CONCURRENT_REQUESTS, CHUNK_SIZE, CACHE_ENABLED, WATCHDOG_TIMEOUT

    parser = argparse.ArgumentParser(description='Gemini API Anonymization Tool (Single API Key)')
    parser.add_argument('--input', default=INPUT_FILE, help='Input JSON file path')
    parser.add_argument('--output', default=OUTPUT_FILE, help='Output JSON file path')
    parser.add_argument('--api-key', type=str, default=None, help='Gemini API Key string.')
    parser.add_argument('--api-key-file', type=str, default='api_key.txt', help='File containing the Gemini API Key.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Number of articles to process before saving')
    parser.add_argument('--concurrent', type=int, default=MAX_CONCURRENT_REQUESTS,
                        help='Maximum concurrent API requests')
    parser.add_argument('--fresh-start', action='store_true', help='Start fresh, ignoring existing output and mappings')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Size of text chunks for API processing')
    parser.add_argument('--no-cache', action='store_true', help='Disable API response caching')
    parser.add_argument('--watchdog-timeout', type=int, default=WATCHDOG_TIMEOUT, help='Watchdog timeout in seconds')

    args = parser.parse_args()

    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    BATCH_SIZE = args.batch_size
    MAX_CONCURRENT_REQUESTS = args.concurrent
    CHUNK_SIZE = args.chunk_size
    CACHE_ENABLED = not args.no_cache
    WATCHDOG_TIMEOUT = args.watchdog_timeout

    api_key_value = None
    if args.api_key:
        api_key_value = args.api_key.strip()
        logger.info("Using API key from --api-key argument.")
    else:
        try:
            with open(args.api_key_file, 'r') as f:
                api_key_value = f.readline().strip()
            if api_key_value:
                logger.info(f"Loaded API key from {args.api_key_file}.")
            else:
                logger.error(f"API key file {args.api_key_file} is empty.")
        except FileNotFoundError:
            logger.error(
                f"API key file not found: {args.api_key_file}. Please provide a key via --api-key or --api-key-file.")
        except Exception as e:
            logger.error(f"Error reading API key file {args.api_key_file}: {e}")

    if not api_key_value or len(api_key_value) < 10:  # Basic validation
        print("\nERROR: A valid Gemini API key is required. Provide it via --api-key or --api-key-file.")
        logger.critical("No valid API key provided. Exiting.")
        return

    print_startup_banner(api_key_value, args)

    # Estimate workload
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data_est = json.load(f)
        article_count_est = len([data_est] if isinstance(data_est, dict) else data_est)
        avg_chars_est = CHUNK_SIZE * 1.5  # Rough guess
        total_api_calls_est = article_count_est * (avg_chars_est / CHUNK_SIZE)
        print(f"Estimated workload for {article_count_est} articles: ~{total_api_calls_est:.0f} API calls.")
        rpm_limit_gemini = 4000
        estimated_minutes = total_api_calls_est / rpm_limit_gemini
        print(
            f"At {rpm_limit_gemini} RPM, estimated processing time (API calls only): {estimated_minutes / 60:.2f} hours.")
    except Exception as e:
        logger.warning(f"Could not estimate workload: {e}")
    print("-" * 60)

    confirm = input("Continue with processing? (y/n): ")
    if confirm.lower() not in ['y', 'yes']:
        print("Operation cancelled by user.")
        return

    try:
        print("\nStarting anonymization process...")
        if not args.fresh_start and os.path.exists(OUTPUT_FILE):
            print(f"Attempting to resume from existing output file: {OUTPUT_FILE}")
            print(f"To start fresh (deleting existing output and maps), use the --fresh-start flag.")

        rate_limiter_instance = SingleApiKeyRateLimiter(api_key_value)

        await anonymize_json_file(
            rate_limiter=rate_limiter_instance,
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            continue_from_previous=not args.fresh_start
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress (if any was made)...")
        # Saving is handled within anonymize_json_file batches, so this is mostly a notification.
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        print(f"\nAn critical error occurred: {e}. Check logs for details.")
    finally:
        print("Anonymization process finished or was terminated.")


def print_startup_banner(api_key: str, args: argparse.Namespace):
    print("\n" + "=" * 60)
    print("     GEMINI API ANONYMIZATION TOOL (Single Key Mode)")
    print("=" * 60)
    print(f"API Key: {api_key[:4]}...{api_key[-4:]}")
    print(f"Input File: {args.input}")
    print(f"Output File: {args.output}")
    print(f"Max Concurrent Requests: {args.concurrent}")
    print(f"Text Chunk Size: {args.chunk_size} chars")
    print(f"Batch Size (articles): {args.batch_size}")
    print(f"Response Cache Enabled: {not args.no_cache}")
    print(f"Watchdog Timeout: {args.watchdog_timeout}s")
    print(f"Fresh Start: {args.fresh_start}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess terminated by user (main level).")
    except Exception as e:
        print(f"Critical error at script execution level: {str(e)}")
        import traceback

        traceback.print_exc()

