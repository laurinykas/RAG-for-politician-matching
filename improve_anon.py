import json
import re
import os
import time
import asyncio
import hashlib
import argparse
import random
from collections import defaultdict, deque
import logging
from tqdm.asyncio import tqdm
import aiohttp
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime  # Added missing import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_anonymization_refiner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
REQUEST_TIMEOUT = 60  # Increased timeout for potentially larger LLM requests
MAX_RETRIES = 3
LLM_BATCH_SIZE = 100  # Number of terms to send to LLM in one go for grouping
MAX_CONCURRENT_LLM_REQUESTS = 5  # Max concurrent calls to Gemini API for grouping


# --- Reused SingleApiKeyRateLimiter (slightly adapted if needed) ---
class SingleApiKeyRateLimiter:
    """Manages a single API key with rate limiting for Gemini API."""

    def __init__(self, api_key: str, rpm_limit: int = 4000):  # Default for Flash-Lite
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.lock = asyncio.Lock()
        self.request_timestamps = deque()
        self.RPM_LIMIT = rpm_limit
        self.daily_request_count = 0
        self.total_tokens_session = 0
        self.last_reset_day = datetime.now().day
        logger.info(f"Initialized SingleApiKeyRateLimiter. RPM Limit: {self.RPM_LIMIT}.")

    async def wait_to_make_request(self) -> str:
        async with self.lock:
            current_time_obj = datetime.now()
            if current_time_obj.day != self.last_reset_day:
                logger.info(
                    f"New day. Resetting daily request count (was {self.daily_request_count}) and token count (was {self.total_tokens_session}).")
                self.daily_request_count = 0
                self.total_tokens_session = 0
                self.last_reset_day = current_time_obj.day
            while True:
                current_timestamp = time.time()
                while self.request_timestamps and self.request_timestamps[0] < current_timestamp - 60:
                    self.request_timestamps.popleft()
                if len(self.request_timestamps) < self.RPM_LIMIT:
                    self.request_timestamps.append(current_timestamp)
                    self.daily_request_count += 1
                    return self.api_key
                else:
                    wait_time = (self.request_timestamps[0] + 60.0) - current_timestamp + 0.001
                    wait_time = max(wait_time, 0.001)  # Ensure positive wait time
                    logger.warning(f"RPM limit ({self.RPM_LIMIT}) reached. Waiting {wait_time:.3f}s.")
                    await asyncio.sleep(wait_time)

    def record_token_usage(self, tokens: int):
        self.total_tokens_session += tokens

    def get_usage_stats(self) -> Dict:
        return {
            "api_key_masked": self.api_key[:4] + "..." + self.api_key[-4:],
            "requests_in_last_minute_approx": len(self.request_timestamps),
            "daily_requests_today": self.daily_request_count,
            "total_tokens_session": self.total_tokens_session,
            "rpm_limit_target": self.RPM_LIMIT
        }


# --- Helper Functions ---
def load_json_file(file_path: str) -> Any:
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        return None


def save_json_file(data: Any, file_path: str):
    """Saves data to a JSON file."""
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):  # Check if output_dir is not empty
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating directory {output_dir}: {e}", exc_info=True)
            # Depending on desired behavior, you might want to raise the error or handle it
            # For now, we'll let the open() call fail if directory creation fails critically

    try:
        temp_file = f"{file_path}.temp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, file_path)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}", exc_info=True)


async def call_gemini_for_grouping(terms: List[str], entity_type: str, rate_limiter: SingleApiKeyRateLimiter) -> Dict[
    str, str]:
    """
    Calls Gemini API to group similar terms.
    entity_type: "PERSON" or "OCCUPATION" to tailor the prompt.
    Returns a dictionary mapping original terms to their canonical group representative.
    """
    if not terms:
        return {}

    api_key = await rate_limiter.wait_to_make_request()
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"  # Using flash-lite

    if entity_type == "PERSON":
        prompt_instruction = """You are an expert linguist specializing in Lithuanian. Your task is to group the following list of Lithuanian person names into semantically equivalent entities.
Consider variations due to initials (e.g., 'V. Adamkus' vs 'Valdas Adamkus'), grammatical case endings if they appear, or minor spelling differences.
The goal is to map all variations of the same person's name to a single, canonical name form (preferably full name, nominative case).
Provide your output as a JSON object with a single key "grouped_terms". The value of "grouped_terms" should be another JSON object where each key is an original term from the input list, and its value is the canonical representative name for the group it belongs to.
Example: If input is ["Jonas Jonaitis", "J. Jonaitis", "Ona Onaitė"], output should be:
{"grouped_terms": {"Jonas Jonaitis": "Jonas Jonaitis", "J. Jonaitis": "Jonas Jonaitis", "Ona Onaitė": "Ona Onaitė"}}"""
    elif entity_type == "OCCUPATION":
        prompt_instruction = """You are an expert linguist specializing in Lithuanian. Your task is to group the following list of Lithuanian job titles, occupations, and positions into semantically equivalent categories.
Consider variations due to grammatical gender (e.g., 'mokytojas' vs 'mokytoja'), case endings if they appear, or minor variations.
The goal is to map all variations of the same occupation to a single, canonical form (preferably masculine singular nominative, or the most common general form).
Provide your output as a JSON object with a single key "grouped_terms". The value of "grouped_terms" should be another JSON object where each key is an original term from the input list, and its value is the canonical representative term for the group it belongs to.
Example: If input is ["mokytojas", "mokytoja", "direktorius", "Prezidentas", "Prezidentė"], output should be:
{"grouped_terms": {"mokytojas": "mokytojas", "mokytoja": "mokytojas", "direktorius": "direktorius", "Prezidentas": "prezidentas", "Prezidentė": "prezidentas"}}"""
    else:
        raise ValueError("Invalid entity_type for grouping.")

    # Create a string representation of the list for the prompt
    terms_list_str = json.dumps(terms, ensure_ascii=False)

    prompt_payload = {
        "contents": [{
            "parts": [{"text": f"{prompt_instruction}\n\nInput List:\n{terms_list_str}\n\nJSON Output:"}]
        }],
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for more deterministic grouping
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 2048,  # Increased for potentially large mapping
            "responseMimeType": "application/json"
            # Removed "responseSchema"
        }
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(
                f"Attempt {attempt + 1}/{MAX_RETRIES} to call Gemini for grouping {entity_type} (batch size: {len(terms)}).")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=prompt_payload, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "usageMetadata" in result:
                            rate_limiter.record_token_usage(result["usageMetadata"].get("totalTokens", 0))

                        if result.get("candidates") and \
                                result["candidates"][0].get("content") and \
                                result["candidates"][0]["content"].get("parts") and \
                                isinstance(result["candidates"][0]["content"]["parts"], list) and \
                                len(result["candidates"][0]["content"]["parts"]) > 0 and \
                                "text" in result["candidates"][0]["content"]["parts"][0]:

                            text_content = result["candidates"][0]["content"]["parts"][0]["text"]
                            try:
                                # The LLM should directly return the JSON string as specified in the prompt
                                parsed_content = json.loads(text_content)
                                if "grouped_terms" in parsed_content and isinstance(parsed_content["grouped_terms"],
                                                                                    dict):
                                    logger.info(
                                        f"Successfully grouped {len(parsed_content['grouped_terms'])} {entity_type} terms via LLM.")
                                    return parsed_content["grouped_terms"]
                                else:
                                    logger.error(
                                        f"LLM response for grouping {entity_type} missing 'grouped_terms' dict after parsing: {text_content[:200]}")
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to parse JSON from LLM text content for {entity_type}: {e}. Content: {text_content[:200]}")
                            except Exception as e:
                                logger.error(
                                    f"Unexpected error parsing LLM response for {entity_type}: {e}. Result: {str(result)[:200]}")
                        else:
                            logger.warning(
                                f"LLM response for grouping {entity_type} did not have the expected structure. Response: {str(result)[:200]}")
                        return {}
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Error from Gemini API for grouping {entity_type} (Attempt {attempt + 1}): Status {response.status}, Text: {error_text[:200]}")
                        if response.status == 429:
                            await asyncio.sleep(random.uniform(5, 10) * (attempt + 1))
                        else:
                            await asyncio.sleep(random.uniform(2, 5) * (attempt + 1))
        except Exception as e:
            logger.error(f"Error calling Gemini for grouping {entity_type} (Attempt {attempt + 1}): {e}", exc_info=True)
            await asyncio.sleep(random.uniform(2, 5) * (attempt + 1))
    logger.error(f"All {MAX_RETRIES} attempts failed for grouping {entity_type} batch.")
    return {}


async def consolidate_map_via_llm(original_entity_map: Dict[str, str], entity_type: str,
                                  rate_limiter: SingleApiKeyRateLimiter,
                                  consolidated_map_output_path: str, old_ids_to_new_ids_output_path: str,
                                  force_regenerate: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Consolidates an entity map (name or occupation) using LLM grouping.
    Returns:
        - original_term_to_new_grouped_id_map: {"original term": "GROUPED_ID_X"}
        - old_anon_id_to_new_grouped_id_map: {"OLD_ANON_ID_Y": "GROUPED_ID_X"}
    """
    if not force_regenerate:
        if os.path.exists(consolidated_map_output_path) and os.path.exists(old_ids_to_new_ids_output_path):
            loaded_consolidated_map = load_json_file(consolidated_map_output_path)
            loaded_old_ids_map = load_json_file(old_ids_to_new_ids_output_path)
            if loaded_consolidated_map is not None and loaded_old_ids_map is not None:
                logger.info(
                    f"Loading existing consolidated maps for {entity_type} from {consolidated_map_output_path} and {old_ids_to_new_ids_output_path}")
                return loaded_consolidated_map, loaded_old_ids_map
            else:
                logger.warning(f"Failed to load one or both existing maps for {entity_type}. Will regenerate.")

    original_terms = list(original_entity_map.keys())
    if not original_terms:
        logger.warning(f"No original terms found for {entity_type} to consolidate.")
        return {}, {}

    all_llm_grouped_terms: Dict[str, str] = {}  # Maps original term to its canonical group representative from LLM

    # Batch terms for LLM processing
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)
    tasks = []

    for i in range(0, len(original_terms), LLM_BATCH_SIZE):
        batch = original_terms[i:i + LLM_BATCH_SIZE]

        # Need to wrap call_gemini_for_grouping in another async function to use semaphore
        async def _process_batch(current_batch):
            async with semaphore:
                return await call_gemini_for_grouping(current_batch, entity_type, rate_limiter)

        tasks.append(_process_batch(batch))

    logger.info(f"Sending {len(tasks)} batches of {entity_type} terms to LLM for grouping.")

    results = await asyncio.gather(*tasks)
    for result_batch in results:
        if isinstance(result_batch, dict):  # Ensure it's a dict before updating
            all_llm_grouped_terms.update(result_batch)
        else:
            logger.warning(f"Received non-dict result from LLM grouping batch for {entity_type}: {type(result_batch)}")

    if not all_llm_grouped_terms:
        logger.error(f"LLM grouping failed to return any results for {entity_type} after all batches.")
        return {}, {}

    # --- Process LLM results to create new grouped IDs ---
    canonical_to_new_grouped_id: Dict[str, str] = {}
    original_term_to_new_grouped_id: Dict[str, str] = {}
    group_id_counter = 1

    # Ensure all original terms are covered, even if LLM missed some (map them to themselves)
    for term in original_terms:
        if term not in all_llm_grouped_terms:
            logger.warning(
                f"Term '{term}' was not returned by LLM grouping for {entity_type}. Will treat as its own group.")
            all_llm_grouped_terms[term] = term  # Treat as its own canonical form

    for original_term, canonical_rep in all_llm_grouped_terms.items():
        if not isinstance(canonical_rep, str):  # Ensure canonical_rep is a string
            logger.warning(
                f"Canonical representative for '{original_term}' is not a string: {canonical_rep}. Using original term as canonical.")
            canonical_rep = original_term

        if canonical_rep not in canonical_to_new_grouped_id:
            new_id_prefix = "GROUPED_PERSON" if entity_type == "PERSON" else "GROUPED_OCCUPATION"
            canonical_to_new_grouped_id[canonical_rep] = f"{new_id_prefix}_{group_id_counter}"
            group_id_counter += 1
        original_term_to_new_grouped_id[original_term] = canonical_to_new_grouped_id[canonical_rep]

    # --- Create mapping from old anonymized IDs to new grouped IDs ---
    old_anon_id_to_new_grouped_id: Dict[str, str] = {}
    for original_term, old_anon_id in original_entity_map.items():
        new_grouped_id = original_term_to_new_grouped_id.get(original_term)
        if new_grouped_id:
            old_anon_id_to_new_grouped_id[old_anon_id] = new_grouped_id
        else:
            logger.warning(
                f"Original term '{original_term}' (old ID: {old_anon_id}) not found in final original_term_to_new_grouped_id map for {entity_type}. "
                f"This old ID will not be updated to a new grouped ID.")

    save_json_file(original_term_to_new_grouped_id, consolidated_map_output_path)
    save_json_file(old_anon_id_to_new_grouped_id, old_ids_to_new_ids_output_path)

    logger.info(f"Consolidation for {entity_type} complete. Found {len(canonical_to_new_grouped_id)} unique groups.")
    return original_term_to_new_grouped_id, old_anon_id_to_new_grouped_id


def apply_consolidated_mappings_to_text(text: str, old_ids_to_new_ids_map: Dict[str, str]) -> str:
    """Replaces old anonymized IDs in text with new grouped IDs."""
    if not text or not old_ids_to_new_ids_map:
        return text

    # Sort by length of old ID (longest first) to prevent partial replacements (e.g. PERSON_1 before PERSON_10)
    # And use word boundaries to ensure only full IDs are replaced.
    sorted_old_ids = sorted(old_ids_to_new_ids_map.keys(), key=len, reverse=True)

    modified_text = text
    for old_id in sorted_old_ids:
        new_id = old_ids_to_new_ids_map[old_id]
        # Regex to match whole word, careful with special characters if IDs could have them
        # Assuming IDs are like PERSON_1, OCCUPATION_1 etc.
        pattern = r'\b' + re.escape(old_id) + r'\b'
        modified_text = re.sub(pattern, new_id, modified_text)
    return modified_text


async def main_refinement_process(args):
    """Main orchestration function for the refinement process."""
    logger.info("Starting anonymization refinement process...")

    api_key = None
    if args.api_key:
        api_key = args.api_key.strip()
    else:
        try:
            with open(args.api_key_file, 'r') as f:
                api_key = f.readline().strip()
        except FileNotFoundError:
            logger.error(f"API key file not found: {args.api_key_file}")
        if not api_key:
            logger.error("API key is missing.")
            return

    rate_limiter = SingleApiKeyRateLimiter(api_key)

    # Load original mappings
    original_name_map = load_json_file(args.name_map_file)
    original_occupation_map = load_json_file(args.occupation_map_file)

    if not original_name_map:  # Check if None or empty
        logger.error(f"Failed to load original name map from {args.name_map_file} or it's empty. Exiting.")
        return
    if not original_occupation_map:  # Check if None or empty
        logger.error(f"Failed to load original occupation map from {args.occupation_map_file} or it's empty. Exiting.")
        return

    # Consolidate Name Mappings
    logger.info("Consolidating NAME mappings...")
    consolidated_name_map, old_name_ids_to_new_ids = await consolidate_map_via_llm(
        original_name_map, "PERSON", rate_limiter,
        args.output_consolidated_name_map, args.output_old_name_ids_map, args.fresh_consolidation
    )
    if not consolidated_name_map or not old_name_ids_to_new_ids:  # Check if empty or None
        logger.error(
            "Name map consolidation resulted in empty maps. Check logs. Proceeding without name refinement for text.")
        old_name_ids_to_new_ids = {}  # Ensure it's a dict to prevent errors later

    # Consolidate Occupation Mappings
    logger.info("Consolidating OCCUPATION mappings...")
    consolidated_occupation_map, old_occupation_ids_to_new_ids = await consolidate_map_via_llm(
        original_occupation_map, "OCCUPATION", rate_limiter,
        args.output_consolidated_occupation_map, args.output_old_occupation_ids_map, args.fresh_consolidation
    )
    if not consolidated_occupation_map or not old_occupation_ids_to_new_ids:  # Check if empty or None
        logger.error(
            "Occupation map consolidation resulted in empty maps. Check logs. Proceeding without occupation refinement for text.")
        old_occupation_ids_to_new_ids = {}  # Ensure it's a dict

    # Combine old_id to new_id maps for text processing
    combined_old_to_new_ids_map = {**old_name_ids_to_new_ids, **old_occupation_ids_to_new_ids}

    if not combined_old_to_new_ids_map:
        logger.warning("No ID mappings were generated from LLM consolidation. Text will not be refined.")
    else:
        logger.info(f"Combined map for text replacement has {len(combined_old_to_new_ids_map)} entries.")

    # Process the main anonymized file
    logger.info(f"Loading main anonymized data from: {args.anonymized_input_file}")
    anonymized_data = load_json_file(args.anonymized_input_file)
    if not anonymized_data:
        logger.error("Failed to load main anonymized data file. Exiting.")
        return

    if not isinstance(anonymized_data, list):  # Ensure it's a list of articles
        anonymized_data = [anonymized_data]

    refined_data_output = []
    logger.info(f"Applying consolidated mappings to {len(anonymized_data)} articles...")

    for article_entry in tqdm(anonymized_data, desc="Refining Articles"):
        new_article_entry = article_entry.copy()  # Shallow copy
        new_article_entry["anonymized"] = article_entry.get("anonymized", {}).copy()  # Deeper copy for anonymized part

        # Refine text fields
        if "title" in new_article_entry["anonymized"] and new_article_entry["anonymized"]["title"]:
            new_article_entry["anonymized"]["title"] = apply_consolidated_mappings_to_text(
                new_article_entry["anonymized"]["title"], combined_old_to_new_ids_map
            )
        if "full_text" in new_article_entry["anonymized"] and new_article_entry["anonymized"]["full_text"]:
            new_article_entry["anonymized"]["full_text"] = apply_consolidated_mappings_to_text(
                new_article_entry["anonymized"]["full_text"], combined_old_to_new_ids_map
            )

        # Update the 'replacements' list
        original_replacements = article_entry.get("replacements", [])
        new_replacements_list = []
        if isinstance(original_replacements, list):
            for rep_item in original_replacements:
                if isinstance(rep_item, dict):
                    new_rep_item = rep_item.copy()
                    original_term = rep_item.get("original")
                    entity_type_from_rep = rep_item.get("type")  # Use a different var name

                    new_grouped_id = None
                    # Use consolidated_name_map and consolidated_occupation_map which map original_term to new_grouped_id
                    if entity_type_from_rep == "PERSON" and original_term in consolidated_name_map:
                        new_grouped_id = consolidated_name_map[original_term]
                    elif entity_type_from_rep in (
                    "OCCUPATION", "POSITION", "JOB") and original_term in consolidated_occupation_map:
                        new_grouped_id = consolidated_occupation_map[original_term]

                    if new_grouped_id:
                        new_rep_item["replacement"] = new_grouped_id
                    else:
                        # If no new grouped ID based on original_term,
                        # check if the old replacement ID itself needs to be updated.
                        old_id_in_rep = rep_item.get("replacement")
                        if old_id_in_rep in combined_old_to_new_ids_map:
                            new_rep_item["replacement"] = combined_old_to_new_ids_map[old_id_in_rep]
                        else:
                            logger.debug(
                                f"Term '{original_term}' or its old ID '{old_id_in_rep}' not found in consolidated maps for replacements list. Keeping old replacement.")

                    new_replacements_list.append(new_rep_item)
                else:
                    new_replacements_list.append(rep_item)  # Keep non-dict items as is
        new_article_entry["replacements"] = new_replacements_list
        refined_data_output.append(new_article_entry)

    save_json_file(refined_data_output, args.output_refined_anonymized_file)
    logger.info("Anonymization refinement process complete.")
    logger.info(f"Final API Usage for this run: {rate_limiter.get_usage_stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine anonymized data by grouping similar entities using an LLM.")

    # Setting default values and removing required=True for file paths
    parser.add_argument("--name-map-file",
                        default="first_step_anon/anonymized_output_single_key_name_map.json",
                        help="Path to the original name map JSON file.")
    parser.add_argument("--occupation-map-file",
                        default="first_step_anon/anonymized_output_single_key_occupation_map.json",
                        help="Path to the original occupation map JSON file.")
    parser.add_argument("--anonymized-input-file",
                        default="first_step_anon/anonymized_output_single_key.json",
                        help="Path to the main anonymized data JSON file.")

    parser.add_argument("--output-refined-anonymized-file",
                        default="second_step_anon/refined_anonymized_output.json",
                        # Changed default name slightly for clarity
                        help="Path to save the refined anonymized data.")
    parser.add_argument("--output-consolidated-name-map",
                        default="second_step_anon/consolidated_name_map.json",  # Changed default name
                        help="Path to save the new consolidated name map (original_term -> GROUPED_ID).")
    parser.add_argument("--output-consolidated-occupation-map",
                        default="second_step_anon/consolidated_occupation_map.json",  # Changed default name
                        help="Path to save the new consolidated occupation map (original_term -> GROUPED_ID).")
    parser.add_argument("--output-old-name-ids-map",
                        default="second_step_anon/old_name_ids_to_grouped_ids.json",  # Changed default name
                        help="Path to save map from old person IDs to new grouped person IDs.")
    parser.add_argument("--output-old-occupation-ids-map",
                        default="second_step_anon/old_occupation_ids_to_grouped_ids.json",  # Changed default name
                        help="Path to save map from old occupation IDs to new grouped occupation IDs.")

    parser.add_argument('--api-key', type=str, default=None, help='Gemini API Key string.')
    parser.add_argument('--api-key-file', type=str, default='api_key.txt', help='File containing the Gemini API Key.')
    parser.add_argument('--fresh-consolidation', action='store_true',
                        help='Force re-running LLM grouping even if consolidated map files exist.')

    args = parser.parse_args()

    # Create output directories if they don't exist, based on default paths
    # This is good practice if defaults point to subdirectories.
    # The save_json_file function also attempts this, but doing it early can be clearer.
    output_dirs_to_check = [
        os.path.dirname(args.output_refined_anonymized_file),
        os.path.dirname(args.output_consolidated_name_map),
        os.path.dirname(args.output_consolidated_occupation_map),
        os.path.dirname(args.output_old_name_ids_map),
        os.path.dirname(args.output_old_occupation_ids_map),
    ]
    for out_dir in output_dirs_to_check:
        if out_dir and not os.path.exists(out_dir):  # Check if out_dir is not empty
            try:
                os.makedirs(out_dir, exist_ok=True)
                logger.info(f"Ensured output directory exists: {out_dir}")
            except Exception as e:
                logger.error(f"Could not create output directory {out_dir}: {e}")

    try:
        asyncio.run(main_refinement_process(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Critical error in refinement script: {e}", exc_info=True)

