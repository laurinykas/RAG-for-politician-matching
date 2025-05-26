import time
import re
import os
import random
from urllib.parse import urlparse
from datetime import datetime
import json
from selenium.webdriver.common.keys import Keys

try:
    import selenium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
    import requests.exceptions
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pip install selenium webdriver-manager requests")
    import sys

    sys.exit(1)

# --- Configuration ---
SEARCH_QUERY = ""  # Empty as requested, to load from general /paieska page
BASE_SEARCH_URL = "https://www.lrt.lt/paieska"
BASE_DOMAIN = "https://www.lrt.lt"
SEARCH_KEYWORD = "Kasčiūnas"

# --- Timing and Limits ---
REQUEST_DELAY_PAGE_LOAD = 2  # Delay after navigating to a new article page
REQUEST_DELAY_ACTION = 3  # Delay after actions like clicking "load more"
REQUEST_DELAY_FILTER = 1  # Delay after filter interactions
MAX_ARTICLES_OVERALL = 500  # STRICT limit on total articles to process
MAX_DAUGIAU_CLICKS = 100
TIMEOUT_WAIT = 15  # Seconds to wait for elements before timing out

# --- New configuration for improvements ---
BATCH_SIZE = 100  # Save every 100 articles
MAX_RETRIES = 5  # Maximum retries for connection errors
RETRY_DELAY_BASE = 10  # Base delay in seconds before retrying (will be randomized)
RETRY_DELAY_MAX = 60  # Maximum delay in seconds before retrying

# --- Output ---
OUTPUT_FOLDER = "./lrt_data"  # Folder to store all output files - absolute path with ./ prefix
OUTPUT_FILE = f"lrt_{SEARCH_KEYWORD}_articles.json"

# --- Selectors ---
LOAD_MORE_BUTTON_SELECTOR = "button.btn.btn--lg.section__button[aria-label='Daugiau rezultatų']"
SEARCH_RESULTS_SELECTOR = "div.search__content__items, div.search__content__results"
ARTICLE_ITEM_SELECTORS = [
    "article.item--search",
    "div.search__content__item",
    "article.item",
    "div.search__content__results article",
    "div.item"
]

# --- Filter Selectors ---
# Based on the HTML images provided
TIPAS_DROPDOWN_SELECTOR = "button.multi-select-dropdown-button:has(span.text-truncate:contains('Tipas'))"
TEMA_DROPDOWN_SELECTOR = "button.multi-select-dropdown-button:has(span.text-truncate:contains('Tema'))"
STRAIPSNIAI_CHECKBOX_SELECTOR = "input[id='id-1'][name='name-1']"
LIETUVOJE_CHECKBOX_SELECTOR = "input[id^='id-'][name^='name-']:has(+label:contains('Lietuvoje'))"


def initialize_driver():
    """Initializes and returns a Selenium WebDriver instance."""
    try:
        from selenium.webdriver.chrome.service import Service as ChromeService
        from webdriver_manager.chrome import ChromeDriverManager

        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36")

        print("Setting up Chrome WebDriver...")
        driver_service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=driver_service, options=options)
        driver.set_page_load_timeout(30)
        print("WebDriver initialized successfully.")
        return driver
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        print("Please ensure you have Google Chrome installed.")
        print("If webdriver-manager fails, you might need to manually specify the chromedriver path.")
        return None


def with_connection_retry(func):
    """
    Decorator to retry functions that might fail due to connection errors.
    Handles connection-related exceptions and retries the operation with exponential backoff.
    """

    def wrapper(*args, **kwargs):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except (TimeoutException, requests.exceptions.RequestException) as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    print(f"Max retries ({MAX_RETRIES}) reached. Could not complete operation: {func.__name__}")
                    # Return a default value or re-raise based on the function
                    if func.__name__ == "extract_article_data_selenium":
                        # For article extraction, return empty data instead of failing
                        print(f"Skipping article due to persistent connection issues.")
                        return {"url": args[1] if len(args) > 1 else "unknown",
                                "keywords": [], "full_text": "", "title": ""}
                    else:
                        # For critical functions like navigation, we might want to re-raise
                        raise

                # Calculate exponential backoff with jitter
                delay = min(RETRY_DELAY_MAX, RETRY_DELAY_BASE * (2 ** (retries - 1)))
                # Add random jitter (up to 30% of delay)
                jitter = random.uniform(0, 0.3 * delay)
                wait_time = delay + jitter

                print(f"Connection error in {func.__name__}: {e}")
                print(f"Waiting {wait_time:.2f} seconds before retry {retries}/{MAX_RETRIES}...")
                time.sleep(wait_time)

                # If the function was called with a driver, try to refresh it
                if args and hasattr(args[0], 'refresh'):
                    try:
                        print("Attempting to refresh the page...")
                        args[0].refresh()
                        time.sleep(REQUEST_DELAY_PAGE_LOAD)
                    except:
                        print("Could not refresh driver, continuing with retry anyway.")
        return None  # This should not be reached due to the re-raise above

    return wrapper


def ensure_output_directory():
    """Ensure the output directory exists"""
    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            print(f"Created output directory: {OUTPUT_FOLDER}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            # Fall back to current directory
            return ""
    return OUTPUT_FOLDER


def find_checkbox_by_text(driver, search_text):
    """Find a checkbox by its associated label text"""
    try:
        # Find all labels containing the text
        labels = driver.find_elements(By.XPATH, f"//label[contains(text(), '{search_text}')]")
        if not labels:
            labels = driver.find_elements(By.XPATH, f"//span[contains(text(), '{search_text}')]/ancestor::label")

        for label in labels:
            try:
                # Get the 'for' attribute to find the checkbox
                for_attr = label.get_attribute('for')
                if for_attr:
                    checkbox = driver.find_element(By.ID, for_attr)
                    return checkbox

                # Alternative: Check for checkbox inside the label
                checkbox = label.find_element(By.TAG_NAME, 'input')
                if checkbox.get_attribute('type') == 'checkbox':
                    return checkbox

                # Last resort: Try to find nearby checkbox
                checkbox = driver.find_element(By.XPATH,
                                               f"//label[contains(text(), '{search_text}')]/preceding::input[@type='checkbox'][1]")
                return checkbox
            except (NoSuchElementException, Exception):
                continue
    except Exception as e:
        print(f"Error finding checkbox with text '{search_text}': {e}")

    return None


def find_dropdown_by_text(driver, text):
    """Find a dropdown button that contains specific text"""
    try:
        # Try first by exact match
        button = driver.find_element(By.XPATH,
                                     f"//button[contains(@class, 'multi-select-dropdown-button')]//span[contains(text(), '{text}')]/ancestor::button")
        return button
    except NoSuchElementException:
        try:
            # Try more generic approach
            buttons = driver.find_elements(By.CSS_SELECTOR, "button.multi-select-dropdown-button")
            for button in buttons:
                if text.lower() in button.text.lower():
                    return button
        except Exception as e:
            print(f"Error in find_dropdown_by_text('{text}'): {e}")

    return None


@with_connection_retry
def apply_filters(driver):
    """Apply the required filters: Tipas > Straipsniai and Tema > Lietuvoje"""
    print("\n--- Applying filters ---")

    # 1. Click Tipas dropdown
    print("Looking for 'Tipas' dropdown...")
    tipas_btn = find_dropdown_by_text(driver, "Tipas")
    if not tipas_btn:
        print("Could not find 'Tipas' dropdown, trying alternative approach...")
        # Try XPath approach
        try:
            tipas_btn = driver.find_element(By.XPATH, "//span[contains(text(), 'Tipas')]/ancestor::button")
        except NoSuchElementException:
            print("Failed to locate 'Tipas' dropdown button.")
            return False

    # Click the dropdown button
    try:
        print("Clicking 'Tipas' dropdown...")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tipas_btn)
        time.sleep(REQUEST_DELAY_FILTER)
        driver.execute_script("arguments[0].click();", tipas_btn)
        time.sleep(REQUEST_DELAY_FILTER * 2)  # Wait for dropdown to expand
        print("'Tipas' dropdown clicked.")
    except Exception as e:
        print(f"Error clicking 'Tipas' dropdown: {e}")
        return False

    # 2. Select Straipsniai checkbox
    try:
        print("Looking for 'Straipsniai' option...")
        straipsniai_checkbox = find_checkbox_by_text(driver, "Straipsniai")

        if not straipsniai_checkbox:
            print("Straipsniai checkbox not found by text. Trying direct approach...")
            # Direct approach - look for labels with straipsniai text
            labels = driver.find_elements(By.XPATH, "//label[contains(., 'Straipsniai')]")
            for label in labels:
                try:
                    for_attr = label.get_attribute('for')
                    if for_attr:
                        straipsniai_checkbox = driver.find_element(By.ID, for_attr)
                        break
                except:
                    continue

        if straipsniai_checkbox:
            print("Found 'Straipsniai' checkbox, clicking...")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", straipsniai_checkbox)
            time.sleep(REQUEST_DELAY_FILTER)
            if not straipsniai_checkbox.is_selected():
                driver.execute_script("arguments[0].click();", straipsniai_checkbox)
                print("'Straipsniai' option selected.")
            else:
                print("'Straipsniai' was already selected.")
        else:
            print("WARNING: Could not find 'Straipsniai' checkbox!")
    except Exception as e:
        print(f"Error selecting 'Straipsniai': {e}")

    # Ensure dropdown closes
    time.sleep(REQUEST_DELAY_FILTER)
    try:
        # Click outside to close the dropdown
        driver.find_element(By.TAG_NAME, "body").click()
        time.sleep(REQUEST_DELAY_FILTER)
    except:
        pass

    # 3. Click Tema dropdown
    print("\nLooking for 'Tema' dropdown...")
    tema_btn = find_dropdown_by_text(driver, "Tema")
    if not tema_btn:
        print("Could not find 'Tema' dropdown, trying alternative approach...")
        # Try XPath approach
        try:
            tema_btn = driver.find_element(By.XPATH, "//span[contains(text(), 'Tema')]/ancestor::button")
        except NoSuchElementException:
            print("Failed to locate 'Tema' dropdown button.")
            # Continue anyway since we got at least the Tipas filter
            pass

    if tema_btn:
        # Click the dropdown button
        try:
            print("Clicking 'Tema' dropdown...")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tema_btn)
            time.sleep(REQUEST_DELAY_FILTER)
            driver.execute_script("arguments[0].click();", tema_btn)
            time.sleep(REQUEST_DELAY_FILTER * 2)  # Wait for dropdown to expand
            print("'Tema' dropdown clicked.")
        except Exception as e:
            print(f"Error clicking 'Tema' dropdown: {e}")

        # 4. Select Lietuvoje checkbox
        try:
            print("Looking for 'Lietuvoje' option...")
            lietuvoje_checkbox = find_checkbox_by_text(driver, "Lietuvoje")

            if not lietuvoje_checkbox:
                print("Lietuvoje checkbox not found by text. Trying direct approach...")
                # Direct approach - look for labels with Lietuvoje text
                labels = driver.find_elements(By.XPATH, "//label[contains(., 'Lietuvoje')]")
                for label in labels:
                    try:
                        for_attr = label.get_attribute('for')
                        if for_attr:
                            lietuvoje_checkbox = driver.find_element(By.ID, for_attr)
                            break
                    except:
                        continue

            if lietuvoje_checkbox:
                print("Found 'Lietuvoje' checkbox, clicking...")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", lietuvoje_checkbox)
                time.sleep(REQUEST_DELAY_FILTER)
                if not lietuvoje_checkbox.is_selected():
                    driver.execute_script("arguments[0].click();", lietuvoje_checkbox)
                    print("'Lietuvoje' option selected.")
                else:
                    print("'Lietuvoje' was already selected.")
            else:
                print("WARNING: Could not find 'Lietuvoje' checkbox!")
        except Exception as e:
            print(f"Error selecting 'Lietuvoje': {e}")

    # Final delay before continuing - no apply button needed based on requirements
    print("Filters applied successfully. Continuing with scraping...")
    time.sleep(REQUEST_DELAY_FILTER * 2)
    return True


def is_valid_article_url(url):
    """Check if URL is a valid LRT article URL."""
    if not url:
        return False
    if not url.startswith(('http://', 'https://')):
        url = BASE_DOMAIN + url if url.startswith('/') else BASE_DOMAIN + '/' + url
    parsed = urlparse(url)
    if parsed.netloc not in ['www.lrt.lt', 'lrt.lt']:
        return False
    valid_sections = ['/naujienos/', '/mediateka/', '/sportas/', '/kultura/',
                      '/verslas/', '/gyvenimas/', '/lituanica/']
    path = parsed.path.lower()
    for section in valid_sections:
        if section in path:
            return True
    return False


@with_connection_retry
def extract_article_links_from_search_page_selenium(driver):
    """
    Extracts individual article links from the currently loaded search results page.
    """
    links = set()
    try:
        try:
            WebDriverWait(driver, TIMEOUT_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SEARCH_RESULTS_SELECTOR))
            )
        except TimeoutException:
            print("Timeout waiting for search results container. Looking for individual items directly.")

        articles_found = False
        for selector in ARTICLE_ITEM_SELECTORS:
            article_item_containers = driver.find_elements(By.CSS_SELECTOR, selector)
            if article_item_containers:
                articles_found = True
                print(f"Found {len(article_item_containers)} articles with selector: {selector}")
                for item_container in article_item_containers:
                    link_candidates = []
                    try:
                        link_candidates.extend(item_container.find_elements(
                            By.CSS_SELECTOR, "a.item__title, a.title, a.headline, h3 a, h2 a"
                        ))
                    except Exception:
                        pass
                    if not link_candidates:
                        try:
                            link_candidates.extend(item_container.find_elements(By.TAG_NAME, "a"))
                        except Exception:
                            continue
                    for link_tag in link_candidates:
                        try:
                            href = link_tag.get_attribute('href')
                            if href and is_valid_article_url(href):
                                links.add(href)
                        except Exception:
                            continue
                if links:
                    break
        if not articles_found:
            print("No article containers found with primary selectors. Trying to find all links on page as fallback...")
            all_links = driver.find_elements(By.TAG_NAME, "a")
            for link_tag in all_links:
                try:
                    href = link_tag.get_attribute('href')
                    if href and is_valid_article_url(href):
                        links.add(href)
                except Exception:
                    continue
    except Exception as e:
        print(f"Error extracting links from search page: {e}")
    if not links:
        print("WARNING: No article links found on this page view. Check selectors or page structure.")
    else:
        print(f"Found {len(links)} unique article links on this page view.")
    return list(links)


@with_connection_retry
def extract_article_data_selenium(driver, article_url):
    """
    Navigates to an article URL and extracts keywords and full text using Selenium.
    """
    print(f"Navigating to article: {article_url}")
    article_data = {"url": article_url, "keywords": [], "full_text": "", "title": ""}
    try:
        driver.get(article_url)
        WebDriverWait(driver, TIMEOUT_WAIT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(REQUEST_DELAY_PAGE_LOAD)

        title_selectors = ["h1.article__headline", "h1.article-title", "h1.title",
                           "div.article__heading h1", "header h1"]
        for selector in title_selectors:
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, selector)
                article_data["title"] = title_element.text.strip()
                if article_data["title"]: break
            except NoSuchElementException:
                continue
        if not article_data["title"]:
            try:
                article_data["title"] = driver.title.replace(" - LRT", "").strip()
            except:
                pass

        keyword_selectors = ["//meta[@name='keywords']", "//meta[@name='news_keywords']",
                             "//meta[@property='article:tag']"]
        for selector in keyword_selectors:
            try:
                meta_tags = driver.find_elements(By.XPATH, selector)
                for tag in meta_tags:
                    content = tag.get_attribute("content")
                    if content: article_data["keywords"].extend([kw.strip() for kw in content.split(',') if kw.strip()])
            except:
                continue
        tag_selectors = ["div.tags a", "ul.tags li a", "div.keywords a",
                         "a[rel='tag']", "div.article__tags a"]
        for selector in tag_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                article_data["keywords"].extend([el.text.strip() for el in elements if el.text.strip()])
            except:
                continue
        article_data["keywords"] = list(set(article_data["keywords"]))

        content_selectors = [
            "div.article-body", "div.article__body", "article.article-content",
            "div[itemprop='articleBody']", "div.content__article", "div.entry-content",
            "div.text.str", "div.article-content__block", "div.article__content",
            "main article", "div.single-content"
        ]
        for selector in content_selectors:
            try:
                content_block = driver.find_element(By.CSS_SELECTOR, selector)
                text_elements = content_block.find_elements(
                    By.CSS_SELECTOR, "p, h1, h2, h3, h4, h5, h6, li, blockquote"
                )
                text_parts = []
                for el in text_elements:
                    try:
                        element_text = el.text.strip()
                        if element_text and len(element_text.split()) > 2:
                            if not any(x in element_text.lower() for x in ["nuotr.", "aut.", "šaltinis:", "©"]):
                                text_parts.append(element_text)
                    except:
                        continue
                if text_parts:
                    article_data["full_text"] = "\n\n".join(text_parts).strip()
                    article_data["full_text"] = re.sub(r'\n\s*\n', '\n\n', article_data["full_text"])
                    print(f"Extracted text ({len(article_data['full_text'])} chars) using selector: {selector}")
                    break
            except NoSuchElementException:
                continue
            except Exception as e:
                print(f"Error extracting content with selector {selector}: {e}")
        if not article_data["full_text"]:
            try:
                paragraphs = driver.find_elements(By.TAG_NAME, "p")
                text_parts = [p.text.strip() for p in paragraphs if p.text.strip() and len(p.text.split()) > 3]
                if text_parts:
                    article_data["full_text"] = "\n\n".join(text_parts).strip()
                    print(f"Fallback: Extracted {len(article_data['full_text'])} chars from all paragraphs")
            except:
                pass
    except TimeoutException:
        print(f"Timeout loading article: {article_url}")
    except Exception as e:
        print(f"Error processing article {article_url}: {e}")
    return article_data


def handle_cookie_banner(driver):
    """Try to close cookie banners that might interfere with clicking"""
    cookie_selectors = [
        "button.agree-button", "button.accept-cookies", "button.cookie-accept",
        "button.consent-accept", "div.cookie-notice button", "div.cookies button",
        "#cookieConsent button", "button[data-testid='cookie-popup-accept']",
        "button#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"  # Common ID for Cookiebot
    ]
    for selector in cookie_selectors:
        try:
            cookie_buttons = driver.find_elements(By.CSS_SELECTOR, selector)
            if cookie_buttons:
                for button in cookie_buttons:  # Try clicking any visible one
                    if button.is_displayed() and button.is_enabled():
                        driver.execute_script("arguments[0].click();", button)
                        print(f"Clicked a cookie consent button with selector: {selector}")
                        time.sleep(1.5)  # Wait for banner to disappear
                        return True
        except:
            continue
    return False


@with_connection_retry
def click_load_more(driver):
    """Attempt to click the 'load more' button with better error handling"""
    try:
        # MODIFIED: Increased timeout for finding the button
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, LOAD_MORE_BUTTON_SELECTOR))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", load_more_button)
        time.sleep(0.5)  # Wait for scroll
        try:
            driver.execute_script("arguments[0].click();", load_more_button)
            print("Clicked 'Load More' with JavaScript.")
            return True
        except Exception as e_js:
            print(f"JavaScript click failed for 'Load More': {e_js}. Trying regular click.")
            try:
                load_more_button.click()
                print("Clicked 'Load More' with regular click.")
                return True
            except ElementClickInterceptedException:
                print("ElementClickInterceptedException for 'Load More'. Handling potential overlay.")
                if handle_cookie_banner(driver):  # Try to handle cookie banner again
                    time.sleep(1)
                    try:  # Retry click after handling overlay
                        load_more_button = driver.find_element(By.CSS_SELECTOR, LOAD_MORE_BUTTON_SELECTOR)  # Re-find
                        driver.execute_script("arguments[0].click();", load_more_button)
                        print("Clicked 'Load More' with JavaScript after handling overlay.")
                        return True
                    except Exception as e_retry:
                        print(f"Retry click after overlay failed: {e_retry}")
                        return False
                return False
            except Exception as e_regular:
                print(f"Regular click failed for 'Load More': {e_regular}")
                return False
    except TimeoutException:
        print(f"Load more button ('{LOAD_MORE_BUTTON_SELECTOR}') not found or not clickable within timeout.")
        # driver.save_screenshot("debug_load_more_not_found.png") # Optional: for debugging
        return False
    except Exception as e:
        print(f"An unexpected error occurred while trying to click 'Load More': {e}")
        return False


def save_batch(data, batch_number, output_folder):
    """Save a batch of articles to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created output directory for batch: {output_folder}")
        except Exception as e:
            print(f"Error creating output directory for batch: {e}")
            # Try current directory as fallback
            output_folder = "."

    batch_filename = os.path.join(output_folder, f"lrt_{SEARCH_KEYWORD}_batch_{batch_number}_{timestamp}.json")
    try:
        with open(batch_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Batch {batch_number} saved with {len(data)} articles to {batch_filename}")
        return True
    except Exception as e:
        print(f"Error saving batch {batch_number}: {e}")
        # Try saving to current directory as fallback
        try:
            alt_filename = f"lrt_articles_batch_{batch_number}_{timestamp}.json"
            with open(alt_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Batch {batch_number} saved to current directory as fallback")
            return True
        except Exception as e2:
            print(f"Fatal error saving batch {batch_number}: {e2}")
            return False


def save_progress(data, filename=None, output_folder=""):
    """Save current data to avoid losing progress if script crashes"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lrt_{SEARCH_KEYWORD}_progress_{timestamp}.json"

    # Ensure output folder exists
    if output_folder and not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created output directory for progress: {output_folder}")
        except Exception as e:
            print(f"Error creating output directory for progress: {e}")
            # Fall back to current directory
            output_folder = ""

    full_path = os.path.join(output_folder, filename) if output_folder else filename

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Progress saved: {len(data)} articles to {full_path}")
        return True
    except Exception as e:
        print(f"Error saving progress to {full_path}: {e}")
        # Try saving to current directory as fallback
        if output_folder:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Progress saved to current directory as fallback: {filename}")
                return True
            except Exception as e2:
                print(f"Fatal error saving progress: {e2}")
        return False


@with_connection_retry
def apply_filters_and_search(driver):
    """Apply the required filters: Tipas > Straipsniai and Tema > Lietuvoje, then perform search with keyword"""
    print("\n--- Applying filters ---")

    # 1. Click Tipas dropdown
    print("Looking for 'Tipas' dropdown...")
    tipas_btn = find_dropdown_by_text(driver, "Tipas")
    if not tipas_btn:
        print("Could not find 'Tipas' dropdown, trying alternative approach...")
        # Try XPath approach
        try:
            tipas_btn = driver.find_element(By.XPATH, "//span[contains(text(), 'Tipas')]/ancestor::button")
        except NoSuchElementException:
            print("Failed to locate 'Tipas' dropdown button.")
            return False

    # Click the dropdown button
    try:
        print("Clicking 'Tipas' dropdown...")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tipas_btn)
        time.sleep(REQUEST_DELAY_FILTER)
        driver.execute_script("arguments[0].click();", tipas_btn)
        time.sleep(REQUEST_DELAY_FILTER * 2)  # Wait for dropdown to expand
        print("'Tipas' dropdown clicked.")
    except Exception as e:
        print(f"Error clicking 'Tipas' dropdown: {e}")
        return False

    # 2. Select Straipsniai checkbox
    try:
        print("Looking for 'Straipsniai' option...")
        straipsniai_checkbox = find_checkbox_by_text(driver, "Straipsniai")

        if not straipsniai_checkbox:
            print("Straipsniai checkbox not found by text. Trying direct approach...")
            # Direct approach - look for labels with straipsniai text
            labels = driver.find_elements(By.XPATH, "//label[contains(., 'Straipsniai')]")
            for label in labels:
                try:
                    for_attr = label.get_attribute('for')
                    if for_attr:
                        straipsniai_checkbox = driver.find_element(By.ID, for_attr)
                        break
                except:
                    continue

        if straipsniai_checkbox:
            print("Found 'Straipsniai' checkbox, clicking...")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", straipsniai_checkbox)
            time.sleep(REQUEST_DELAY_FILTER)
            if not straipsniai_checkbox.is_selected():
                driver.execute_script("arguments[0].click();", straipsniai_checkbox)
                print("'Straipsniai' option selected.")
            else:
                print("'Straipsniai' was already selected.")
        else:
            print("WARNING: Could not find 'Straipsniai' checkbox!")
    except Exception as e:
        print(f"Error selecting 'Straipsniai': {e}")

    # Ensure dropdown closes
    time.sleep(REQUEST_DELAY_FILTER)
    try:
        # Click outside to close the dropdown
        driver.find_element(By.TAG_NAME, "body").click()
        time.sleep(REQUEST_DELAY_FILTER)
    except:
        pass

    # 3. Click Tema dropdown
    print("\nLooking for 'Tema' dropdown...")
    tema_btn = find_dropdown_by_text(driver, "Tema")
    if not tema_btn:
        print("Could not find 'Tema' dropdown, trying alternative approach...")
        # Try XPath approach
        try:
            tema_btn = driver.find_element(By.XPATH, "//span[contains(text(), 'Tema')]/ancestor::button")
        except NoSuchElementException:
            print("Failed to locate 'Tema' dropdown button.")
            # Continue anyway since we got at least the Tipas filter
            pass

    if tema_btn:
        # Click the dropdown button
        try:
            print("Clicking 'Tema' dropdown...")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tema_btn)
            time.sleep(REQUEST_DELAY_FILTER)
            driver.execute_script("arguments[0].click();", tema_btn)
            time.sleep(REQUEST_DELAY_FILTER * 2)  # Wait for dropdown to expand
            print("'Tema' dropdown clicked.")
        except Exception as e:
            print(f"Error clicking 'Tema' dropdown: {e}")

        # 4. Select Lietuvoje checkbox
        try:
            print("Looking for 'Lietuvoje' option...")
            lietuvoje_checkbox = find_checkbox_by_text(driver, "Lietuvoje")

            if not lietuvoje_checkbox:
                print("Lietuvoje checkbox not found by text. Trying direct approach...")
                # Direct approach - look for labels with Lietuvoje text
                labels = driver.find_elements(By.XPATH, "//label[contains(., 'Lietuvoje')]")
                for label in labels:
                    try:
                        for_attr = label.get_attribute('for')
                        if for_attr:
                            lietuvoje_checkbox = driver.find_element(By.ID, for_attr)
                            break
                    except:
                        continue

            if lietuvoje_checkbox:
                print("Found 'Lietuvoje' checkbox, clicking...")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", lietuvoje_checkbox)
                time.sleep(REQUEST_DELAY_FILTER)
                if not lietuvoje_checkbox.is_selected():
                    driver.execute_script("arguments[0].click();", lietuvoje_checkbox)
                    print("'Lietuvoje' option selected.")
                else:
                    print("'Lietuvoje' was already selected.")
            else:
                print("WARNING: Could not find 'Lietuvoje' checkbox!")
        except Exception as e:
            print(f"Error selecting 'Lietuvoje': {e}")

    # Final delay before continuing
    time.sleep(REQUEST_DELAY_FILTER * 2)

    # Now perform the search with the constant keyword
    print(f"\n--- Performing search with keyword: '{SEARCH_KEYWORD}' ---")

    # Multiple approaches to find and interact with the search input
    search_successful = False

    # Approach 1: Using the exact selectors from the inspected element
    try:
        search_input = WebDriverWait(driver, TIMEOUT_WAIT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input.form-input__input.d-block.border-0.flex-grow-1[type='search']"))
        )

        print("Found search input field (Approach 1)")

        # Clear any existing text
        search_input.clear()
        time.sleep(0.5)

        # Enter the search keyword
        search_input.send_keys(SEARCH_KEYWORD)
        time.sleep(1)

        # Press Enter to submit
        search_input.send_keys(Keys.RETURN)
        print(f"Entered '{SEARCH_KEYWORD}' in search field and submitted")

        time.sleep(REQUEST_DELAY_PAGE_LOAD)
        search_successful = True
    except Exception as e:
        print(f"Approach 1 failed: {e}")

    # Approach 2: Using more generic selectors if the first approach fails
    if not search_successful:
        try:
            search_input = WebDriverWait(driver, TIMEOUT_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search']"))
            )

            print("Found search input field (Approach 2)")

            # Clear any existing text
            search_input.clear()
            time.sleep(0.5)

            # Enter the search keyword
            search_input.send_keys(SEARCH_KEYWORD)
            time.sleep(1)

            # Press Enter to submit
            search_input.send_keys(Keys.RETURN)
            print(f"Entered '{SEARCH_KEYWORD}' in search field and submitted")

            time.sleep(REQUEST_DELAY_PAGE_LOAD)
            search_successful = True
        except Exception as e:
            print(f"Approach 2 failed: {e}")

    # Approach 3: Try using form submission directly
    if not search_successful:
        try:
            search_form = WebDriverWait(driver, TIMEOUT_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form.form-input"))
            )

            search_input = search_form.find_element(By.CSS_SELECTOR, "input[type='search']")
            print("Found search input field in form (Approach 3)")

            # Clear any existing text
            search_input.clear()
            time.sleep(0.5)

            # Enter the search keyword
            search_input.send_keys(SEARCH_KEYWORD)
            time.sleep(1)

            # Submit the form instead of pressing Enter
            search_form.submit()
            print(f"Entered '{SEARCH_KEYWORD}' in search field and submitted form")

            time.sleep(REQUEST_DELAY_PAGE_LOAD)
            search_successful = True
        except Exception as e:
            print(f"Approach 3 failed: {e}")

    # Approach 4: Try using JavaScript to set the value and trigger search
    if not search_successful:
        try:
            js_script = """
            var searchInput = document.querySelector('input[type="search"]');
            if (searchInput) {
                searchInput.value = arguments[0];

                // Create and dispatch an input event
                var inputEvent = new Event('input', { bubbles: true });
                searchInput.dispatchEvent(inputEvent);

                // Create and dispatch an Enter key event
                var enterEvent = new KeyboardEvent('keydown', {
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13,
                    which: 13,
                    bubbles: true
                });
                searchInput.dispatchEvent(enterEvent);

                return true;
            }
            return false;
            """

            result = driver.execute_script(js_script, SEARCH_KEYWORD)
            if result:
                print(f"Used JavaScript to set search value to '{SEARCH_KEYWORD}' and trigger search")
                time.sleep(REQUEST_DELAY_PAGE_LOAD)
                search_successful = True
            else:
                print("JavaScript approach couldn't find search input")
        except Exception as e:
            print(f"JavaScript approach failed: {e}")

    if search_successful:
        print("Search operation completed successfully")
    else:
        print("All approaches to perform search failed")

    print("Filters applied and search attempted. Continuing with scraping...")
    return True
def merge_batches(output_folder, output_filename):
    """
    Merge all batch files into a single output file and return the total count
    """
    all_data = []

    # Ensure output folder exists and is accessible
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist. Creating it.")
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            print(f"Could not create output folder: {e}")
            output_folder = "."  # Fall back to current directory

    # Find all batch files
    try:
        batch_files = [f for f in os.listdir(output_folder) if
                       f.startswith(f"lrt_{SEARCH_KEYWORD}_batch_") and f.endswith(".json")]
    except Exception as e:
        print(f"Error listing directory {output_folder}: {e}")
        # Try current directory as fallback
        try:
            batch_files = [f for f in os.listdir(".") if f.startswith("lrt_articles_batch_") and f.endswith(".json")]
            output_folder = "."
        except:
            print("Fatal error: Cannot access batch files in any directory")
            return 0

    if not batch_files:
        print("No batch files found to merge.")
        return 0

    print(f"Found {len(batch_files)} batch files to merge.")
    for batch_file in sorted(batch_files):
        try:
            full_path = os.path.join(output_folder, batch_file)
            with open(full_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)
                print(f"Added {len(batch_data)} articles from {batch_file}")
        except Exception as e:
            print(f"Error reading batch file {batch_file}: {e}")

    if all_data:
        try:
            full_output_path = os.path.join(output_folder, output_filename)
            with open(full_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"Merged {len(all_data)} articles into {output_filename}")
        except Exception as e:
            print(f"Error saving merged file to {full_output_path}: {e}")
            # Try current directory as fallback
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                print(f"Merged file saved to current directory as fallback")
            except Exception as e2:
                print(f"Fatal error saving merged file: {e2}")

    return len(all_data)


def check_for_existing_progress(output_folder):
    """Check for existing progress files and ask if user wants to resume"""
    urls_file = os.path.join(output_folder, f"collected_urls_{SEARCH_KEYWORD}_progress.json")
    if os.path.exists(urls_file):
        try:
            with open(urls_file, 'r', encoding='utf-8') as f:
                urls = json.load(f)
                if urls:
                    print(f"Found previous URL collection with {len(urls)} URLs.")
                    user_input = input("Resume from previous URL collection? (y/n): ").strip().lower()
                    if user_input.startswith('y'):
                        return urls
        except Exception as e:
            print(f"Error reading previous progress: {e}")
    return None


def main():
    # Ensure output directory exists
    try:
        output_folder = ensure_output_directory()
        print(f"Starting LRT scraper for: {BASE_SEARCH_URL} with keyword: '{SEARCH_KEYWORD}'")
        print(f"Output files will include the keyword '{SEARCH_KEYWORD}' in their names")
        print(f"Max articles (strict limit): {MAX_ARTICLES_OVERALL}, Max pagination clicks: {MAX_DAUGIAU_CLICKS}")
        print(f"Load More Button Selector: '{LOAD_MORE_BUTTON_SELECTOR}'")
        print(f"Batch size: {BATCH_SIZE}, Output folder: {output_folder}")

        # Print absolute path of output folder for debugging
        abs_path = os.path.abspath(output_folder)
        print(f"Absolute path of output folder: {abs_path}")
        print(f"Current working directory: {os.getcwd()}")

        # Check if output folder is writable
        try:
            test_file = os.path.join(output_folder, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Output folder is writable: {output_folder}")
        except Exception as e:
            print(f"WARNING: Output folder may not be writable: {e}")
            print("Will attempt to use current directory as fallback if needed.")

        # Check for previous progress
        previous_urls = check_for_existing_progress(output_folder)

        # Track the script's progress state
        script_state = {
            "url_collection_complete": False,
            "current_batch": 1,
            "articles_processed": 0,
            "batches_saved": 0
        }
    except Exception as e:
        print(f"Error during initialization: {e}")
        output_folder = "."
        previous_urls = None
        script_state = {
            "url_collection_complete": False,
            "current_batch": 1,
            "articles_processed": 0,
            "batches_saved": 0
        }

    driver = None
    try:
        driver = initialize_driver()
        if not driver:
            print("WebDriver initialization failed. Exiting.")
            return

        all_articles_data = []
        batch_articles = []  # For collecting articles in batches
        collected_article_urls = set(previous_urls) if previous_urls else set()

        # Begin URL collection if we don't already have them from a previous run
        if not collected_article_urls or not script_state["url_collection_complete"]:
            try:
                print(f"Navigating to search page: {BASE_SEARCH_URL}")
                driver.get(BASE_SEARCH_URL)
                WebDriverWait(driver, TIMEOUT_WAIT).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                time.sleep(REQUEST_DELAY_PAGE_LOAD)
                handle_cookie_banner(driver)

                # Apply filters before starting the scraping process
                filter_success = apply_filters_and_search(driver)
                if not filter_success:
                    print("Warning: Filters may not have been fully applied. Continuing with scraping...")

                # Let the page settle after filter application
                time.sleep(REQUEST_DELAY_ACTION)
            except Exception as e:
                print(f"Error during initial setup and filter application: {e}")
                print("Attempting to continue despite setup error...")

            if not collected_article_urls:
                print("\n--- Extracting links from initial filtered page ---")
                initial_links = extract_article_links_from_search_page_selenium(driver)
                for link in initial_links:
                    collected_article_urls.add(link)
                print(f"Collected {len(collected_article_urls)} unique URLs from initial filtered page.")

                click_count = 0
                consecutive_failures = 0
                max_failures = 3

                while (len(collected_article_urls) < MAX_ARTICLES_OVERALL and
                       click_count < MAX_DAUGIAU_CLICKS and
                       consecutive_failures < max_failures):
                    print(f"\n--- Attempting pagination: Load More click #{click_count + 1} ---")
                    click_successful = click_load_more(driver)

                    if click_successful:
                        consecutive_failures = 0
                        print("Successfully clicked 'Load More'. Waiting for new content to load...")

                        # Enhanced wait logic for content loading
                        dynamic_wait_timeout = max(5, REQUEST_DELAY_ACTION * 2)
                        try:
                            button_elements = driver.find_elements(By.CSS_SELECTOR, LOAD_MORE_BUTTON_SELECTOR)
                            if button_elements and button_elements[0].is_displayed():
                                WebDriverWait(driver, dynamic_wait_timeout).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, LOAD_MORE_BUTTON_SELECTOR))
                                )
                                print(f"Load More button is clickable again (waited up to {dynamic_wait_timeout}s).")
                            else:
                                print("Load More button seems to have disappeared or is no longer displayed.")
                        except TimeoutException:
                            print(f"Timed out waiting for Load More button to be clickable.")
                        except NoSuchElementException:
                            print("Load More button not found after click, likely end of results.")

                        # General delay for rendering and final settlement of content
                        time.sleep(REQUEST_DELAY_ACTION)

                        new_links = extract_article_links_from_search_page_selenium(driver)
                        initial_count_on_this_iteration = len(collected_article_urls)
                        for link in new_links:
                            collected_article_urls.add(link)
                        new_added = len(collected_article_urls) - initial_count_on_this_iteration
                        print(
                            f"Added {new_added} new unique URLs this iteration (total unique URLs: {len(collected_article_urls)})")

                        if new_added == 0 and new_links:
                            print(
                                f"Warning: No new *unique* URLs added despite finding {len(new_links)} links on page.")
                            consecutive_failures += 1
                        elif not new_links and not button_elements:
                            print("No new links found and Load More button disappeared. Assuming end of results.")
                            break

                        # Save URL progress every few clicks
                        if (click_count + 1) % 2 == 0 and collected_article_urls:
                            save_progress(list(collected_article_urls), f"collected_urls_{SEARCH_KEYWORD}_progress.json", output_folder)
                    else:
                        consecutive_failures += 1
                        print(f"Load more click failed. Consecutive failures: {consecutive_failures}/{max_failures}")
                        if consecutive_failures >= max_failures:
                            print("Max consecutive failures reached for clicking 'Load More'. Stopping pagination.")

                    click_count += 1
                    if len(collected_article_urls) >= MAX_ARTICLES_OVERALL:
                        print(f"Reached MAX_ARTICLES_OVERALL ({MAX_ARTICLES_OVERALL}). Stopping pagination.")
                        break
                    if click_count >= MAX_DAUGIAU_CLICKS:
                        print(f"Reached MAX_DAUGIAU_CLICKS ({MAX_DAUGIAU_CLICKS}). Stopping pagination.")
                        break

            script_state["url_collection_complete"] = True

            # Save final URL collection
            if collected_article_urls:
                save_progress(list(collected_article_urls), "collected_urls_progress.json", output_folder)

        # IMPORTANT: Enforce maximum article count strictly before starting article processing
        final_urls_to_scrape = list(collected_article_urls)
        if len(final_urls_to_scrape) > MAX_ARTICLES_OVERALL:
            print(f"Limiting URL list from {len(final_urls_to_scrape)} to MAX_ARTICLES_OVERALL={MAX_ARTICLES_OVERALL}")
            final_urls_to_scrape = final_urls_to_scrape[:MAX_ARTICLES_OVERALL]

        print(f"\n--- Starting article content extraction for {len(final_urls_to_scrape)} articles ---")

        # Check for existing batches to determine starting point
        existing_batches = [f for f in os.listdir(output_folder) if
                            f.startswith(f"lrt_{SEARCH_KEYWORD}_batch_") and f.endswith(".json")]
        if existing_batches:
            # Extract batch numbers
            batch_numbers = []
            for batch_file in existing_batches:
                try:
                    # Extract number between "batch_" and "_"
                    parts = batch_file.split("_")
                    if len(parts) > 2:
                        batch_numbers.append(int(parts[2]))
                except:
                    continue

            if batch_numbers:
                highest_batch = max(batch_numbers)
                articles_already_processed = highest_batch * BATCH_SIZE
                script_state["current_batch"] = highest_batch + 1
                script_state["articles_processed"] = articles_already_processed
                script_state["batches_saved"] = highest_batch

                if articles_already_processed < len(final_urls_to_scrape):
                    print(
                        f"Resuming from article {articles_already_processed + 1}, batch {script_state['current_batch']}")
                    final_urls_to_scrape = final_urls_to_scrape[articles_already_processed:]
                else:
                    print("All articles have already been processed in previous runs.")
                    final_urls_to_scrape = []

        for i, url in enumerate(final_urls_to_scrape):
            try:
                current_article_index = script_state["articles_processed"] + i + 1
                print(f"\nProcessing article {current_article_index}/{len(collected_article_urls)}: {url}")

                # Add connection retry logic with exponential backoff
                article_data = extract_article_data_selenium(driver, url)

                if article_data and (article_data["full_text"] or article_data["title"]):
                    batch_articles.append(article_data)

                    # Save batch when it reaches BATCH_SIZE or if we've hit the MAX_ARTICLES_OVERALL limit
                    if len(batch_articles) >= BATCH_SIZE or current_article_index >= MAX_ARTICLES_OVERALL:
                        batch_number = script_state["current_batch"]
                        print(f"Saving batch #{batch_number} with {len(batch_articles)} articles...")
                        batch_saved = save_batch(batch_articles, batch_number, output_folder)
                        if batch_saved:
                            script_state["batches_saved"] += 1
                            script_state["current_batch"] += 1
                            script_state["articles_processed"] += len(batch_articles)
                            # Clear batch after saving to free memory
                            all_articles_data.extend(batch_articles)  # Keep track of total count
                            batch_articles = []

                    # Check if we've reached the overall limit
                    if current_article_index >= MAX_ARTICLES_OVERALL:
                        print(f"Reached MAX_ARTICLES_OVERALL limit of {MAX_ARTICLES_OVERALL}. Stopping.")
                        break
                else:
                    print(f"Skipping article with no content or title: {url}")
            except Exception as e:
                print(f"Error processing article {url}: {e}")
                # Continue with next article instead of crashing
                continue

        # Save any remaining articles in the last batch
        if batch_articles:
            print(f"Saving final batch with {len(batch_articles)} remaining articles...")
            batch_saved = save_batch(batch_articles, script_state["current_batch"], output_folder)
            if batch_saved:
                script_state["batches_saved"] += 1
                all_articles_data.extend(batch_articles)

        # Merge all batches into final output file
        total_articles = merge_batches(output_folder, OUTPUT_FILE)

        if total_articles > 0:
            print(
                f"\nSuccess! Scraped and processed {total_articles} articles across {script_state['batches_saved']} batches.")
            print(f"Final merged data saved to {os.path.join(output_folder, OUTPUT_FILE)}")
        else:
            print("\nNo articles were successfully scraped or processed.")

    except Exception as e:
        print(f"Critical error in main execution: {e}")
        # If we crashed during article processing, try to save any collected articles
        if 'batch_articles' in locals() and batch_articles:
            print("Saving articles collected before error...")
            error_filename = f"lrt_articles_error_recovery_{int(time.time())}.json"
            save_progress(batch_articles, error_filename, output_folder)
    finally:
        # Ensure driver is closed properly
        if driver:
            try:
                driver.quit()
                print("WebDriver closed.")
            except:
                print("Error while closing WebDriver.")


if __name__ == "__main__":
    main()