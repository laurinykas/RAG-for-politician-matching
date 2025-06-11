# RAG for Politician Matching

This repository contains a collection of Python scripts that implement a sophisticated Retrieval-Augmented Generation (RAG) system. The primary goal of this project is to identify Lithuanian political figures from anonymized news articles. The system leverages state-of-the-art language models for both anonymization and identification, supported by a robust pipeline that includes web scraping, data processing, and in-depth performance evaluation.

## Key Features

* **Web Scraping**: A Selenium-based scraper (`scraper.py`) to gather articles from news sources like LRT.
* **Data Anonymization**: A two-step process to anonymize personal and professional entities in the collected articles.
    * `anon4.py`: Initial anonymization using the Gemini API.
    * `improve_anon.py`: Refines the initial anonymization by grouping similar entities for consistency.
* **RAG Pipeline**:
    * `RAG2.py` and `RAG3.py`: Implement the core RAG system with parameter and prompt optimization capabilities.
    * `RAG_final_local.py` and `RAG_local2.py`: Run the optimized RAG pipeline with local models like Llama 3 and Gemma for evaluation.
* **In-Depth Evaluation**: The project includes comprehensive evaluation scripts (`final_llama3.py`) to measure the performance of the models using metrics like accuracy, and F1 scores.
* **Modular Design**: The project is organized into distinct scripts for each stage of the pipeline, from data collection to final evaluation.

## Repository Structure
```
├── scraper.py                           # Scrapes articles from LRT.lt
├── json_combine.py                      # Combines scraped JSON files into a single dataset
├── anon4.py                             # Step 1: Anonymizes text using Gemini API
├── improve_anon.py                      # Step 2: Refines and consolidates anonymized entities using Gemini API
├── spliter_learning_testing.py          # Splits the dataset into training, validation, and testing sets
├── RAG2.py                              # RAG pipeline for parameter optimization
├── RAG3.py                              # RAG pipeline for prompt optimization
├── RAG_local2.py                        # RAG pipeline for local model evaluation (WASN'T USED)
├── RAG_final_local.py                   # Final RAG pipeline with local models and RAGAS evaluation (RAGAS EVALUTION IS CURRENTLY NOT WORKING, USED ACC AND F1)
├── final_llama3.py                      # Final evaluation script for Llama 3 8B model
├── main.py                              # Initial RAG for testing different embeding models (name will be changed in the future)
└── README.md                            # This file
```
*A portion of the code in this repository was generated with the assistance of an AI language model.*
