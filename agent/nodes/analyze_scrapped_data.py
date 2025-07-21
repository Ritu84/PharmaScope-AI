import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import json
import time
from utils.tokenize import graceful_json_parse, extract_valid_entries, safe_json_dump

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.1
)

MIN_DELAY_BETWEEN_REQUESTS = 2.0  
MAX_TOKENS_PER_REQUEST = 25000 

messages = [
    ("system", 
    """You are an expert biomedical data analyst that specializes in drug development pipelines. You are helping extract and normalize information about drug assets targeting a specific molecular target from raw clinical trial data. Your job is to identify only therapeutic trials where a drug or biologic targets the given molecule (e.g., CD47, KRAS), and return the output in a structured JSON format. Ignore any studies that do not involve therapeutic intervention (e.g., biological studies, diagnostics, expression analysis). If the input is in plain text, understand it properly and give the output in JSON, and make sure that JSON output is correctly formatted.
    """
    ),
    
    ("user", """Here is a list of clinical trials involving the target molecule '{target}'. Normalize the following attributes for each valid therapeutic trial:

- drug_name: The drug or biologic being tested (e.g., AK117, Hu5F9-G4)
- phase: One of ["Preclinical", "Phase I", "Phase II", "Phase III", "Approved"]
- status: One of ["Recruiting", "Not yet recruiting", "Active, not recruiting", "Completed", "Terminated"]
- modality: Normalize terms like "antibody", "ADC", "small molecule" into common formats like "Monoclonal Antibody", "Bispecific", etc.
- sponsor: The company or institution leading the trial
- indication: The condition being targeted (e.g., Acute Myeloid Leukemia)
- mechanism_of_action: Describe how the drug works in detail
- acquisition/licensing signals : only add if available, else use empty string

Only include trials that are true drug development efforts (not purely academic, mechanistic, or diagnostic studies). Return the results as a JSON list of assets. If a trial targets multiple molecules, still include it if {target} is among them.

Input:
{clinical_trials_scraped_data}
{euctr_scraped_data}
{pubmed_scraped_data}
{google_patent_abstracts}
"""
    )
]

def estimate_tokens(text):
    """Rough estimation of tokens (4 characters per token)"""
    if isinstance(text, (list, dict)):
        text = json.dumps(text)
    return len(str(text)) // 4

def split_data_by_tokens(data, max_tokens_per_batch=8000):
    """Split data into batches based on token estimation"""
    if not data:
        return []
    
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # If it's a string that can't be parsed, split by length
            if estimate_tokens(data) <= max_tokens_per_batch:
                return [data]
            else:
                chunk_size = max_tokens_per_batch * 3  
                chunks = []
                for i in range(0, len(data), chunk_size):
                    chunks.append(data[i:i + chunk_size])
                return chunks
    
    if not isinstance(data, list):
        return [data] if estimate_tokens(data) <= max_tokens_per_batch else []
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for item in data:
        item_tokens = estimate_tokens(item)
        
        if current_tokens + item_tokens > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        current_batch.append(item)
        current_tokens += item_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def split_pubmed_data(pubmed_data, batch_size=15):
    """Split pubmed data into smaller batches to avoid token limit"""
    if not pubmed_data:
        return []
    
    if isinstance(pubmed_data, str):
        try:
            pubmed_data = json.loads(pubmed_data)
        except json.JSONDecodeError:
            return [pubmed_data] 
    
    if not isinstance(pubmed_data, list):
        return [pubmed_data]
    
    batches = []
    for i in range(0, len(pubmed_data), batch_size):
        batch = pubmed_data[i:i + batch_size]
        batches.append(batch)
    
    return batches

def rate_limited_api_call(func, *args, **kwargs):
    """Make an API call with rate limiting"""
    try:
        result = func(*args, **kwargs)
        time.sleep(MIN_DELAY_BETWEEN_REQUESTS)  
        return result
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            print(f"Rate limit hit, waiting 30 seconds before retry...")
            time.sleep(30)
            return func(*args, **kwargs)
        else:
            raise e

def process_clinical_euctr_batch(target_molecule, clinical_trials_scraped_data, euctr_scraped_data):
    """Process clinical trials and EUCTR data in batches"""
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt | llm | StrOutputParser()
    
    clinical_batches = split_data_by_tokens(clinical_trials_scraped_data, max_tokens_per_batch=6000)
    euctr_batches = split_data_by_tokens(euctr_scraped_data, max_tokens_per_batch=6000)
    
    all_results = []
    
    for i, clinical_batch in enumerate(clinical_batches):
        print(f"Processing clinical trials batch {i+1}/{len(clinical_batches)}")
        
        try:
            result = rate_limited_api_call(
                chain.invoke,
                {
                    "target": target_molecule,
                    "clinical_trials_scraped_data": clinical_batch,
                    "euctr_scraped_data": "",
                    "pubmed_scraped_data": "",
                    "google_patent_abstracts": "",
                    "format": "json"
                }
            )
            
            fallback_template = create_fallback_clinical_entry(target_molecule, "Clinical Trials")
            
            parsed_result = graceful_json_parse(
                result, 
                fallback_template, 
                f"Clinical Trials Batch {i+1}"
            )
            
            required_fields = ["drug_name", "phase", "status", "modality", "sponsor", "indication", "mechanism_of_action"]
            valid_entries = extract_valid_entries(parsed_result, required_fields)
            
            all_results.extend(valid_entries)
            print(f"Successfully processed clinical trials batch {i+1}: {len(valid_entries)} entries")
            
        except Exception as e:
            print(f"Error processing clinical trials batch {i+1}: {str(e)}")
            fallback_entry = create_fallback_clinical_entry(target_molecule, "Clinical Trials")
            fallback_entry["_processing_error"] = True
            fallback_entry["_batch"] = i+1
            fallback_entry["_error_message"] = str(e)
            all_results.append(fallback_entry)
    
    for i, euctr_batch in enumerate(euctr_batches):
        print(f"Processing EUCTR batch {i+1}/{len(euctr_batches)}")
        
        try:
            result = rate_limited_api_call(
                chain.invoke,
                {
                    "target": target_molecule,
                    "clinical_trials_scraped_data": "",
                    "euctr_scraped_data": euctr_batch,
                    "pubmed_scraped_data": "",
                    "google_patent_abstracts": "",
                    "format": "json"
                }
            )
            
            fallback_template = create_fallback_clinical_entry(target_molecule, "EUCTR")
            
            parsed_result = graceful_json_parse(
                result, 
                fallback_template, 
                f"EUCTR Batch {i+1}"
            )
            
            required_fields = ["drug_name", "phase", "status", "modality", "sponsor", "indication", "mechanism_of_action"]
            valid_entries = extract_valid_entries(parsed_result, required_fields)
            
            all_results.extend(valid_entries)
            print(f"Successfully processed EUCTR batch {i+1}: {len(valid_entries)} entries")
            
        except Exception as e:
            print(f"Error processing EUCTR batch {i+1}: {str(e)}")
            fallback_entry = create_fallback_clinical_entry(target_molecule, "EUCTR")
            fallback_entry["_processing_error"] = True
            fallback_entry["_batch"] = i+1
            fallback_entry["_error_message"] = str(e)
            all_results.append(fallback_entry)
    
    return all_results

def process_pubmed_batch(target_molecule, pubmed_batch):
    """Process only PubMed data in a batch"""
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = rate_limited_api_call(
            chain.invoke,
            {
                "target": target_molecule,
                "clinical_trials_scraped_data": "",
                "euctr_scraped_data": "",
                "pubmed_scraped_data": pubmed_batch,
                "google_patent_abstracts": "",
                "format": "json"
            }
        )
        
        fallback_template = create_fallback_clinical_entry(target_molecule, "PubMed")
        
        parsed_result = graceful_json_parse(
            result, 
            fallback_template, 
            "PubMed Batch"
        )
        
        required_fields = ["drug_name", "phase", "status", "modality", "sponsor", "indication", "mechanism_of_action"]
        valid_entries = extract_valid_entries(parsed_result, required_fields)
        
        return valid_entries
        
    except Exception as e:
        print(f"Error processing PubMed batch: {str(e)}")
        fallback_entry = create_fallback_clinical_entry(target_molecule, "PubMed")
        fallback_entry["_processing_error"] = True
        fallback_entry["_error_message"] = str(e)
        return [fallback_entry]

def process_patent_abstracts_batch(target_molecule, patent_abstracts_batch):
    """Process Google Patent Abstracts data in a batch"""
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = rate_limited_api_call(
            chain.invoke,
            {
                "target": target_molecule,
                "clinical_trials_scraped_data": "",
                "euctr_scraped_data": "",
                "pubmed_scraped_data": "",
                "google_patent_abstracts": patent_abstracts_batch,
                "format": "json"
            }
        )
        
        fallback_template = create_fallback_clinical_entry(target_molecule, "Patent Abstracts")
        
        parsed_result = graceful_json_parse(
            result, 
            fallback_template, 
            "Patent Abstracts Batch"
        )
        
        required_fields = ["drug_name", "phase", "status", "modality", "sponsor", "indication", "mechanism_of_action"]
        valid_entries = extract_valid_entries(parsed_result, required_fields)
        
        return valid_entries
        
    except Exception as e:
        print(f"Error processing patent abstracts batch: {str(e)}")
        fallback_entry = create_fallback_clinical_entry(target_molecule, "Patent Abstracts")
        fallback_entry["_processing_error"] = True
        fallback_entry["_error_message"] = str(e)
        return [fallback_entry]

def split_patent_abstracts_data(patent_abstracts_data, batch_size=10):
    """Split patent abstracts data into smaller batches to avoid token limit"""
    if not patent_abstracts_data:
        return []
    if isinstance(patent_abstracts_data, str):
        try:
            patent_abstracts_data = json.loads(patent_abstracts_data)
        except json.JSONDecodeError:
            return [patent_abstracts_data]
    if not isinstance(patent_abstracts_data, list):
        return [patent_abstracts_data]
    batches = []
    for i in range(0, len(patent_abstracts_data), batch_size):
        batch = patent_abstracts_data[i:i + batch_size]
        batches.append(batch)
    return batches

def combine_results(batch_results):
    """Combine results from multiple batches with enhanced error handling"""
    combined_assets = []
    
    for result in batch_results:
        if not result:
            continue
        
        if isinstance(result, list):
            combined_assets.extend(result)
        elif isinstance(result, dict):
            combined_assets.append(result)
        else:
            try:
                parsed_result = graceful_json_parse(result, None, "Combine Results")
                if isinstance(parsed_result, list):
                    combined_assets.extend(parsed_result)
                elif isinstance(parsed_result, dict):
                    combined_assets.append(parsed_result)
            except Exception as e:
                print(f"Failed to parse result in combine_results: {str(e)}")
                # Add fallback entry for unparseable result
                fallback_entry = {
                    "drug_name": "Unknown",
                    "phase": "Unknown",
                    "status": "Unknown",
                    "modality": "Unknown",
                    "sponsor": "Unknown",
                    "indication": "Unknown",
                    "mechanism_of_action": "Unknown",
                    "acquisition/licensing signals": "",
                    "_parsing_error": True,
                    "_error_message": str(e)
                }
                combined_assets.append(fallback_entry)
    
    return combined_assets

def analyze_scrapped_data(state : dict) -> dict:
    print("running analyzer scraped data tool...")
    try: 
        target_molecule = state.get("target", "Unknown")
        clinical_trials_scraped_data = state.get("clinical_trials_scraped_data")
        euctr_scraped_data = state.get("euctr_scraped_data")
        pubmed_scraped_data = state.get("pubmed_scraped_data")
        google_patent_abstracts = state.get("google_patent_abstracts")
        
        batch_results = []
        
        if clinical_trials_scraped_data or euctr_scraped_data:
            print("Processing clinical trials and EUCTR data in batches...")
            try:
                clinical_euctr_results = process_clinical_euctr_batch(
                    target_molecule, 
                    clinical_trials_scraped_data, 
                    euctr_scraped_data
                )
                batch_results.extend(clinical_euctr_results)
                print("Clinical trials and EUCTR batches processed successfully")
            except Exception as e:
                print(f"Error processing clinical trials and EUCTR batches: {str(e)}")
                # Add fallback entry for failed processing
                fallback_entry = create_fallback_clinical_entry(target_molecule, "Clinical Trials/EUCTR")
                fallback_entry["_processing_error"] = True
                fallback_entry["_error_message"] = str(e)
                batch_results.append(fallback_entry)
        
        # Process PubMed data in smaller batches
        if pubmed_scraped_data:
            pubmed_batches = split_pubmed_data(pubmed_scraped_data, batch_size=15)
            print(f"Split pubmed data into {len(pubmed_batches)} batches")
            
            for i, pubmed_batch in enumerate(pubmed_batches):
                print(f"Processing PubMed batch {i+1}/{len(pubmed_batches)}")
                
                try:
                    batch_result = process_pubmed_batch(target_molecule, pubmed_batch)
                    batch_results.extend(batch_result)
                    print(f"PubMed batch {i+1} processed successfully")
                except Exception as e:
                    print(f"Error processing PubMed batch {i+1}: {str(e)}")
                    # Add fallback entry for failed batch
                    fallback_entry = create_fallback_clinical_entry(target_molecule, "PubMed")
                    fallback_entry["_processing_error"] = True
                    fallback_entry["_batch"] = i+1
                    fallback_entry["_error_message"] = str(e)
                    batch_results.append(fallback_entry)
                    continue

        # Process Google Patent Abstracts data in smaller batches
        if google_patent_abstracts:
            patent_abstracts_batches = split_patent_abstracts_data(google_patent_abstracts, batch_size=10)
            print(f"Split google patent abstracts data into {len(patent_abstracts_batches)} batches")
            
            for i, patent_abstracts_batch in enumerate(patent_abstracts_batches):
                print(f"Processing Google Patent Abstracts batch {i+1}/{len(patent_abstracts_batches)}")
                try:
                    batch_result = process_patent_abstracts_batch(target_molecule, patent_abstracts_batch)
                    batch_results.extend(batch_result)
                    print(f"Google Patent Abstracts batch {i+1} processed successfully")
                except Exception as e:
                    print(f"Error processing Google Patent Abstracts batch {i+1}: {str(e)}")
                    # Add fallback entry for failed batch
                    fallback_entry = create_fallback_clinical_entry(target_molecule, "Patent Abstracts")
                    fallback_entry["_processing_error"] = True
                    fallback_entry["_batch"] = i+1
                    fallback_entry["_error_message"] = str(e)
                    batch_results.append(fallback_entry)
                    continue
        
        # Combine results with enhanced error handling
        combined_results = combine_results(batch_results)
        
        # Ensure we have at least one result
        if not combined_results:
            print("No results found, creating fallback entry")
            combined_results = [create_fallback_clinical_entry(target_molecule, "No Data")]
        
        final_result = json.dumps(combined_results, indent=2)
        
        print(f"tool ran successfully - processed {len(combined_results)} total assets")
        
        # Save with safe JSON dump
        output_file = "output/analyzed_data.json"
        safe_json_dump(combined_results, output_file, "Analyzed Scraped Data")
        
        return {
            **state,
            "extracted_info": combined_results
        }
        
    except Exception as e:
        print("error analyzing data:",str(e))
        # Return fallback state with error info
        fallback_result = [create_fallback_clinical_entry(state.get("target", "Unknown"), "Critical Error")]
        fallback_result[0]["_critical_error"] = True
        fallback_result[0]["_error_message"] = str(e)
        
        return {
            **state,
            "extracted_info": fallback_result,
            "message": "error analyzing scraped data",
            "error": str(e)
        }


def create_fallback_clinical_entry(target_molecule: str, source: str) -> dict:
    """Create a fallback clinical entry when processing fails."""
    return {
        "drug_name": "Unknown",
        "phase": "Preclinical",
        "status": "Unknown",
        "modality": "Unknown",
        "sponsor": "Unknown",
        "indication": "Unknown",
        "mechanism_of_action": "Unknown",
        "acquisition/licensing signals": "",
        "_fallback_entry": True,
        "_target": target_molecule,
        "_source": source
    }

def format(data):
    if isinstance(data, str):
        data = data.strip()
        if data.startswith("```json"):
            data = data[len("```json"):].lstrip()
        if data.endswith("```"):
            data = data[:-3].rstrip()
    return data