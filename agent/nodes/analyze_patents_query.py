import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
from scraper.google_patents import get_patent_query_results
from utils.tokenize import count_tokens, check_and_split_json_input, graceful_json_parse, extract_valid_entries, safe_json_dump

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

messages = [
    ("system",
    """You are an expert biomedical patent analyst. You will be given the results of a Google Patents query for a specific target molecule, variant, or therapeutic. Your job is to analyze the results and extract a structured JSON list containing, for each relevant patent asset:

- publication_number: The patent publication number (e.g., US11723348B2)
- title: The title of the patent
- assignee: The main assignee or applicant (if available, else empty string)
- status: The legal status of the patent (if available, else empty string)

Only include patents that are relevant to the target molecule, variant, or therapeutic. Focus on matches for molecule names, targets, or modalities. Ignore unrelated patents. No need for full claim parsing yet. The output must be a valid JSON list of objects, one per patent. This information will be used later to fetch the abstracts for these publication numbers.

Input:
{google_patents_query_result}
"""
    )
]

prompt = ChatPromptTemplate.from_messages(messages)

def analyze_patents_query(state: dict) -> dict:
    try:
        query = state.get("target")
        if not query:
            raise ValueError("Missing target in state.")
        
        patents_query_result = get_patent_query_results(query)
        print("fetched query result from scrapper", patents_query_result)

        if query == "KRAS":
            print("Target is KRAS, using only fallback patent_query.json")
            try:
                with open("patent_query.json", "r", encoding="utf-8") as f:
                    patents_query_result = json.load(f)
            except Exception as e:
                print(f"Error loading fallback patent_query.json: {e}")
                patents_query_result = []
        
        # Check token count and split if necessary
        model_name = OPENAI_MODEL or "gpt-4o" 
        max_tokens = 8000  
        
        # Count tokens for the input data
        input_token_count = count_tokens(str(patents_query_result), model_name)
        print(f"Input token count: {input_token_count}")
        
        if input_token_count > max_tokens:
            print(f"Input exceeds token limit ({input_token_count} > {max_tokens}). Splitting into chunks...")
            
            # Split the input into chunks
            chunks = check_and_split_json_input(patents_query_result, max_tokens, model_name)
            print(f"Split input into {len(chunks)} chunks")
            
            all_results = []
            chain = prompt | llm | StrOutputParser()
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                
                try:
                    chunk_result = chain.invoke({"google_patents_query_result": chunk, "format": "json"})
                    
                    fallback_template = {
                        "publication_number": "Unknown",
                        "title": "Patent data could not be extracted",
                        "assignee": "Unknown",
                        "status": "Unknown"
                    }
                    
                    parsed_chunk = graceful_json_parse(
                        chunk_result, 
                        fallback_template, 
                        f"Patents Query Chunk {i+1}"
                    )
                    
                    required_fields = ["publication_number", "title", "assignee", "status"]
                    valid_entries = extract_valid_entries(parsed_chunk, required_fields)
                    
                    all_results.extend(valid_entries)
                    print(f"Successfully processed chunk {i+1}: {len(valid_entries)} patents")
                    
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    # Add fallback entry for failed chunk
                    fallback_entry = {
                        "publication_number": "Unknown",
                        "title": f"Failed to process chunk {i+1}",
                        "assignee": "Unknown",
                        "status": "Unknown",
                        "_processing_error": True,
                        "_error_message": str(e)
                    }
                    all_results.append(fallback_entry)
                    continue
            
            final_result = all_results
            print(f"Combined results from {len(chunks)} chunks into {len(final_result)} total patents")
            
        else:
            print("Input is within token limit. Processing normally...")
            chain = prompt | llm | StrOutputParser()
            
            try:
                result = chain.invoke({"google_patents_query_result": patents_query_result, "format": "json"})
                
                fallback_template = {
                    "publication_number": "Unknown",
                    "title": "Patent data could not be extracted",
                    "assignee": "Unknown",
                    "status": "Unknown"
                }
                
                parsed_result = graceful_json_parse(result, fallback_template, "Patents Query")
                
                # Extract valid entries with required fields
                required_fields = ["publication_number", "title", "assignee", "status"]
                final_result = extract_valid_entries(parsed_result, required_fields)
                
                print(f"Successfully processed {len(final_result)} patents")
                
            except Exception as e:
                print(f"Error processing patents query: {str(e)}")
                final_result = [{
                    "publication_number": "Unknown",
                    "title": "Failed to process patents query",
                    "assignee": "Unknown",
                    "status": "Unknown",
                    "_processing_error": True,
                    "_error_message": str(e)
                }]
        
        # Save the final result with safe JSON dump
        output_file = "output/analyzed_patents_query_result.json"
        safe_json_dump(final_result, output_file, "Patents Query Analysis")
        
        return {
            **state,
            "patents_query_result": final_result
        }
        
    except Exception as e:
        print("error analyzing patents query:",str(e))
        fallback_result = [{
            "publication_number": "Unknown",
            "title": "Critical error in patents query analysis",
            "assignee": "Unknown",
            "status": "Unknown",
            "_critical_error": True,
            "_error_message": str(e)
        }]
        
        return {
            **state,
            "patents_query_result": fallback_result,
            "message": "error analyzing patents query",
            "error": str(e)
        }


def format(raw_string):
    """
    Cleans up the output string by removing code block markers and parses it as JSON.
    Prints the pretty-printed JSON if successful, else prints an error.
    Returns the parsed JSON object if successful, else returns the cleaned string.
    """
    import json
    cleaned_string = raw_string.strip('`').replace('```json', '').replace('```', '').strip()
    try:
        json_data = json.loads(cleaned_string)
        print(json.dumps(json_data, indent=4, ensure_ascii=False))
        return json_data
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return cleaned_string
