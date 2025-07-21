import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from utils.tokenize import count_tokens, check_and_split_json_input, graceful_json_parse, extract_valid_entries, safe_json_dump
import json

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.1
)

messages = [
    {
        "role": "system",
        "content": (
            """You are an expert biomedical data analyst specializing in drug development pipelines. 
            Your task is to normalize inconsistent data entries from various clinical trial databases. 
            You should:\n\n
            1. Unify drug names: Recognize aliases, suffixes, or partial names (e.g., ALX148 → Evorpacept (ALX148)).\n
            2. Standardize clinical phases:\n
                - Preclinical
                - Treat Phase I/II as Phase II\n
                - Treat Phase II/III as Phase III\n
                - Treat N/A or missing values as Preclinical\n
            3. Normalize MoA (Mechanism of Action) into:\n
                - CD47 Blockade\n
                - CD47/PD-1 Blockade\n
                - SIRPα-CD47 Blockade\n
                - If not matching, return the original string\n
            4. Standardize modality into:\n
                - mAb\n
                - Bispecific mAb\n
                - Fc-fusion Protein\n
                - Or keep original if unrecognized\n
            5. Group the output by `phase` and remove exact duplicate entries.\n\n
            6. And keep the rest of the fields(status, modality, sponsor, indication, mechanism_of_action, acquisition/licensing signals) as they are.
            Return the final result as a JSON object grouped by phase."""
        )
    },
    {
        "role": "user",
        "content": "Normalize the following clinical trial data:\n\n {extracted_info}" 
    }
]

def normalize_data(state : dict) -> dict:
    try: 
        print("running data normalizer tool...")
        extracted_info = state.get("extracted_info")
        
        if not extracted_info:
            print("No extracted_info found, returning fallback normalized data")
            fallback_result = create_fallback_normalized_data(state.get("target", "Unknown"))
            return {
                **state,
                "normalized_data": fallback_result
            }
        
        model_name = OPENAI_MODEL or "gpt-4o"  
        max_tokens = 8000  
        
        input_token_count = count_tokens(str(extracted_info), model_name)
        print(f"Input token count: {input_token_count}")
        
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        chain = prompt | llm | StrOutputParser()
        
        if input_token_count > max_tokens:
            print(f"Input exceeds token limit ({input_token_count} > {max_tokens}). Splitting into chunks...")
            
            chunks = check_and_split_json_input(extracted_info, max_tokens, model_name)
            print(f"Split input into {len(chunks)} chunks")
            
            all_results = {}
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                
                try:
                    chunk_result = chain.invoke({
                        "extracted_info": chunk,
                        "format": "json"
                    })
                    
                    fallback_template = create_fallback_normalized_data("Unknown")
                    
                    parsed_chunk = graceful_json_parse(
                        chunk_result, 
                        fallback_template, 
                        f"Normalize Data Chunk {i+1}"
                    )
                    
                    if isinstance(parsed_chunk, dict):
                        for phase, trials in parsed_chunk.items():
                            if phase.startswith("_"):  
                                continue
                            if phase not in all_results:
                                all_results[phase] = []
                            if isinstance(trials, list):
                                all_results[phase].extend(trials)
                            else:
                                all_results[phase].append(trials)
                    else:
                        print(f"Warning: Chunk {i+1} result is not a dictionary")
                       
                        if "Preclinical" not in all_results:
                            all_results["Preclinical"] = []
                        all_results["Preclinical"].append({
                            "drug_name": "Unknown",
                            "status": "Unknown",
                            "modality": "Unknown",
                            "sponsor": "Unknown",
                            "indication": "Unknown",
                            "mechanism_of_action": "Unknown",
                            "acquisition/licensing signals": "",
                            "_processing_error": True,
                            "_chunk": i+1
                        })
                    
                    print(f"Successfully processed chunk {i+1}")
                    
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    
                    if "Preclinical" not in all_results:
                        all_results["Preclinical"] = []
                    all_results["Preclinical"].append({
                        "drug_name": "Unknown",
                        "status": "Unknown",
                        "modality": "Unknown",
                        "sponsor": "Unknown",
                        "indication": "Unknown",
                        "mechanism_of_action": "Unknown",
                        "acquisition/licensing signals": "",
                        "_processing_error": True,
                        "_chunk": i+1,
                        "_error_message": str(e)
                    })
                    continue
            
            # Remove duplicates within each phase
            for phase in all_results:
                if isinstance(all_results[phase], list):
                    all_results[phase] = remove_duplicates_in_phase(all_results[phase])
            
            final_result = all_results
            print(f"Combined results from {len(chunks)} chunks into {len(final_result)} phases")
            
        else:
            print("Input is within token limit. Processing normally...")
            
            try:
                result = chain.invoke({
                    "extracted_info": extracted_info,
                    "format": "json"
                })
                
                # Use graceful JSON parsing with fallback template
                fallback_template = create_fallback_normalized_data(state.get("target", "Unknown"))
                
                parsed_result = graceful_json_parse(result, fallback_template, "Normalize Data")
                
                # Ensure we have a valid phase structure
                if isinstance(parsed_result, dict):
                    # Validate and clean the result
                    final_result = validate_normalized_structure(parsed_result)
                else:
                    print("Warning: Result is not a dictionary, using fallback")
                    final_result = fallback_template
                
                print(f"Successfully processed normalized data with {len(final_result)} phases")
                
            except Exception as e:
                print(f"Error processing normalize data: {str(e)}")
                # Return fallback result
                final_result = create_fallback_normalized_data(state.get("target", "Unknown"))
                final_result["_processing_error"] = True
                final_result["_error_message"] = str(e)
        
        # Save the final result with safe JSON dump
        output_file = "output/normalized_data.json"
        safe_json_dump(final_result, output_file, "Normalized Data")
        
        print("tool ran successfully")
        return {
            **state,
            "normalized_data": final_result
        }
        
    except Exception as e:
        print("error normalising data:",str(e))
        # Return fallback state with error info
        fallback_result = create_fallback_normalized_data(state.get("target", "Unknown"))
        fallback_result["_critical_error"] = True
        fallback_result["_error_message"] = str(e)
        
        return {
            **state,
            "normalized_data": fallback_result,
            "message": f"error normalizing data: {e}",
            "error": str(e)
        }


def create_fallback_normalized_data(target: str) -> dict:
    """Create a fallback normalized data structure when processing fails."""
    return {
        "Preclinical": [{
            "drug_name": "Unknown",
            "status": "Unknown",
            "modality": "Unknown",
            "sponsor": "Unknown",
            "indication": "Unknown",
            "mechanism_of_action": "Unknown",
            "acquisition/licensing signals": "",
            "_fallback_entry": True,
            "_target": target
        }],
        "Phase I": [],
        "Phase II": [],
        "Phase III": [],
        "Approved": []
    }


def validate_normalized_structure(data: dict) -> dict:
    """Validate and clean the normalized data structure."""
    valid_phases = ["Preclinical", "Phase I", "Phase II", "Phase III", "Approved"]
    result = {}
    
    for phase in valid_phases:
        if phase in data:
            phase_data = data[phase]
            if isinstance(phase_data, list):
                # Validate entries in the phase
                validated_entries = []
                for entry in phase_data:
                    if isinstance(entry, dict):
                        # Ensure required fields exist
                        validated_entry = {
                            "drug_name": entry.get("drug_name", "Unknown"),
                            "status": entry.get("status", "Unknown"),
                            "modality": entry.get("modality", "Unknown"),
                            "sponsor": entry.get("sponsor", "Unknown"),
                            "indication": entry.get("indication", "Unknown"),
                            "mechanism_of_action": entry.get("mechanism_of_action", "Unknown"),
                            "acquisition/licensing signals": entry.get("acquisition/licensing signals", ""),
                            "_validated": True
                        }
                        validated_entries.append(validated_entry)
                result[phase] = validated_entries
            else:
                result[phase] = []
        else:
            result[phase] = []
    
    return result


def remove_duplicates_in_phase(trials: list) -> list:
    """Remove exact duplicates within a phase."""
    seen = set()
    unique_trials = []
    
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        
        # Create a key for comparison (excluding metadata fields)
        key_fields = {k: v for k, v in trial.items() if not k.startswith("_")}
        trial_key = json.dumps(key_fields, sort_keys=True)
        
        if trial_key not in seen:
            seen.add(trial_key)
            unique_trials.append(trial)
    
    return unique_trials


def format(data):
    """
    Cleans up the output string by removing code block markers, any leading plain text,
    and any appended plain-text explanations or methodology sections after the JSON object.
    """
    if isinstance(data, str):
        data = data.strip()
        # Find the first '{' (start of JSON)
        json_start = data.find('{')
        if json_start != -1:
            data = data[json_start:]
        # Remove code block markers
        if data.startswith("```json"):
            data = data[len("```json"):].lstrip()
        if data.startswith("```"):
            data = data[len("```"):].lstrip()
        if data.endswith("```"):
            data = data[:-3].rstrip()
        # Truncate after the last closing brace
        last_brace = data.rfind("}")
        if last_brace != -1:
            data = data[:last_brace+1]
    return data