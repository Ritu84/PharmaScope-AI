import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from utils.tokenize import count_tokens, check_and_split_json_input, graceful_json_parse, extract_valid_entries, safe_json_dump
import json
import re

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

messages = [
    ("system", 
    "You are a biotech analyst assistant. You analyze normalized drug development data to assess competition levels in a given molecular target landscape. "
    "Your job is to count the number of drug assets in each development phase, compute a competition (crowding) score, and identify any whitespace or strategic differentiation opportunities. "
    "You output structured JSON and provide an explanation of your methodology. A high crowding score means the target space is highly saturated with competitors; a low score means it is relatively empty.make sure to maintain the consistency of data between the number of 'total_competitors' and the sum of the number of assets in each phase. Finally No hallucinated scoring text:If scoring input is empty or invalid, don't return made-up methodology text. Fail explicitly and mark the target as insufficient data."
    ),

    ("user", 
    """Here is the normalized data for all therapeutic drug assets targeting the molecule '{target}'.
1. Count the number of unique assets in each of the following clinical phases: "Preclinical", "Phase I", "Phase II", "Phase III", "Approved".

2. Calculate a "crowding_score" between 0 and 1.
   - A score of 0 means no assets exist (no competition).
   - A score of 1 means many assets across advanced stages (hyper-saturation).
   - Use a scoring function that considers both the **number** of assets and their **development stages**.
   - Explain your scoring function in plain English.

3. Output "white_space_flags" if applicable — flag areas of strategic opportunity based on:
   - Missing or rare **modalities** (e.g., only monoclonal antibodies, no small molecules or ADCs)
   - First-in-class mechanisms (if any asset uses a mechanism not seen in others)
   - Unserved indications (diseases where no CD47 asset is being tested)

Return your results in this JSON format:
{{
  "target": "{{target}}",
  "crowding_score": 0.78,
  "total_competitors": 42,
  "phase_distribution": {{
    "Preclinical": 10,
    "Phase I": 14,
    "Phase II": 9,
    "Phase III": 6,
    "Approved": 3
  }},
  "modalities": ["Monoclonal Antibody", "Bispecific Antibody", "ADC"],
  "notable_acquisitions": ["Gilead acquired Forty Seven Inc. for $4.9B"],
  "white_space_flags": [
    "No bispecifics in Phase III",
    "No assets for pediatric AML",
    "Only monoclonal antibodies used — no small molecules or fusion proteins"
  ],
  "scoring_methodology": "The crowding score is calculated by assigning weights to phases: Preclinical = 0.2, Phase I = 0.4, Phase II = 0.6, Phase III = 0.8, Approved = 1.0. The weighted sum is normalized over a maximum possible saturation to produce a score between 0 and 1."
}}

Input Data:
{normalized_data}
"""
    )
]

def calculate_competitive_score_and_white_space_flags(state : dict) -> dict:
    try: 
        print("running competitive_analysis tool...")
        target_molecule = state.get("target")
        normalized_data = state.get("normalized_data")
        
        if not target_molecule:
            target_molecule = "Unknown"
            print("Warning: Missing target in state, using 'Unknown'")
        
        if not normalized_data:
            print("No normalized_data found, returning fallback competitive analysis")
            fallback_result = create_fallback_competitive_analysis(target_molecule)
            return {
                **state,
                "competitive_analysis": fallback_result
            }
        
        model_name = OPENAI_MODEL or "gpt-4o"  
        max_tokens = 8000  
        
        input_token_count = count_tokens(str(normalized_data), model_name)
        print(f"Input token count: {input_token_count}")
        
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        chain = prompt | llm | StrOutputParser()
        
        if input_token_count > max_tokens:
            print(f"Input exceeds token limit ({input_token_count} > {max_tokens}). Splitting into chunks...")
            
            chunks = check_and_split_json_input(normalized_data, max_tokens, model_name)
            print(f"Split input into {len(chunks)} chunks")
            
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                
                try:
                    chunk_result = chain.invoke({
                        "target": target_molecule,
                        "normalized_data": chunk,
                        "format": "json"
                    })
                    
                    fallback_template = create_fallback_competitive_analysis(target_molecule)
                    
                    parsed_chunk = graceful_json_parse(
                        chunk_result, 
                        fallback_template, 
                        f"Competitive Analysis Chunk {i+1}"
                    )
                    
                    chunk_results.append(parsed_chunk)
                    print(f"Successfully processed chunk {i+1}")
                    
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    fallback_chunk = create_fallback_competitive_analysis(target_molecule)
                    fallback_chunk["_processing_error"] = True
                    fallback_chunk["_chunk"] = i+1
                    fallback_chunk["_error_message"] = str(e)
                    chunk_results.append(fallback_chunk)
                    continue
            
            final_result = combine_competitive_analysis_results(chunk_results, target_molecule)
            print(f"Combined results from {len(chunks)} chunks")
            
        else:
            print("Input is within token limit. Processing normally...")
            
            try:
                result = chain.invoke({
                    "target": target_molecule,
                    "normalized_data": normalized_data,
                    "format": "json"
                })
                
                fallback_template = create_fallback_competitive_analysis(target_molecule)
                
                parsed_result = graceful_json_parse(result, fallback_template, "Competitive Analysis")
                
                if isinstance(parsed_result, dict):
                    final_result = validate_competitive_analysis_structure(parsed_result, target_molecule)
                else:
                    print("Warning: Result is not a dictionary, using fallback")
                    final_result = fallback_template
                
                print(f"Successfully processed competitive analysis")
                
            except Exception as e:
                print(f"Error processing competitive analysis: {str(e)}")
                final_result = create_fallback_competitive_analysis(target_molecule)
                final_result["_processing_error"] = True
                final_result["_error_message"] = str(e)

       
        safe_target = re.sub(r'[^A-Za-z0-9_\-]', '_', str(target_molecule))
        output_file = f"output/{safe_target}_competitive_analysis.json"
        safe_json_dump(final_result, output_file, "Competitive Analysis")

        return {
            **state,
            "competitive_analysis": final_result
        }
        
    except Exception as e:
        print("err:",str(e))
        # Return fallback state with error info
        fallback_result = create_fallback_competitive_analysis(state.get("target", "Unknown"))
        fallback_result["_critical_error"] = True
        fallback_result["_error_message"] = str(e)
        
        return {
            **state,
            "competitive_analysis": fallback_result,
            "message": "error analyzing competitive score and whitelist flags",
            "error": str(e)
        }


def create_fallback_competitive_analysis(target_molecule: str) -> dict:
    """Create a fallback competitive analysis structure when processing fails."""
    return {
        "target": target_molecule,
        "crowding_score": 0.0,
        "total_competitors": 0,
        "phase_distribution": {
            "Preclinical": 0,
            "Phase I": 0,
            "Phase II": 0,
            "Phase III": 0,
            "Approved": 0
        },
        "modalities": [],
        "notable_acquisitions": [],
        "white_space_flags": ["Insufficient data to analyze competitive landscape"],
        "scoring_methodology": "Could not calculate score due to insufficient or invalid data"
    }


def validate_competitive_analysis_structure(data: dict, target_molecule: str) -> dict:
    """Validate and clean the competitive analysis structure."""
    # Ensure required fields exist with proper defaults
    result = {
        "target": data.get("target", target_molecule),
        "crowding_score": float(data.get("crowding_score", 0.0)),
        "total_competitors": int(data.get("total_competitors", 0)),
        "phase_distribution": {
            "Preclinical": int(data.get("phase_distribution", {}).get("Preclinical", 0)),
            "Phase I": int(data.get("phase_distribution", {}).get("Phase I", 0)),
            "Phase II": int(data.get("phase_distribution", {}).get("Phase II", 0)),
            "Phase III": int(data.get("phase_distribution", {}).get("Phase III", 0)),
            "Approved": int(data.get("phase_distribution", {}).get("Approved", 0))
        },
        "modalities": data.get("modalities", []) if isinstance(data.get("modalities"), list) else [],
        "notable_acquisitions": data.get("notable_acquisitions", []) if isinstance(data.get("notable_acquisitions"), list) else [],
        "white_space_flags": data.get("white_space_flags", []) if isinstance(data.get("white_space_flags"), list) else [],
        "scoring_methodology": data.get("scoring_methodology", "Unknown methodology")
    }
    
    # Validate crowding score is between 0 and 1
    if result["crowding_score"] < 0 or result["crowding_score"] > 1:
        result["crowding_score"] = 0.0
    
    # Ensure total competitors matches sum of phase distribution
    calculated_total = sum(result["phase_distribution"].values())
    if result["total_competitors"] != calculated_total:
        result["total_competitors"] = calculated_total
        result["_total_competitors_corrected"] = True
    
    return result


def combine_competitive_analysis_results(chunk_results, target_molecule):
    """
    Combine competitive analysis results from multiple chunks.
    Merges phase distributions, recalculates crowding score, and combines insights.
    """
    combined_phase_distribution = {
        "Preclinical": 0,
        "Phase I": 0,
        "Phase II": 0,
        "Phase III": 0,
        "Approved": 0
    }
    
    all_modalities = set()
    all_acquisitions = []
    all_white_space_flags = []
    
    valid_chunks = [chunk for chunk in chunk_results if isinstance(chunk, dict) and "error" not in chunk]
    
    if not valid_chunks:
        fallback_result = create_fallback_competitive_analysis(target_molecule)
        fallback_result["_combine_error"] = True
        return fallback_result
    
    # Combine phase distributions
    for chunk in valid_chunks:
        if "phase_distribution" in chunk and isinstance(chunk["phase_distribution"], dict):
            phase_dist = chunk["phase_distribution"]
            for phase, count in phase_dist.items():
                if phase in combined_phase_distribution:
                    try:
                        combined_phase_distribution[phase] += int(count)
                    except (ValueError, TypeError):
                        # Skip invalid count values
                        pass
        
        # Collect modalities
        if "modalities" in chunk and isinstance(chunk["modalities"], list):
            all_modalities.update(chunk["modalities"])
        
        # Collect acquisitions
        if "notable_acquisitions" in chunk and isinstance(chunk["notable_acquisitions"], list):
            all_acquisitions.extend(chunk["notable_acquisitions"])
        
        # Collect white space flags
        if "white_space_flags" in chunk and isinstance(chunk["white_space_flags"], list):
            all_white_space_flags.extend(chunk["white_space_flags"])
    
    # Calculate total competitors
    total_competitors = sum(combined_phase_distribution.values())
    
    # Calculate crowding score using weighted approach
    phase_weights = {
        "Preclinical": 0.2,
        "Phase I": 0.4,
        "Phase II": 0.6,
        "Phase III": 0.8,
        "Approved": 1.0
    }
    
    weighted_sum = sum(combined_phase_distribution[phase] * phase_weights[phase] 
                      for phase in combined_phase_distribution)
    
    # Normalize the score (assuming max saturation of 50 assets as reference)
    max_saturation = 50
    crowding_score = min(weighted_sum / max_saturation, 1.0)
    
    # Remove duplicate entries
    unique_modalities = list(set(all_modalities))
    unique_acquisitions = list(set(all_acquisitions))
    unique_white_space_flags = list(set(all_white_space_flags))
    
    return {
        "target": target_molecule,
        "crowding_score": round(crowding_score, 3),
        "total_competitors": total_competitors,
        "phase_distribution": combined_phase_distribution,
        "modalities": unique_modalities,
        "notable_acquisitions": unique_acquisitions,
        "white_space_flags": unique_white_space_flags,
        "scoring_methodology": f"The crowding score is calculated by assigning weights to phases: Preclinical = 0.2, Phase I = 0.4, Phase II = 0.6, Phase III = 0.8, Approved = 1.0. The weighted sum ({weighted_sum:.2f}) is normalized over a maximum saturation of {max_saturation} to produce a score between 0 and 1. Combined from {len(valid_chunks)} data chunks."
    }


def format(data):
    """
    Cleans up the output string by removing code block markers and any appended
    plain-text explanations or methodology sections after the JSON object.
    """
    if isinstance(data, str):
        data = data.strip()
        # Remove code block markers
        if data.startswith("```json"):
            data = data[len("```json"):].lstrip()
        if data.startswith("```"):
            data = data[len("```"):].lstrip()
        if data.endswith("```"):
            data = data[:-3].rstrip()
        last_brace = data.rfind("}")
        if last_brace != -1:
            data = data[:last_brace+1]
    return data
    