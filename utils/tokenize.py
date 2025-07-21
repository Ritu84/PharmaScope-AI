import tiktoken
from typing import List, Dict, Any, Union, Optional
import json
import re
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string for a specific model.
    
    Args:
        text (str): The text to count tokens for
        model (str): The model name to get the correct encoding for
        
    Returns:
        int: The number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def split_text_by_tokens(text: str, max_tokens: int = 6000, model: str = "gpt-3.5-turbo", 
                        overlap_tokens: int = 100) -> List[str]:
    """
    Split text into chunks based on token count to avoid context limit.
    
    Args:
        text (str): The text to split
        max_tokens (int): Maximum tokens per chunk (default 6000 to leave room for system prompt)
        model (str): The model name to get the correct encoding for
        overlap_tokens (int): Number of tokens to overlap between chunks for context
        
    Returns:
        List[str]: List of text chunks, each within the token limit
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # If text is already within limit, return as single chunk
    if count_tokens(text, model) <= max_tokens:
        return [text]
    
    # Split text into sentences first for better chunk boundaries
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence + '. '))
        
        # If adding this sentence would exceed the limit, start a new chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk if specified
            if overlap_tokens > 0 and chunks:
                overlap_text = get_overlap_text(current_chunk, overlap_tokens, encoding)
                current_chunk = overlap_text + sentence + '. '
                current_tokens = len(encoding.encode(current_chunk))
            else:
                current_chunk = sentence + '. '
                current_tokens = sentence_tokens
        else:
            current_chunk += sentence + '. '
            current_tokens += sentence_tokens
    
    # Add the final chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def get_overlap_text(text: str, overlap_tokens: int, encoding) -> str:
    """
    Get the last N tokens from text for overlapping chunks.
    
    Args:
        text (str): The text to get overlap from
        overlap_tokens (int): Number of tokens to overlap
        encoding: The tiktoken encoding object
        
    Returns:
        str: The overlapping text
    """
    tokens = encoding.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    
    overlap_token_slice = tokens[-overlap_tokens:]
    return encoding.decode(overlap_token_slice)


def check_and_split_json_input(data: Any, max_tokens: int = 6000, model: str = "gpt-3.5-turbo") -> List[str]:
    """
    Check if JSON data exceeds token limit and split it into chunks if needed.
    
    Args:
        data: The data to check (can be string, dict, list, etc.)
        max_tokens (int): Maximum tokens per chunk
        model (str): The model name to get the correct encoding for
        
    Returns:
        List[str]: List of JSON strings, each within the token limit
    """
    # Convert data to JSON string if it's not already a string
    if isinstance(data, str):
        json_str = data
    else:
        json_str = json.dumps(data, ensure_ascii=False)
    
    token_count = count_tokens(json_str, model)
    
    # If within limit, return as single chunk
    if token_count <= max_tokens:
        return [json_str]
    
    # For JSON data, we need to be more careful about splitting
    # Try to split by logical boundaries (e.g., array elements, object properties)
    try:
        parsed_data = json.loads(json_str) if isinstance(data, str) else data
        
        if isinstance(parsed_data, list):
            return split_json_array(parsed_data, max_tokens, model)
        elif isinstance(parsed_data, dict):
            return split_json_object(parsed_data, max_tokens, model)
        else:
            # For simple values, just split as text
            return split_text_by_tokens(json_str, max_tokens, model)
            
    except json.JSONDecodeError:
        # If JSON parsing fails, treat as regular text
        return split_text_by_tokens(json_str, max_tokens, model)


def split_json_array(arr: List[Any], max_tokens: int, model: str) -> List[str]:
    """Split a JSON array into chunks based on token limits."""
    chunks = []
    current_chunk = []
    current_tokens = 2  # Account for array brackets []
    
    for item in arr:
        item_json = json.dumps(item, ensure_ascii=False)
        item_tokens = count_tokens(item_json, model) + 1  # +1 for comma
        
        if current_tokens + item_tokens > max_tokens and current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = [item]
            current_tokens = 2 + item_tokens
        else:
            current_chunk.append(item)
            current_tokens += item_tokens
    
    if current_chunk:
        chunks.append(json.dumps(current_chunk, ensure_ascii=False))
    
    return chunks


def split_json_object(obj: Dict[str, Any], max_tokens: int, model: str) -> List[str]:
    """Split a JSON object into chunks based on token limits."""
    chunks = []
    current_chunk = {}
    current_tokens = 2  # Account for object braces {}
    
    for key, value in obj.items():
        item_json = json.dumps({key: value}, ensure_ascii=False)
        item_tokens = count_tokens(item_json, model) - 2  # -2 for braces we already counted
        
        if current_tokens + item_tokens > max_tokens and current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = {key: value}
            current_tokens = 2 + item_tokens
        else:
            current_chunk[key] = value
            current_tokens += item_tokens
    
    if current_chunk:
        chunks.append(json.dumps(current_chunk, ensure_ascii=False))
    
    return chunks


def graceful_json_parse(data: Union[str, dict, list], fallback_template: Optional[Dict[str, Any]] = None, 
                       context: str = "Unknown") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Gracefully parse JSON data with fallback handling for GPT failures.
    
    Args:
        data: The data to parse (string, dict, or list)
        fallback_template: Template for fallback response when parsing fails
        context: Context description for logging
        
    Returns:
        Parsed JSON data or fallback response
    """
    if data is None:
        logger.warning(f"No data provided for {context}")
        return fallback_template or {"error": "No data", "status": "Unknown"}
    
    # If already parsed, return as-is
    if isinstance(data, (dict, list)):
        return data
    
    # Clean the string data
    if isinstance(data, str):
        cleaned_data = clean_json_string(data)
        
        # Try multiple parsing strategies
        parsed_data = try_parse_json_strategies(cleaned_data, context)
        
        if parsed_data is not None:
            return parsed_data
        
        # If all parsing fails, return fallback
        logger.error(f"Failed to parse JSON for {context}. Raw data: {str(data)[:200]}...")
        return create_fallback_response(fallback_template, data, context)
    
    # For other types, create fallback
    logger.warning(f"Unexpected data type for {context}: {type(data)}")
    return create_fallback_response(fallback_template, data, context)


def clean_json_string(data: str) -> str:
    """Clean JSON string by removing common formatting issues."""
    data = data.strip()
    
    # Remove code block markers
    if data.startswith("```json"):
        data = data[len("```json"):].lstrip()
    if data.startswith("```"):
        data = data[len("```"):].lstrip()
    if data.endswith("```"):
        data = data[:-3].rstrip()
    
    # Remove common prefixes
    if data.startswith("json\n"):
        data = data.replace("json\n", "", 1).strip()
    
    return data


def try_parse_json_strategies(data: str, context: str) -> Optional[Union[Dict, List]]:
    """Try multiple strategies to parse JSON data."""
    
    # Strategy 1: Direct JSON parsing
    try:
        parsed = json.loads(data)
        logger.info(f"Successfully parsed JSON for {context} using direct parsing")
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from text using regex
    try:
        # Look for JSON array or object patterns
        json_match = re.search(r'\[.*\]|\{.*\}', data, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            logger.info(f"Successfully parsed JSON for {context} using regex extraction")
            return parsed
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Strategy 3: Try to find the last complete JSON object/array
    try:
        # Find last closing brace or bracket
        last_brace = data.rfind("}")
        last_bracket = data.rfind("]")
        
        if last_brace > last_bracket:
            # Try parsing up to the last brace
            json_candidate = data[:last_brace + 1]
            # Find the matching opening brace
            brace_count = 0
            start_pos = -1
            for i in range(len(json_candidate) - 1, -1, -1):
                if json_candidate[i] == '}':
                    brace_count += 1
                elif json_candidate[i] == '{':
                    brace_count -= 1
                    if brace_count == 0:
                        start_pos = i
                        break
            
            if start_pos != -1:
                json_str = json_candidate[start_pos:]
                parsed = json.loads(json_str)
                logger.info(f"Successfully parsed JSON for {context} using brace matching")
                return parsed
        
        elif last_bracket > last_brace:
            # Try parsing up to the last bracket
            json_candidate = data[:last_bracket + 1]
            # Find the matching opening bracket
            bracket_count = 0
            start_pos = -1
            for i in range(len(json_candidate) - 1, -1, -1):
                if json_candidate[i] == ']':
                    bracket_count += 1
                elif json_candidate[i] == '[':
                    bracket_count -= 1
                    if bracket_count == 0:
                        start_pos = i
                        break
            
            if start_pos != -1:
                json_str = json_candidate[start_pos:]
                parsed = json.loads(json_str)
                logger.info(f"Successfully parsed JSON for {context} using bracket matching")
                return parsed
        
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 4: Try to parse individual lines as JSON objects
    try:
        lines = data.split('\n')
        objects = []
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    objects.append(obj)
                except json.JSONDecodeError:
                    continue
        
        if objects:
            logger.info(f"Successfully parsed JSON for {context} using line-by-line parsing")
            return objects
    except Exception:
        pass
    
    logger.warning(f"All JSON parsing strategies failed for {context}")
    return None


def create_fallback_response(fallback_template: Optional[Dict[str, Any]], 
                           original_data: Any, context: str) -> Dict[str, Any]:
    """Create a fallback response when JSON parsing fails."""
    
    if fallback_template:
        # Use provided template and mark unknown fields
        fallback = fallback_template.copy()
        for key, value in fallback.items():
            if isinstance(value, str) and not value:
                fallback[key] = "Unknown"
            elif isinstance(value, list) and not value:
                fallback[key] = []
            elif isinstance(value, dict) and not value:
                fallback[key] = {}
        
        # Add metadata about the failure
        fallback["_parsing_error"] = True
        fallback["_context"] = context
        fallback["_original_data_preview"] = str(original_data)[:100] if original_data else "No data"
        
        return fallback
    
    # Default fallback structure
    return {
        "error": "JSON parsing failed",
        "status": "Unknown",
        "context": context,
        "parsing_error": True,
        "original_data_preview": str(original_data)[:100] if original_data else "No data"
    }


def extract_valid_entries(data: Union[List, Dict], required_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Extract valid entries from parsed data, filling missing fields with 'Unknown'.
    
    Args:
        data: Parsed JSON data (list or dict)
        required_fields: List of required field names
        
    Returns:
        List of validated entries with missing fields filled
    """
    if required_fields is None:
        required_fields = []
    
    entries = []
    
    # Convert single dict to list
    if isinstance(data, dict):
        data = [data]
    
    if not isinstance(data, list):
        logger.warning(f"Expected list or dict, got {type(data)}")
        return []
    
    for item in data:
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict item: {type(item)}")
            continue
        
        # Fill missing required fields
        validated_item = item.copy()
        for field in required_fields:
            if field not in validated_item or not validated_item[field]:
                validated_item[field] = "Unknown"
        
        # Mark as manually validated
        validated_item["_validated"] = True
        entries.append(validated_item)
    
    return entries


def safe_json_dump(data: Any, filepath: str, context: str = "Unknown") -> bool:
    """
    Safely dump data to JSON file with error handling.
    
    Args:
        data: Data to dump
        filepath: Path to output file
        context: Context for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JSON data for {context} to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON data for {context} to {filepath}: {str(e)}")
        return False