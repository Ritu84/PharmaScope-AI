from typing import TypedDict, Optional
from pydantic import BaseModel

class Analysis(TypedDict):
    target:Optional[str]
    patents_query_result:Optional[str]
    google_patent_abstracts:Optional[str]
    clinical_trials_scraped_data:Optional[str]
    euctr_scraped_data:Optional[str]
    pubmed_scraped_data:Optional[str]
    extracted_info:Optional[str]
    competitive_analysis:Optional[str]
    normalized_data:Optional[str]
    
class TargetMetadata(BaseModel):
    molecule_name: str
    variant: Optional[str] = None
    modality: Optional[str] = None
    mechanism: Optional[str] = None
    therapeutic_area: Optional[str] = None
    constraints: Optional[str] = None