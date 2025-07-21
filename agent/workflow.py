from langgraph.graph import StateGraph, END
from agent.nodes.analyze_scrapped_data import analyze_scrapped_data
from agent.nodes.scraper_tool import scraper_tool
from agent.schema import Analysis
from agent.nodes.calculate_competitive_score import calculate_competitive_score_and_white_space_flags
from agent.nodes.normalize_data import normalize_data
from agent.nodes.analyze_patents_query import analyze_patents_query

def build_analyze_landscape_flow():
    builder = StateGraph(Analysis)
    builder.add_node("analyze_patents_query",analyze_patents_query)
    builder.add_node("scrape",scraper_tool)
    builder.add_node("analyze_scraped_data",analyze_scrapped_data)
    builder.add_node("normalize_data",normalize_data)
    builder.add_node("calculate_competitive_score_and_white_space_flags",calculate_competitive_score_and_white_space_flags)
    
    builder.set_entry_point("analyze_patents_query")
    builder.add_edge("analyze_patents_query","scrape")
    builder.add_edge("scrape","analyze_scraped_data")
    builder.add_edge("analyze_scraped_data","normalize_data")
    builder.add_edge("normalize_data","calculate_competitive_score_and_white_space_flags")
    
    return builder.compile()


def competitive_score(target: str) -> dict:
    
    graph = build_analyze_landscape_flow()
    inital_state = {
        "target":target,
        "clinical_trials_scraped_data":None,
        "google_patent_abstracts":None,
        "euctr_scraped_data":None,
        "pubmed_scraped_data":None,
        "extracted_info":None,
        "normalized_data":None,
        "competitive_score": None
    }
    
    result = graph.invoke(inital_state)
    return result