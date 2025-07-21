# Competitive Landscape Analysis Tool

An AI-powered system for analyzing competitive landscapes in drug development, focusing on specific molecular targets. This tool automates the collection, analysis, and scoring of therapeutic assets to provide comprehensive competitive intelligence for pharmaceutical research and development.

## Overview

This system takes a molecular target (e.g., CD47, KRAS, PD-1) as input and generates a detailed competitive analysis by:
- You can provide your input as a `.txt`, `.csv`, or `.json` file. The tool supports all three formats for specifying your list of molecular targets.  **For `.csv` and `.json` files, make sure your file includes a `target` or `molecular_target` field for each entry.** and for the .txt file simple pass the name of molecular targets line by line
- Scraping data from multiple authoritative sources
- Analyzing and extracting relevant drug development information
- Normalizing data across different formats and terminologies
- Calculating competitive scores and identifying strategic opportunities

## Test the platform using streamlit ui:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_ui.py
```
For more commands such as file input, please check the setup instructions mentioned below

## Methodology

### 1. Agent-Based Architecture
The system uses **LangGraph** to orchestrate a multi-step workflow through specialized AI agents:

```
Target Input → Scraper → Analyzer → Normalizer → Scorer → Final Report
```

### 2. Four-Stage Processing Pipeline

#### Stage 1: Data Scraping
For Data scraping, we are using Four primary sources:
- **ClinicalTrials.gov**: US and international clinical trials
- **EUCTR**: European Union clinical trials register
- **PubMed**: Biomedical literature for drug development publications
- **Google Patents**: Extracts patent publication numbers from the initial patent query results, then scrapes abstracts and metadata for each patent using a custom extraction tool. This enables the system to analyze the intellectual property landscape alongside clinical and scientific data.

> **Note:** We do not use any third-party scraping services like Firecrawl or GPT-based scrapers. Instead, we have identified and reverse-engineered the APIs ourselves by inspecting network activity (e.g., using browser network tabs). We handle the different types of responses from each source appropriately, and finally convert all collected data into a unified JSON format for downstream processing.


The project collects a lot of data from PubMed, including abstracts, titles, authors, and publication dates. Then, we filter the data, separate out all the abstracts, and send them to the AI for analysis.

For ClinicalTrials.gov and EUCTR, we use their respective APIs to fetch structured data. For PubMed, we utilize the `pymed_paperscraper` library to access abstracts and metadata.

> **Note:** There may be fluctuations in the scraped data returned from external sources. This can lead to slight changes in the number of identified assets, which in turn may cause minor variations in the calculated competitive score each time the analysis is run.


#### Stage 2: Data Analysis
- We use OpenAI's GPT models to analyze the scraped data.
- Because AI models like GPT have a limit on how much text they can process at once, the data is split into smaller groups (batches) of 25 articles.
- Each batch is sent to the AI, which reads and analyzes the articles.
- The AI filters out studies that aren’t about therapies (like those only about diagnostics or biomarkers).
- For the relevant data, the AI pulls out important details such as drug name, how the drug works (modality and mechanism of action), sponsor, what disease it treats (indication), its status, and information about licensing or acquisitions.
The result is a structured, easy-to-use summary of drug information.

#### Stage 3: Data Normalization
- Extracts the list of assets (therapeutic candidates) from the input data
- Unifies drug names, recognizing aliases and alternative spellings (e.g., "ALX148" → "Evorpacept (ALX148)")
- Standardizes clinical trial phases (e.g., "Phase I/II" becomes "Phase II", missing values become "Preclinical")
- Normalizes mechanism of action (MoA) into a set of standard categories, or keeps the original if it doesn't match
- Standardizes modality (e.g., "mAb", "Bispecific mAb", "Fc-fusion Protein"), or keeps the original if unrecognized
- Groups the output by clinical phase and removes exact duplicate entries

#### Stage 4: Competitive Scoring
- **Weighted Scoring Algorithm**: Assigns different weights to clinical phases
- **Crowding Score Calculation**: Produces 0-1 scale competitive intensity score
- **White Space Analysis**: Identifies underexplored therapeutic opportunities

## Sources Used

### Primary Data Sources

1. **ClinicalTrials.gov (US)**
   - **Coverage**: US and international clinical trials
   - **API Endpoint**: `https://clinicaltrials.gov/api/int/studies`
   - **Data Fields**: 30+ fields including phases, status, interventions, sponsors
   - **Search Strategy**: Condition-based queries with spell-checking enabled
   - **Limits**: 100 studies per query (expandable)

2. **EUCTR (European Union Clinical Trials Register)**
   - **Coverage**: EU clinical trials
   - **API Endpoint**: `https://www.clinicaltrialsregister.eu/ctr-search/rest/download/summary`
   - **Format**: JSON summary downloads
   - **Scope**: European regulatory submissions

3. **PubMed (Biomedical Literature)**
   - **Coverage**: Global biomedical research publications
   - **Library**: pymed_paperscraper for API access
   - **Data**: Abstracts, titles, authors, publication dates
   - **Limits**: 500 articles per query (configurable)
   - **Processing**: Batch processing for large result sets

4. **Google Patents**
   - **Coverage**: Global database of patents, including biomedical, pharmaceutical, and biotech innovations
   - **Access Method**: Custom scraper or API to extract metadata and full-text when available


### Data Quality Assurance
- **Cross-Validation**: Information verified across multiple sources
- **Recency Filters**: Prioritizes recent publications and trials
- **Relevance Scoring**: AI-powered relevance assessment
- **Duplicate Removal**: Automated deduplication across sources

## Scoring Explanation

### Crowding Score Algorithm

The competitive crowding score is calculated using a **weighted phase-based approach**:

```python
Crowding Score = (Σ(Phase_Weight × Asset_Count)) / Maximum_Possible_Score
```

#### Phase Weights
- **Preclinical**: 0.2 (early-stage, high uncertainty)
- **Phase I**: 0.4 (safety established, moderate commitment)
- **Phase II**: 0.6 (efficacy signals, significant investment)
- **Phase III**: 0.8 (late-stage, high probability of success)
- **Approved**: 1.0 (market-validated, maximum competitive impact)

#### Score Interpretation
- **0.0 - 0.3**: Low competition (Blue Ocean)
- **0.3 - 0.6**: Moderate competition (Strategic opportunities exist)
- **0.6 - 0.8**: High competition (Crowded field)
- **0.8 - 1.0**: Hyper-competitive (Saturated market)

### White Space Analysis

The system identifies strategic opportunities through:

1. **Modality Gaps**: Missing therapeutic approaches (e.g., no small molecules in advanced phases)
2. **Indication Gaps**: Underserved disease areas
3. **Mechanism Gaps**: Novel or underexplored mechanisms of action
4. **Phase Gaps**: Stages with few competitors
5. **Geographic Gaps**: Regional development disparities

### Notable Metrics
- **Total Competitors**: Unique drug assets identified
- **Phase Distribution**: Asset count by development stage
- **Modality Diversity**: Range of therapeutic approaches
- **Acquisition Signals**: M&A activity indicators

## Assumptions + Generalization Logic

### Core Assumptions

#### 1. Data Source Completeness
- **Assumption**: ClinicalTrials.gov, EUCTR, and PubMed collectively capture >90% of relevant drug development activity
- **Limitation**: Some proprietary or early-stage programs may not be publicly disclosed
- **Mitigation**: Cross-referencing multiple sources to minimize gaps

#### 2. AI Analysis Accuracy
- **Assumption**: OpenAI GPT models can accurately identify and extract drug development information
- **Validation**: Structured prompts with explicit criteria and examples
- **Error Handling**: Multiple validation passes and format verification

#### 3. Competitive Weight Assignment
- **Assumption**: Later-stage assets represent stronger competitive threats
- **Rationale**: Higher investment, lower failure rates, greater market impact
- **Customization**: Weights can be adjusted for specific therapeutic areas

#### 4. Therapeutic Equivalence
- **Assumption**: Assets targeting the same molecule compete directly
- **Nuance**: System accounts for different indications and mechanisms
- **Differentiation**: Modality and mechanism analysis identifies competitive positioning

### Generalization Logic

#### 1. Cross-Target Applicability
- **Design**: System architecture is target-agnostic
- **Scalability**: Can analyze any molecular target with sufficient data
- **Adaptability**: Scoring methodology applies across therapeutic areas

#### 2. Temporal Consistency
- **Assumption**: Competitive dynamics remain relatively stable in 6-12 month timeframes
- **Refresh Strategy**: Designed for periodic re-analysis
- **Trend Analysis**: Historical data comparison capabilities

#### 3. Market Extrapolation
- **Assumption**: Clinical development activity predicts market competition
- **Validation**: Includes approved products and acquisition activity
- **Business Intelligence**: Incorporates commercial signals beyond clinical data

### Known Limitations

1. **Private Company Bias**: Publicly traded companies more likely to disclose information
2. **Geographic Bias**: English-language and Western database emphasis
3. **Temporal Lag**: Publication delays may miss recent developments
4. **Indication Granularity**: Broad disease categories may miss subspecialty competition
5. **Mechanism Complexity**: AI may oversimplify complex mechanisms of action

### Enhancement Opportunities

1. **Real-Time Updates**: Integration with news APIs and SEC filings
2. **Competitive Intelligence**: Patent landscape analysis
3. **Market Sizing**: Integration with epidemiological data
4. **Predictive Modeling**: Machine learning for development success prediction
5. **Expert Validation**: Human expert review workflows

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

### Running the Analysis with Multiple Targets from a File

You can provide a list of molecular targets in a `.txt`, `.csv`, or `.json` file (one target per line or entry). To run the analysis on all targets in the file, use:
```bash
python main.py path/to/your_targets.txt
# or
python main.py path/to/your_targets.csv
# or
python main.py path/to/your_targets.json


```

### Running the Analysis with a single target
```bash
python main.py 
```

Enter your target molecule when prompted (e.g., "CD47", "KRAS", "PD-1").

### Output Files
- `output/analyzed_data.json`: Raw extracted drug information
- `output/normalized_data.json`: Standardized and cleaned data
- `output/competitive_analysis.json`: Final competitive analysis with scores
- `output/scraper-results/`: Raw data from each source

## Technical Architecture

### Dependencies
- **LangGraph**: Agent orchestration and workflow management
- **LangChain**: AI model integration and prompt management
- **OpenAI API**: GPT models for data analysis
- **Requests**: HTTP client for API calls
- **PyMed**: PubMed API integration

### File Structure
```
competitive_landscape/
├── agent/                # AI agent system
│   ├── workflow.py       # Main orchestration logic
│   ├── schema.py         # Data models
│   └── tools/            # Individual processing tools
├── utils/                # Input/output helpers and utilities and tokenising logic
├── scraper/              # Data collection modules
├── output/               # Analysis results
└── main.py               # Entry point
```
---

*This tool is designed for competitive intelligence and research purposes. Results should be validated with additional sources and expert analysis before making strategic decisions.*
