# Competitive Landscape Analysis UI

A simple Streamlit web interface for analyzing competitive landscapes of molecular targets.

## Features

- **Dual Input Methods**: 
  - Single target analysis: Enter any molecular target (e.g., CD47, EGFR, VEGF)
  - Bulk file upload: Upload CSV, TXT, or JSON files with multiple targets
- **Comprehensive Analysis**: Get detailed competitive landscape analysis including:
  - Crowding score (0-1 scale)
  - Total competitors count
  - Phase distribution (Preclinical, Phase I, II, III, Approved)
  - Modalities information
  - Notable acquisitions
  - White space opportunities
  - Scoring methodology
- **Multiple File Formats**: 
  - **CSV**: Column named 'target' or 'molecular_target'
  - **TXT**: One target per line (or comma-separated)
  - **JSON**: Array of objects with 'target' or 'molecular_target' keys
- **Clean Design**: Minimalist white and black interface
- **Raw Data Access**: Expandable JSON output for detailed analysis

## How to Run

### Option 1: Using the run script
```bash
python run_ui.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run ui/streamlit_app.py
```

## Usage

1. Start the application using one of the methods above
2. Open your browser to `http://localhost:8001`
3. Choose your analysis method:

### Single Target Analysis
- Enter a molecular target in the input field
- Click "Analyze Single Target"
- Review the analysis results

### Bulk File Analysis
- Click on the "File Upload" tab
- Upload a CSV, TXT, or JSON file with molecular targets
- Click "Analyze All Targets from File"
- Review results for all targets sequentially

### File Format Examples

**CSV Format:**
```csv
target,indication
CD47,Cancer
EGFR,Oncology
VEGF,Angiogenesis
```

**TXT Format:**
```
CD47
EGFR
VEGF
```

**JSON Format:**
```json
[
  {"target": "CD47", "indication": "Cancer"},
  {"molecular_target": "EGFR"},
  "VEGF"
]
```

## Sample Files

Sample files are provided in the `ui/sample_files/` directory for testing:
- `sample_targets.csv` - CSV format example
- `sample_targets.txt` - TXT format example  
- `sample_targets.json` - JSON format example

## Requirements

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

## Output Structure

The analysis provides:
- **Target**: The molecular target analyzed
- **Crowding Score**: Competition level (0 = no competition, 1 = highly saturated)
- **Total Competitors**: Number of competing assets
- **Phase Distribution**: Breakdown by development phase
- **Modalities**: Types of therapeutic approaches
- **Notable Acquisitions**: M&A activity in the space
- **White Space Flags**: Strategic opportunities
- **Scoring Methodology**: How the crowding score was calculated

## Notes

- Analysis is based on publicly available data
- Results may not reflect the complete competitive landscape
- Processing time varies depending on the target complexity 