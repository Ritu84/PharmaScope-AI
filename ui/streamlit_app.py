import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflow import competitive_score
from utils.input import extract_targets_from_file
import json

st.set_page_config(
    page_title="Competitive Landscape Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#  CSS 
st.markdown("""
<style>
    .main {
        background-color: white;
        color: black;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
        color: black;
    }
    
    .stTextInput > div > div > input {
        background-color: white;
        color: black;
        border: 2px solid black;
    }
    
    .stButton > button {
        background-color: black;
        color: white;
        border: 2px solid black;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: white;
        color: black;
        border: 2px solid black;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border: 2px solid black;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .phase-distribution {
        background-color: #f8f9fa;
        padding: 15px;
        border: 1px solid black;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .white-space-flags {
        background-color: #f8f9fa;
        padding: 15px;
        border: 1px solid black;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .json-output {
        background-color: #f8f9fa;
        padding: 15px;
        border: 1px solid black;
        border-radius: 5px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Define the analysis functions
def display_analysis_results(analysis, target_name):
    """Display the competitive analysis results for a single target"""
    # Display results
    st.markdown("---")
    st.header(f"Analysis Results - {target_name}")
    
    # Key metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Target",
            value=analysis.get("target", "N/A")
        )
    
    with col2:
        st.metric(
            label="Crowding Score",
            value=f"{analysis.get('crowding_score', 0.0):.3f}"
        )
    
    with col3:
        st.metric(
            label="Total Competitors",
            value=analysis.get("total_competitors", 0)
        )
    
    # Phase distribution
    st.subheader("Phase Distribution")
    phase_dist = analysis.get("phase_distribution", {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Preclinical", phase_dist.get("Preclinical", 0))
    with col2:
        st.metric("Phase I", phase_dist.get("Phase I", 0))
    with col3:
        st.metric("Phase II", phase_dist.get("Phase II", 0))
    with col4:
        st.metric("Phase III", phase_dist.get("Phase III", 0))
    with col5:
        st.metric("Approved", phase_dist.get("Approved", 0))
    
    # Modalities
    st.subheader("Modalities")
    modalities = analysis.get("modalities", [])
    if modalities:
        for modality in modalities:
            st.write(f"• {modality}")
    else:
        st.write("No modalities found")
    
    # Notable acquisitions
    st.subheader("Notable Acquisitions")
    acquisitions = analysis.get("notable_acquisitions", [])
    if acquisitions:
        for acquisition in acquisitions:
            st.write(f"• {acquisition}")
    else:
        st.write("No notable acquisitions found")
    
    # White space flags
    st.subheader("White Space Opportunities")
    white_space = analysis.get("white_space_flags", [])
    if white_space:
        for flag in white_space:
            st.write(f"🔍 {flag}")
    else:
        st.write("No white space opportunities identified")
    
    # Scoring methodology
    st.subheader("Scoring Methodology")
    methodology = analysis.get("scoring_methodology", "No methodology available")
    st.write(methodology)
    
    # Raw JSON output (expandable)
    with st.expander("Raw JSON Output"):
        st.json(analysis)

def analyze_single_target(target):
    """Analyze a single molecular target"""
    with st.spinner(f"Analyzing competitive landscape for {target}..."):
        try:
            # Call the competitive_score function
            result = competitive_score(target)
            
            # Extract the competitive analysis from the result
            if "competitive_analysis" in result:
                analysis = result["competitive_analysis"]
                display_analysis_results(analysis, target)
            else:
                st.error("No competitive analysis found in the result")
                st.json(result)
                
        except Exception as e:
            st.error(f"Error analyzing competitive landscape: {str(e)}")
            st.write("Please check if all required dependencies are installed and the target is valid.")

def analyze_file_targets(uploaded_file):
    """Analyze multiple molecular targets from uploaded file"""
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract targets from the file
        with st.spinner("Extracting targets from file..."):
            targets = extract_targets_from_file(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if not targets:
            st.error("No targets found in the uploaded file. Please check the file format.")
            return
        
        st.success(f"Found {len(targets)} targets in the file: {', '.join(targets)}")
        
        # Process each target
        for i, target in enumerate(targets):
            st.markdown("---")
            st.subheader(f"Target {i+1}/{len(targets)}: {target}")
            
            with st.spinner(f"Analyzing {target}..."):
                try:
                    result = competitive_score(target)
                    
                    if "competitive_analysis" in result:
                        analysis = result["competitive_analysis"]
                        display_analysis_results(analysis, target)
                    else:
                        st.error(f"No competitive analysis found for {target}")
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Error analyzing {target}: {str(e)}")
                    continue
        
        st.markdown("---")
        st.success("Analysis complete for all targets!")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check if the file format is correct and try again.")

# title
st.title("🧬 Competitive Landscape Analysis")
st.markdown("---")

# Input 
st.header("Input")

# single target vs file upload
tab1, tab2 = st.tabs(["Single Target", "File Upload"])

with tab1:
    target = st.text_input(
        "Enter Molecular Target:",
        placeholder="e.g., CD47, EGFR, VEGF",
        help="Enter the molecular target you want to analyze"
    )
    
    # Single target analysis button
    if st.button("Analyze Single Target", type="primary"):
        if target.strip():
            analyze_single_target(target)
        else:
            st.warning("Please enter a molecular target")

with tab2:
    st.markdown("Upload a file containing multiple molecular targets:")
    st.markdown("**Supported formats:**")
    st.markdown("- **CSV**: Should have a column named 'target' or 'molecular_target'")
    st.markdown("- **TXT**: One target per line (or comma-separated)")
    st.markdown("- **JSON**: Array of objects with 'target' or 'molecular_target' keys")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt', 'json'],
        help="Upload a CSV, TXT, or JSON file with molecular targets"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # File analysis button
        if st.button("Analyze All Targets from File", type="primary"):
            analyze_file_targets(uploaded_file)

# Footer
st.markdown("---")
st.markdown("**Note:** This analysis is based on publicly available data and may not reflect the complete competitive landscape.") 