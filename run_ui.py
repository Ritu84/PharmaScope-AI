"""
Simple script to run the Streamlit UI for competitive landscape analysis.
"""

import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit application."""
    app_path = os.path.join("ui", "streamlit_app.py")
    

    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    try:
        print("Starting Streamlit application...")
        print("Open your browser and go to: http://localhost:8001")
        print("Press Ctrl+C to stop the application")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port=8001",
            "--server.address=localhost"
        ])
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit() 