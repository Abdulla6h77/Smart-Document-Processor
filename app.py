# app.py - Clean entry point
import sys
import os
from pathlib import Path

# Set up path
project_root = Path(__file__).parent.absolute()
os.chdir(project_root)  # Change to project directory
sys.path.insert(0, str(project_root / "src"))

# Import and run
from streamlit_app import main  # Import main function
import streamlit as st

if __name__ == "__main__":
    main()  # Run your Streamlit app