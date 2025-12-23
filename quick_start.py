#!/usr/bin/env python3
"""
Quick start script for the document processor
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 Smart Document Processor - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    # Check for .env file
    if not Path(".env").exists() and Path(".env.example").exists():
        print("⚠️  Creating .env file from template...")
        with open(".env.example", "r") as f:
            content = f.read()
        with open(".env", "w") as f:
            f.write(content.replace("your_ernie_api_key_here", "YOUR_API_KEY_HERE"))
        print("✅ Created .env file - please add your ERNIE API key")
    
    # Check for test documents
    test_dir = Path("test_documents")
    if not test_dir.exists():
        test_dir.mkdir()
        print("📁 Created test_documents folder")
        print("💡 Add some PDF/JPG/PNG files to test the system")
    
    print("\n🎯 Next Steps:")
    print("1. Add your ERNIE API key to .env file")
    print("2. Add test documents to test_documents/ folder")
    print("3. Choose your interface:")
    print("   • Streamlit: streamlit run streamlit_app.py")
    print("   • CLI: python cli_interface.py process document.pdf")
    print("   • Direct: python use_directly.py")
    print("   • API: python src/main.py")
    print("4. Test: python test_system.py")

if __name__ == "__main__":
    main()