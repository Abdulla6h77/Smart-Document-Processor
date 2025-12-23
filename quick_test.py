# final_quick_test.py - Works with either CAMEL or fallback
import sys
from pathlib import Path

print("🚀 Testing Smart Document Processor - CAMEL Compatible")
print("=" * 60)

# Test the import fix
print("1. Testing agent imports...")

# Test CAMEL import with fallback
try:
    import camel
    from camel.agents import ChatAgent
    print("✅ CAMEL-AI imported successfully")
    USING_CAMEL = True
except ImportError:
    try:
        # Use our fallback
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from agents.fallback_agent import FallbackAgent
        print("✅ Using fallback agent implementation")
        USING_CAMEL = False
    except Exception as e:
        print(f"❌ Agent import failed: {e}")
        USING_CAMEL = False

# Test core functionality
print("\n2. Testing core packages...")
core_tests = [
    ("paddleocr", "OCR Engine"),
    ("fastapi", "Web Framework"), 
    ("streamlit", "UI Framework"),
    ("pandas", "Data Processing"),
    ("aiohttp", "HTTP Client"),
    ("loguru", "Logging"),
    ("PIL", "Image Processing"),
    ("yaml", "Configuration"),
]

all_good = True
for package, description in core_tests:
    try:
        if package == "PIL":
            from PIL import Image
        else:
            __import__(package)
        print(f"✅ {package} - {description}")
    except ImportError as e:
        print(f"❌ {package} - {description}: {e}")
        all_good = False

# Test environment
print("\n3. Testing environment...")
import os
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv('OPENROUTER_API_KEY'):
        print("✅ OpenRouter API key found!")
    else:
        print("⚠️ No OpenRouter API key found")
else:
    print("⚠️ No .env file found")

print("\n" + "=" * 60)
if all_good:
    print("🎉 System ready!")
    if USING_CAMEL:
        print("✅ Using CAMEL-AI framework")
    else:
        print("✅ Using fallback agent implementation")
    print("✅ You can now run the application!")
else:
    print("❌ Some packages missing")