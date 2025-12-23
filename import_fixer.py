# create_import_fixer.py - Converts all relative imports to absolute
import os
import re
from pathlib import Path

def fix_relative_imports():
    """Convert all relative imports to absolute imports in src/"""
    
    src_path = Path("src")
    if not src_path.exists():
        print("❌ src directory not found")
        return
    
    # Files to process
    python_files = list(src_path.rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files to fix")
    
    fixed_count = 0
    
    for py_file in python_files:
        relative_path = py_file.relative_to(src_path)
        parent_dir = relative_path.parent
        
        # Read file content
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Determine the package based on file location
        if 'agents' in str(parent_dir):
            package = 'src.agents'
        elif 'models' in str(parent_dir):
            package = 'src.models'
        elif 'utils' in str(parent_dir):
            package = 'src.utils'
        else:
            package = 'src'
        
        # Fix imports in order (most specific first)
        
        # 1. Fix parent-level imports: 'from ..models.X' → 'from src.models.X'
        content = re.sub(r'from \.\.models\.([a-z_]+)', r'from src.models.\1', content)
        content = re.sub(r'from \.\.utils\.([a-z_]+)', r'from src.utils.\1', content)
        content = re.sub(r'from \.\.agents\.([a-z_]+)', r'from src.agents.\1', content)
        
        # 2. Fix same-package imports: 'from .module' → 'from src.package.module'
        # Handle specific known modules first
        if 'agents' in str(parent_dir):
            # Fix agent-specific imports
            content = re.sub(r'from \.base_agent import', 'from src.agents.base_agent import', content)
            content = re.sub(r'from \.([a-z_]+_agent) import', r'from src.agents.\1 import', content)
            # Fix other same-package imports in agents
            content = re.sub(r'from \.([a-z_]+) import', rf'from {package}.\1 import', content)
        elif 'models' in str(parent_dir):
            # Fix model imports
            content = re.sub(r'from \.([a-z_]+) import', rf'from {package}.\1 import', content)
        elif 'utils' in str(parent_dir):
            # Fix util imports
            content = re.sub(r'from \.([a-z_]+) import', rf'from {package}.\1 import', content)
        
        # 3. Fix 'from . import X' pattern
        content = re.sub(r'from \. import ([a-z_]+)', rf'from {package} import \1', content)
        
        # Write back if changes were made
        if content != original_content:
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"✅ Fixed: {relative_path}")
    
    print(f"🎉 Fixed {fixed_count} files!")

if __name__ == "__main__":
    fix_relative_imports()