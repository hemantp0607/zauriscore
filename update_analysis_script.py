"""
This script updates the comprehensive_contract_analysis.py file to fix the temporary directory handling.
"""
import os
import re

def update_file(file_path):
    """Update the file with the necessary changes."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the temp directory creation and handling
    pattern = r'(# Directory to place temporary contract files\n\s+temp_dir = os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\), \'temp_contracts\'\)\n\s+os\.makedirs\(temp_dir, exist_ok=True\)\n\n\s+# Default main file path\n\s+import time\n\s+main_file = os\.path\.abspath\(os\.path\.join\(temp_dir, f\'Contract_\{int\(time\.time\(\)\}\.sol\'\)\))'

    replacement = (
        '# Create a unique temporary directory for this analysis\n'
        '            import tempfile\n'
        '            temp_dir = tempfile.mkdtemp(prefix=\'zauriscore_\')\n'
        '            self.logger.debug(f\"Created temporary directory: {temp_dir}\")\n'
        '            # Default main file path\n'
        '            import time\n'
        '            main_file = os.path.abspath(os.path.join(temp_dir, f\'{os.path.basename(temp_dir)}.sol\'))'
    )
    
    # Perform the replacement
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Successfully updated {file_path}")

if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'src',
        'zauriscore',
        'analyzers',
        'comprehensive_contract_analysis.py'
    )
    
    if os.path.exists(file_path):
        update_file(file_path)
    else:
        print(f"Error: File not found: {file_path}")
