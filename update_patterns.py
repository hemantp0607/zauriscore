import re
import shutil
from pathlib import Path

# Define the target file path
target_path = Path('src/zauriscore/analyzers/gas_optimizer.py')

# Read the gas optimizer file
with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new patterns
new_patterns = {
    # Pattern for multiple small uints
    'multiple_small_uints': {
        'pattern': r'(?:\s*\w+\s+)?(?:uint(?:8|16|32|64|128|256)?)\s+\w+\s*;.*\n(?:\s*\w+\s+)?(?:uint(?:8|16|32|64|128|256)?)\s+\w+\s*;.*\n(?:\s*\w+\s+)?(?:uint(?:8|16|32|64|128|256)?)\s+\w+\s*;',
        'issue': 'Multiple small uints that could be packed',
        'severity': 'medium',
        'suggestion': 'Group smaller uints together to use storage slots more efficiently',
        'saving': '~2000-5000 gas per slot saved',
        'category': 'storage',
        'example_before': 'uint256 public a;\n    uint256 public b;\n    uint8 public c;',
        'example_after': '// Consider packing smaller uints together\n    uint256 public a;\n    uint256 public b;\n    uint8 public c;\n    // Could be packed with other small uints',
        'rationale': 'Multiple small uints can be packed into a single storage slot to save gas'
    },
    # Pattern for dynamic bytes array
    'dynamic_bytes_array': {
        'pattern': r'bytes\s+\w+\s*=\s*new\s+bytes\(\s*\d+\s*\)',
        'issue': 'Dynamic bytes array in storage',
        'severity': 'high',
        'suggestion': 'Use bytes32 or smaller fixed-size bytes if the size is known and <= 32 bytes',
        'saving': '~20000 gas per operation',
        'category': 'storage',
        'example_before': 'bytes data = new bytes(20);',
        'example_after': 'bytes20 data;  // For fixed-size 20-byte data',
        'rationale': 'Fixed-size bytes (bytes1 to bytes32) are more gas efficient than dynamic bytes arrays'
    }
}

# Update the patterns in the content
updated = False
for i, pattern in enumerate(re.finditer(r"\{\s*\'pattern\'\s*:\s*\'([^\']+)\'", content)):
    pattern_name = pattern.group(1)
    if pattern_name in new_patterns:
        # Replace the pattern with the new one
        new_pattern = new_patterns[pattern_name]
        pattern_start = pattern.start()
        # Find the end of this pattern definition (look for the next '}' at the same indentation level)
        brace_count = 1
        pos = pattern_start + 1
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            # Replace the entire pattern definition
            new_content = content[:pattern_start] + str(new_pattern) + content[pos:]
            if new_content != content:
                content = new_content
                updated = True
                print(f"Updated pattern: {pattern_name}")

# Write the updated content back to the file only if changes were made
if updated:
    # Create a backup first
    backup_path = target_path.with_suffix(target_path.suffix + '.bak')
    try:
        shutil.copy2(target_path, backup_path)
        print(f"Backup created at: {backup_path}")
    except Exception as e:
        print(f"Warning: Failed to create backup: {e}")
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patterns updated successfully!")
else:
    print("No matching patterns found; no changes made.")
