import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = str(file_path) + '.bak'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at: {backup_path}")

def fix_gas_optimizer():
    """Fix the gas optimizer file with correct patterns and syntax"""
    file_path = Path('src/zauriscore/analyzers/gas_optimizer.py')
    
    # Create a backup
    backup_file(file_path)
    
    # Read the original content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the fixed patterns
    fixed_patterns = """    def __init__(self):
        self.optimizations = []
        self.patterns = [
            # Multiple small uints that could be packed
            {
                'pattern': r'((?:\s*\w+\s+)?(?:uint(?:8|16|32|64|128|256)?)\s+\w+\s*;.*\n){2,}(?:\s*\w+\s+)?(?:uint(?:8|16|32|64|128|256)?)\s+\w+\s*;',
                'issue': 'Multiple small uints that could be packed',
                'severity': 'medium',
                'suggestion': 'Group smaller uints together to use storage slots more efficiently',
                'saving': '~2000-5000 gas per slot saved',
                'category': 'storage',
                'example_before': 'uint256 public a;\\n    uint256 public b;\\n    uint8 public c;',
                'example_after': '// Consider packing smaller uints together\\\\n    uint256 public a;\\\\n    uint256 public b;\\\\n    uint8 public c;\\\\n    // Could be packed with other small uints',
                'rationale': 'Multiple small uints can be packed into a single storage slot to save gas'
            },
            # Dynamic bytes array in storage
            {
                'pattern': r'bytes\\s+\\w+\\s*=\\s*new\\s+bytes\\s*\\(\\s*\\d+\\s*\\)',
                'issue': 'Dynamic bytes array in storage',
                'severity': 'high',
                'suggestion': 'Use bytes32 or smaller fixed-size bytes if the size is known and <= 32 bytes',
                'saving': '~20000 gas per operation',
                'category': 'storage',
                'example_before': 'bytes data = new bytes(20);',
                'example_after': 'bytes20 data;  // For fixed-size 20-byte data',
                'rationale': 'Fixed-size bytes (bytes1 to bytes32) are more gas efficient than dynamic bytes arrays'
            },
            # Public mapping with no external usage
            {
                'pattern': r'mapping\\([^)]+\\)\\s+(public\\s+)?(\\w+)\\s*;',
                'issue': 'Public mapping with no external usage',
                'severity': 'medium',
                'suggestion': 'Consider making the mapping private if not needed externally',
                'saving': '~2000 gas per read',
                'category': 'storage',
                'example_before': 'mapping(address => uint) public balances;',
                'example_after': 'mapping(address => uint) private _balances;',
                'rationale': 'Public variables generate implicit getter functions which cost gas'
            },
            # Inefficient struct packing
            {
                'pattern': r'struct\\s+(\\w+?)\\s*{([^}]*)}',
                'issue': 'Inefficient struct packing',
                'severity': 'high',
                'suggestion': 'Re-order struct fields to pack variables more efficiently (32-byte slots)',
                'saving': '~2000-5000 gas per slot',
                'category': 'storage',
                'example_before': 'struct User {\\\\n    bool isActive;\\\\n    uint256 id;\\\\n    address user;\\\\n    uint8 age;\\\\n}',
                'example_after': 'struct User {\\\\n    address user;\\\\n    uint256 id;\\\\n    uint8 age;\\\\n    bool isActive;\\\\n}',
                'rationale': 'Packing smaller types together can reduce storage slots used'
            }
        ]
"""
    
    # Replace the patterns section in the file
    new_content = re.sub(
        r'def __init__\(self\):.*?self\.patterns\s*=\s*\[.*?\]',
        fixed_patterns,
        content,
        flags=re.DOTALL
    )
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Gas optimizer file has been updated with fixed patterns.")

if __name__ == "__main__":
    fix_gas_optimizer()
