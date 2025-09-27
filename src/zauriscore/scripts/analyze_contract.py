import sys
import logging
from zauriscore.analyzers.comprehensive_contract_analysis import ComprehensiveContractAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_section_header(title):
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")

def print_subsection_header(title):
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")

def format_dict(data, indent=2):
    if not isinstance(data, dict):
        return str(data)
    result = []
    for key, value in data.items():
        if isinstance(value, dict):
            result.append(f"{' ' * indent}{key}:")
            result.append(format_dict(value, indent + 2))
        else:
            result.append(f"{' ' * indent}{key}: {value}")
    return '\n'.join(result)

def main():
    # Contract address to analyze
    contract_address = "0xe72E24b7991445Af9BCcf6918209eBf6Bf304f78"
    
    # Verify contract address format
    if not contract_address.startswith('0x') or len(contract_address) != 42:
        print("Error: Invalid contract address format. Must start with '0x' and be 42 characters long.")
        sys.exit(1)
    
    print_section_header(f"Analyzing contract: {contract_address}")
    
    try:
        # Initialize the analyzer
        analyzer = ComprehensiveContractAnalyzer()
        
        # Get raw source code first
        print("\n[+] Fetching raw contract data from Etherscan...")
        raw_data = analyzer.get_contract_source(contract_address)
        
        print("\n[+] Raw Data Received:")
        print(f"Status: {raw_data.get('status', 'N/A')}")
        print(f"Message: {raw_data.get('message', 'N/A')}")
        
        if raw_data.get('status') == '1':
            contract_data = raw_data.get('result', [{}])[0]
            print("\n[+] Contract Details:")
            print(f"Name: {contract_data.get('ContractName', 'N/A')}")
            print(f"Compiler: {contract_data.get('CompilerVersion', 'N/A')}")
            print(f"Optimization: {contract_data.get('OptimizationUsed', 'N/A')}")
            print(f"License: {contract_data.get('LicenseType', 'N/A')}")
            
            # Handle source code
            source_code = contract_data.get('SourceCode', '')
            if source_code:
                print("\n[+] Source Code Analysis:")
                print(f"Source Code Length: {len(source_code)} bytes")
                print(f"Is JSON: {source_code.startswith('{')}")
                
                # Try to parse JSON source code
                if source_code.startswith('{'):
                    try:
                        import json
                        source_json = json.loads(source_code)
                        print("\n[+] JSON Source Structure:")
                        print(f"Files: {len(source_json.get('sources', {}))}")
                        for file_path, file_data in source_json.get('sources', {}).items():
                            print(f"\nFile: {file_path}")
                            print(f"Content Length: {len(file_data.get('content', ''))} bytes")
                            print(f"First 100 chars: {file_data.get('content', '')[:100]}")
                    except json.JSONDecodeError as e:
                        print(f"\n[!] JSON parsing error: {e}")
                        print("Raw source code (first 100 chars):")
                        print(source_code[:100])
            else:
                print("\n[!] No source code found in response")
        else:
            print(f"\n[!] Error from Etherscan: {raw_data.get('message', 'Unknown error')}")
            print(f"Details: {raw_data.get('result', 'No details')}")
            
        # Analyze the contract
        print("\n[+] Running full analysis...")
        result = analyzer.analyze_contract(contract_address=contract_address)
        
        # Print the analysis results
        print_section_header("Analysis Results")
        
        # Basic contract info
        print_subsection_header("Contract Information")
        print(format_dict({
            'Name': result.get('contract_name', 'N/A'),
            'Compiler Version': result.get('compiler_version', 'N/A'),
            'Optimization Used': result.get('optimization_used', 'N/A'),
            'Optimization Runs': result.get('optimization_runs', 'N/A'),
            'License Type': result.get('license_type', 'N/A'),
            'Source Code Size': f"{len(result.get('source_code', ''))} bytes"
        }))
        
        # Functions and Events
        print_subsection_header("Contract Functions and Events")
        if 'functions' in result:
            print("Functions:")
            for func in result['functions']:
                print(f"  - {func['name']} ({func['visibility']})")
                if func.get('modifiers'):
                    print(f"    Modifiers: {', '.join(func['modifiers'])}")
                if func.get('state_mutability'):
                    print(f"    Mutability: {func['state_mutability']}")
        else:
            print("  No functions found")
            
        if 'events' in result:
            print("\nEvents:")
            for event in result['events']:
                print(f"  - {event['name']}")
        else:
            print("\nNo events found")
        
        # State Variables
        print_subsection_header("State Variables")
        if 'state_variables' in result:
            for var in result['state_variables']:
                print(f"  - {var['name']}: {var['type']}")
                if var.get('visibility'):
                    print(f"    Visibility: {var['visibility']}")
                if var.get('constant'):
                    print(f"    Constant: {var['constant']}")
        else:
            print("No state variables found")
            
        # Gas Optimizations
        print_subsection_header("Gas Optimization Analysis")
        if 'gas_optimizations' in result and result['gas_optimizations']:
            print("Optimization Opportunities:")
            for idx, issue in enumerate(result['gas_optimizations'], 1):
                print(f"\n{idx}. {issue['issue']} ({issue['severity'].upper()})")
                print(f"  Suggestion: {issue['suggestion']}")
                print(f"  Potential savings: {issue.get('saving', 'N/A')}")
                print(f"  Category: {issue['category']}")
                print(f"  Example Before: {issue.get('example_before', 'N/A')}")
                print(f"  Example After: {issue.get('example_after', 'N/A')}")
                print(f"  Rationale: {issue.get('rationale', 'N/A')}")
        else:
            print("No gas optimization issues found")
            
        # Security Analysis
        print_subsection_header("Security Analysis")
        if 'security_issues' in result and result['security_issues']:
            print("Security Issues Found:")
            for idx, issue in enumerate(result['security_issues'], 1):
                print(f"\n{idx}. {issue['check']} ({issue['impact'].upper()})")
                print(f"  Description: {issue['description']}")
                if issue.get('recommendation'):
                    print(f"  Recommendation: {issue['recommendation']}")
                if issue.get('lines_of_code'):
                    print(f"  Lines of Code: {issue['lines_of_code']}")
        else:
            print("No security issues found")
            
        # Contract Inheritance
        print_subsection_header("Contract Inheritance")
        if 'inheritance' in result:
            print("Inheritance Chain:")
            for parent in result['inheritance']:
                print(f"  - {parent}")
        else:
            print("No inheritance found")
            
        # Libraries Used
        print_subsection_header("Libraries Used")
        if 'libraries' in result:
            print("Libraries:")
            for lib in result['libraries']:
                print(f"  - {lib}")
        else:
            print("No libraries used")
            
        print_section_header("Analysis Complete")
        
    except Exception as e:
        print(f"\n[!] Error analyzing contract: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
