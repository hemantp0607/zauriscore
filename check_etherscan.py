import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Etherscan API key
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')

if not ETHERSCAN_API_KEY:
    print("Error: Etherscan API key not found. Please set ETHERSCAN_API_KEY in your .env file.")
    exit(1)

# Contract address to analyze
contract_address = "0x33b531c0CE13c6e20a1DD852332026707Ca44e9C"

# Make API request
url = "https://api.etherscan.io/api"
params = {
    "module": "contract",
    "action": "getsourcecode",
    "address": contract_address,
    "apikey": ETHERSCAN_API_KEY
}

print(f"\nMaking request to Etherscan API...")
print(f"URL: {url}")
print(f"Params: {params}")

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    print("\nAPI Response:")
    print(f"Status: {data.get('status', 'N/A')}")
    print(f"Message: {data.get('message', 'N/A')}")
    print(f"Full Response: {data}")
    
    if data.get('status') == '1':
        result = data.get('result', [{}])[0]
        print("\nContract Details:")
        print(f"Name: {result.get('ContractName', 'N/A')}")
        print(f"Compiler: {result.get('CompilerVersion', 'N/A')}")
        print(f"Optimization: {result.get('OptimizationUsed', 'N/A')}")
        print(f"License: {result.get('LicenseType', 'N/A')}")
        
        source_code = result.get('SourceCode', '')
        print(f"\nSource Code Length: {len(source_code)} bytes")
        print(f"First 200 chars: {source_code[:200]}")
    else:
        print(f"\nError: {data.get('message', 'Unknown error')}")
        print(f"Details: {data.get('result', 'No details')}")

except requests.exceptions.RequestException as e:
    print(f"\nError making request: {e}")
except Exception as e:
    print(f"\nUnexpected error: {e}")
