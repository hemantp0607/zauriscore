# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- Ethereum node access (optional)
- Etherscan API key

## Step-by-Step Installation

1. **Clone the Repository**
```bash
git clone https://github.com/zauriscore/zauriscore.git
cd zauriscore
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
```

3. **Install Dependencies**
```bash
pip install -e '.[dev]'
```

4. **Configure Environment Variables**
Create a `.env` file in the project root:
```env
ETHERSCAN_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here  # Optional
```

5. **Verify Installation**
```bash
zauriscore --version
```

## Common Issues

### Installation Errors

1. **Missing System Dependencies**
   - Install required system packages:
     ```bash
     # Ubuntu/Debian
     sudo apt-get install python3-dev build-essential

     # Windows
     # Install Visual C++ Build Tools
     ```

2. **SSL Certificate Issues**
   - Update certificates:
     ```bash
     pip install --upgrade certifi
     ```

### Configuration Issues

1. **API Key Not Found**
   - Ensure `.env` file exists in project root
   - Check environment variable names
   - Verify API key format

2. **Import Errors**
   - Verify virtual environment is activated
   - Check Python version compatibility
   - Reinstall package if needed
