"""
This script is a temporary fix to ensure the temp_contracts directory is properly set up
and cleaned before running the analysis.
"""
import os
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_temp_directory():
    """Set up the temporary directory for contract analysis."""
    # Path to the temp_contracts directory
    temp_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'src',
        'zauriscore',
        'analyzers',
        'temp_contracts'
    )
    
    # Remove the directory if it exists
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed existing directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to remove directory {temp_dir}: {e}")
            return False
    
    # Create the directory
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created directory: {temp_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {temp_dir}: {e}")
        return False

if __name__ == "__main__":
    if setup_temp_directory():
        print("Temporary directory setup completed successfully.")
        print("You can now run the analysis again.")
    else:
        print("Failed to set up temporary directory. Please check the logs for details.")
