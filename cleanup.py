import os
import shutil

def cleanup_temp_files():
    """Clean up temporary files and directories."""
    temp_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'zauriscore', 'analyzers', 'temp_contracts'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zauriscore', 'analyzers', 'temp_contracts'),
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            print(f"Removing directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print(f"Successfully removed {temp_dir}")
            except Exception as e:
                print(f"Error removing {temp_dir}: {e}")
        else:
            print(f"Directory does not exist: {temp_dir}")

if __name__ == "__main__":
    cleanup_temp_files()
