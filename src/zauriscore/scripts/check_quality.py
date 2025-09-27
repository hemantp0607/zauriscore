"""
Run static analysis and quality checks on the codebase.
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd: str, cwd: str = None) -> int:
    """Run a shell command and return the return code."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or os.getcwd(),
            check=False,
            text=True,
            capture_output=True
        )
        print(f"\n{'='*80}\n{cmd}\n{'='*80}")
        print(result.stdout)
        if result.stderr:
            print(f"ERRORS:\n{result.stderr}")
        return result.returncode
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return 1

def main():
    """Run all quality checks."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    
    # File to analyze
    target_file = src_dir / "zauriscore" / "analyzers" / "heuristic_analyzer.py"
    test_file = project_root / "tests" / "test_heuristic_analyzer.py"
    
    print(f"🔍 Analyzing {target_file}")
    print(f"📝 Test file: {test_file}")
    
    # 1. Run pylint
    print("\n" + "="*80)
    print("1/5: Running pylint...")
    pylint_cmd = f"pylint {target_file} --rcfile={project_root}/.pylintrc"
    pylint_rc = run_command(pylint_cmd, project_root)
    
    # 2. Run flake8
    print("\n" + "="*80)
    print("2/5: Running flake8...")
    flake8_cmd = f"flake8 {target_file} --config={project_root}/.flake8"
    flake8_rc = run_command(flake8_cmd, project_root)
    
    # 3. Run mypy
    print("\n" + "="*80)
    print("3/5: Running mypy...")
    mypy_cmd = f"mypy {target_file} --config-file {project_root}/mypy.ini"
    mypy_rc = run_command(mypy_cmd, project_root)
    
    # 4. Run tests
    print("\n" + "="*80)
    print("4/5: Running tests...")
    test_cmd = f"pytest {test_file} -v"
    test_rc = run_command(test_cmd, project_root)
    
    # 5. Generate test coverage
    print("\n" + "="*80)
    print("5/5: Generating test coverage...")
    cov_cmd = f"pytest {test_file} --cov={src_dir} --cov-report=term-missing"
    cov_rc = run_command(cov_cmd, project_root)
    
    # Summary
    print("\n" + "="*80)
    print("📊 Quality Check Summary")
    print("="*80)
    print(f"✅ pylint: {'PASSED' if pylint_rc == 0 else 'FAILED'}")
    print(f"✅ flake8: {'PASSED' if flake8_rc == 0 else 'FAILED'}")
    print(f"✅ mypy:   {'PASSED' if mypy_rc == 0 else 'FAILED'}")
    print(f"✅ tests:  {'PASSED' if test_rc == 0 else 'FAILED'}")
    print(f"✅ coverage: {'PASSED' if cov_rc == 0 else 'CHECK REPORT'}")
    
    # Return non-zero if any check failed
    if any(rc != 0 for rc in [pylint_rc, flake8_rc, mypy_rc, test_rc, cov_rc]):
        sys.exit(1)

if __name__ == "__main__":
    main()
