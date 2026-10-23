import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("  Starting LithoScan Dashboard (Automated Drill-Core Logging)")
    print("=" * 60)
    
    root_dir = Path(__file__).resolve().parent
    litho_dir = root_dir / "LithoScan"
    app_path = litho_dir / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
        
    # Check for virtual environments
    candidate_venvs = [
        litho_dir / "venv_cuda" / "Scripts" / "python.exe",
        litho_dir / "venv" / "Scripts" / "python.exe",
        litho_dir / ".venv" / "Scripts" / "python.exe",
        root_dir / "venv" / "Scripts" / "python.exe",
        root_dir / ".venv" / "Scripts" / "python.exe",
        litho_dir / "venv_cuda" / "bin" / "python",
        litho_dir / "venv" / "bin" / "python",
        root_dir / "venv" / "bin" / "python",
    ]
    
    python_exe = sys.executable
    for venv in candidate_venvs:
        if venv.exists():
            python_exe = str(venv)
            print(f"Using virtual environment Python: {python_exe}")
            break
    else:
        print(f"Using current Python executable: {python_exe}")
        
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", str(app_path)], check=True, cwd=str(litho_dir))
    except KeyboardInterrupt:
        print("\nLithoScan Dashboard stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running the dashboard: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
