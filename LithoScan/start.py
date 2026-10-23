import subprocess
import sys
import os

def main():
    print("Starting LithoScan Dashboard...")
    
    # Determine the path to app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
        
    # Check for virtual environment first
    venv_python = os.path.join(current_dir, "venv_cuda", "Scripts", "python.exe")
    
    if os.path.exists(venv_python):
        python_exe = venv_python
        print(f"Using virtual environment: {venv_python}")
    else:
        python_exe = sys.executable
        print(f"Using system python: {python_exe}")
        
    try:
        # Run streamlit app
        subprocess.run([python_exe, "-m", "streamlit", "run", app_path], check=True, cwd=current_dir)
    except KeyboardInterrupt:
        print("\nLithoScan Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running the dashboard: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
