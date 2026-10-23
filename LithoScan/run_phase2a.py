import subprocess
import sys

def main():
    print("==================================================")
    print("Running Smoke Test...")
    print("==================================================")
    res = subprocess.run([sys.executable, "src/models/train_baseline.py", "--smoke-test"])
    if res.returncode != 0:
        print("Smoke test failed. Stopping.")
        sys.exit(res.returncode)

    print("\\n==================================================")
    print("Smoke test passed. Launching Full Training...")
    print("==================================================")
    res = subprocess.run([sys.executable, "src/models/train_baseline.py"])
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
