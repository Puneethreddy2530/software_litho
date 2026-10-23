import subprocess
import sys

def main():
    cmds = [
        ["venv_cuda\\Scripts\\python", "src\\scripts\\train_server.py", "--config", "configs\\phase5_dcid35_server.yaml"],
        ["venv_cuda\\Scripts\\python", "src\\scripts\\train_server.py", "--config", "configs\\phase5_dcid35_cnn_only.yaml"]
    ]
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Failed on command: {' '.join(cmd)}")
            sys.exit(res.returncode)
    print("All Phase 5A trainings completed successfully.")

if __name__ == "__main__":
    main()
