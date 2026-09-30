import os
import sys
import subprocess

if __name__ == "__main__":
    # Ensure Streamlit is invoked with the active Python environment
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        # Fallback to direct system invocation
        os.system("streamlit run app.py")