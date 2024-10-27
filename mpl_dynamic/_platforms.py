import os
import sys
in_vscode = "VSCODE_CWD" in os.environ or "VSCODE_PID" in os.environ # running in vscode
in_colab = "google.colab" in sys.modules # running in colab
in_pyodide = '_pyodide' in sys.modules