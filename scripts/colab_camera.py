"""Run a koopman_sbi command inside a Colab session, from the repo copy on the mounted Google Drive.

Use with a persistent session (the code, logs/ caches and outputs all live on Drive):
    colab new --gpu A100 -s camera
    colab drivemount -s camera
    colab install -s camera sbibm torchdiffeq normflows
    colab exec -s camera --timeout 86400 -f scripts/colab_camera.py
    colab stop camera        # when finished, to release the GPU
Set KOOPMAN_REPO / KOOPMAN_COMMAND with --env to change the repo path or the command.
"""

import os
import subprocess
import sys

REPO = os.environ.get("KOOPMAN_REPO", "/content/drive/Othercomputers/My MacBook Air (1)/KOOPMAN_SBI")
COMMAND = os.environ.get("KOOPMAN_COMMAND", "train-tensorproduct-koopman --task camera_model").split()

os.chdir(REPO)
print(f"running in {REPO}: python -m koopman_sbi {' '.join(COMMAND)}", flush=True)
result = subprocess.run([sys.executable, "-m", "koopman_sbi", *COMMAND])
if result.returncode != 0:
    print(f"\nkoopman_sbi failed with exit code {result.returncode}", file=sys.stderr)
    sys.exit(result.returncode)
