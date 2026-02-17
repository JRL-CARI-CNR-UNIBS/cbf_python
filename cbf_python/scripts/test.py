import os
from pathlib import Path

# 1. Get the absolute path of the directory where THIS script is located
script_dir = Path(__file__).parent.resolve()

# 2. Navigate relative to the script's location
# Using / operator with Path objects is the cleanest way to join paths
params_path = (script_dir / ".." / "parameters_set.csv").resolve()

# 3. Change the working directory to the folder containing the target file
os.chdir(params_path.parent)

print(f"Script directory: {script_dir}")
print(f"Target file path: {params_path}")
print(f"New working directory: {os.getcwd()}")