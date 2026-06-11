from pathlib import Path
import shutil

# # Method 1. Recursive search
# patterns = [
#     '*.so', '*/__pycache__/', '*build/', '*.egg-info/', '*cython_debug/'
# ]
# to_remove = []
# for pattern in patterns:
#     for path in Path('.').rglob(pattern):
#         to_remove.append(path)

# Method 2. Removing .gitignored files
with open('.gitignore') as f:
    patterns = [x for x in f.read().split('\n') if (len(x)>0) and (x.strip()[0]!='#')]
to_remove = []
for pattern in patterns:
    for path in Path('.').glob(pattern):
        to_remove.append(path)

for path in to_remove:
    print(f"Removing {path}")
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif not path.exists():
        continue
    else:
        raise ValueError(f"Unknown path type: {path}")