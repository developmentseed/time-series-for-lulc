from pathlib import Path
from sklearn.model_selection import train_test_split

cubes_dir = Path("./data/cubesxy")

stack_list = list(cubes_dir.glob("*.npz"))
train_list, val_list = train_test_split(stack_list, test_size=0.25, random_state=42)
val_list, test_list = train_test_split(val_list, test_size=0.4, random_state=42)

(cubes_dir / "train").mkdir(exist_ok=True)
(cubes_dir / "val").mkdir(exist_ok=True)
(cubes_dir / "test").mkdir(exist_ok=True)

# Move files to respective folders
for f in train_list:
    f.rename(cubes_dir / "train" / f.name)

for f in val_list:
    f.rename(cubes_dir / "val" / f.name)

for f in test_list:
    f.rename(cubes_dir / "test" / f.name)
