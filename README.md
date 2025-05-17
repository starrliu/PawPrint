# PawPrint
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)


PawPrint is a python package for social behavior analyse tool from animal trajectory data.

## Usage

**Step 1: Clone the repository**
```bash
git clone https://github.com/starrliu/PawPrint.git
cd PawPrint
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Download the dataset**

You can download the sample data from [link](https://disk.pku.edu.cn/link/AAA918DF0706484EA99506EC1B5C57C0B9). This is a 1 hour trajectory data of 10 mice following [the format of idtracker.ai](https://idtracker.ai/latest/user_guide/output_structure.html#:~:text=trajectories%3A%20Numpy%20array%20with%20shape%20(N_frames%2C%20N_animals%2C%202)%20with%20the%20xy%20coordinate%20for%20each%20identity%20and%20frame%20in%20the%20video.).

**Step 4: Use API to analyze the data**
```python
from pawprint.data import TrajectoryCollection

# Load the data
tc = TrajectoryCollection(
    trajectory_path="path/to/your/data.csv",
    fps=30,
    scale=SCALE_FROM_PIXEL_TO_CM,
)
# Get the speed distribution of each identity
speeds = tc.to_speed(window_size=5, mode="linear")
```

## Contributing
We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more information on how to contribute to this project.