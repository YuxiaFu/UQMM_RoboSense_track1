# UQMM_RoboSense_track1

## Getting Started
This repository is built on top of [Senna](https://github.com/hustvl/Senna). Please follow the installation instructions provided in the Senna repository to set up the environment before running this code. For package versions, please refer to the [`environment.yml`](environment.yml) file.

### Data Preparation
We first download the raw test data from the competition page and convert it into the required format following the official instructions.
After conversion, place the files into the `QA_data` folder.

To further adapt the dataset into the format required by the **Senna** model, update the `dataset_root` path inside [`convert_fmt_from_track1_to_Senna.py`](data_tools/convert_fmt_from_track1_to_Senna.py) and then run the script.

### Weight

### Evaluation

### Fine-tuning
