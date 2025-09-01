# UQMM_RoboSense_track1

## Getting Started
This repository is built on top of [Senna](https://github.com/hustvl/Senna). Please follow the installation instructions provided in the Senna repository to set up the environment before running this code. For package versions, please refer to the [`environment.yml`](environment.yml) file.

### Data Preparation
We first download the raw test data from the competition page and convert it into the required format following the [official instructions](data_tools/convert_format.py).
After conversion, place the files into the `QA_data` folder.

To further adapt the dataset into the format required by the **Senna** model, first update the `dataset_root` path inside [`convert_fmt_from_track1_to_Senna.py`](data_tools/convert_fmt_from_track1_to_Senna.py) to point to the parent directory of the nuScenes image data. For example, if your image data is located at `dataset/nuScenes/v1.0/samples`, then `dataset_root` should be set to `dataset/nuScenes/v1.0`. After updating the path, run the script.

### Weight
We finetune the pretrained **Senna** model for Phase 1 using LoRA, and release the checkpoint [here](https://drive.google.com/drive/folders/1p-HK3H_aLLFPldVaNRzoO_TiHiF6mQQj?usp=drive_link).
To use this checkpoint, first download the [pretrained Senna](https://huggingface.co/rb93dett/Senna) and [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) models.
Then, update the `mm_vision_tower` path in the `config.json` file of the Senna model folder to point to the downloaded [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) model.

### Evaluation
To evaluate on the Phase 1 test set, update the `ckpt` path in `eval.sh`, then run:
```shell
sh eval_tools/eval.sh
```

### Fine-tuning
The data we use for fine-tuning Senna in Phase 1 can be found [here](https://drive.google.com/drive/folders/12QIqvhG5h2MgLx4RUJyRbps_c798Gpbx?usp=drive_link). To fine-tune, update the path in `robosense_FT.sh`, then run:
```shell
sh train_tools/robosense_FT.sh
```
