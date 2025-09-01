import json
import os

CAM_NAME = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
PROMPT = "<CAM_FRONT>:\n<image>\n" \
         "<CAM_FRONT_LEFT>:\n<image>\n" \
         "<CAM_FRONT_RIGHT>:\n<image>\n" \
         "<CAM_BACK>:\n<image>\n" \
         "<CAM_BACK_LEFT>:\n<image>\n" \
         "<CAM_BACK_RIGHT>:\n<image>\n"

def format_qa(question, answer):
    question_dict, answer_dict = {}, {}
    question_dict["from"] = "human"
    question_dict["value"] = PROMPT + question
    answer_dict["from"] = "gpt"
    answer_dict["value"] = answer

    return [question_dict, answer_dict]

def convert_img_path(dataset_root, img_path):
    abs_path = os.path.join(dataset_root, img_path)
    return abs_path

def main(root, dataset_root, file_path, output_file):
    new_data = []
    with open(os.path.join(root, file_path), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            new_sample = {}
            new_sample['scene_token'] = sample['scene_token']
            new_sample['frame_token'] = sample['frame_token']
            new_sample['category'] = sample['category']
            new_sample['image'] = os.path.join(dataset_root, convert_img_path(dataset_root, sample['img_paths']['CAM_FRONT']))
            new_sample['images'] = [os.path.join(dataset_root, convert_img_path(dataset_root, sample['img_paths'][cam])) for cam in CAM_NAME]
            new_sample['question'] = PROMPT + sample['question']
            new_sample['conversations'] = format_qa(sample['question'], sample['answer'])
            new_data.append(new_sample)

    output_file = os.path.join(root, output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"✅ successful, save to {output_file}, total {len(new_data)} samples")

if __name__ == "__main__":
    root = 'QA_data'

    # path of phase 1
    dataset_root = '/path/to/nuscenes_data/nuScenes/v1.0/'
    file_path = 'robosense_track1_converted.json'
    output_file = 'robosense_track1_converted2senna.json'

    # path of phase 2
    dataset_root = '/path/to/phase2_image/'
    file_path = 'robosense_track1_phase2_converted.json'
    output_file = 'robosense_track1_phase2_converted2senna.json'

    main(root, dataset_root, file_path, output_file)