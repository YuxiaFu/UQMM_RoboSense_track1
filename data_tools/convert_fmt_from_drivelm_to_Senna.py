import json
import os
from tqdm import tqdm

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
    abs_path = os.path.join(dataset_root, img_path[len("../nuscenes/"):])
    return abs_path

def main(root, dataset_root, file_path, output_file):
    new_data = []
    with open(os.path.join(root, file_path), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for scene_token, sample in tqdm(data.items()):
            for frame_token, frame in sample['key_frames'].items():
                image = convert_img_path(dataset_root, frame['image_paths']['CAM_FRONT'])
                images = [convert_img_path(dataset_root, frame['image_paths'][cam]) for cam in CAM_NAME]
                for qa_type, qa_pairs in frame['QA'].items():
                    for qa in qa_pairs:
                        new_sample = {}
                        new_sample['scene_token'] = scene_token
                        new_sample['frame_token'] = frame_token
                        new_sample['image'] = image
                        new_sample['images'] = images
                        new_sample['type'] = qa_type
                        new_sample['question'] = PROMPT + qa['Q']
                        new_sample['conversations'] = format_qa(qa['Q'], qa['A'])
                        new_data.append(new_sample)
    print(len(new_data))

    output_file = os.path.join(root, output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"✅ successful, save to {output_file}")

if __name__ == "__main__":
    root = 'QA_data'
    dataset_root = '/path/to/nuscenes_data/nuScenes/v1.0/'

    file_path = 'drivelm_train_nus.json'
    output_file = 'robosense_track1_drivelm2senna.json'

    main(root, dataset_root, file_path, output_file)