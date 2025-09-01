import json
import os
import time
from tqdm import tqdm
from typing import Any, Dict, List, Literal
from peft import PeftModel, PeftConfig

from eval_tools.senna_qa_utils import eval_multi_img_model_wo_init
from llava.model.builder import load_senna_pretrained_model

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

def load_or_create_output(output_path: str) -> List[Dict[str, Any]]:
    """Load existing output if it exists, or create a new output file."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing results from {output_path}")
            return existing_data
        except Exception as e:
            print(f"Error loading existing output file: {str(e)}. Starting fresh.")
            return []
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return []

def save_output(output_path: str, data: List[Dict[str, Any]]):
    """Save output data to file."""
    temp_path = output_path + '.tmp'
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"Error saving output: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
def process_qa_data(tokenizer, model, image_processor, data: List[Dict[str, Any]], data_senna_fmt, output_path: str) -> List[Dict[str, Any]]:
    """Process QA data and generate answers, saving results in real-time."""
    # Load existing results or create new output file
    output_data = load_or_create_output(output_path)
    
    # Create set of already processed sample IDs
    processed_ids = {sample.get('id') for sample in output_data}
    
    # Filter out already processed samples
    remaining_data = [sample for sample in data if sample.get('id') not in processed_ids]
    
    if not remaining_data:
        print("All samples have already been processed!")
        return output_data
    
    # Process each remaining sample
    with tqdm(total=len(remaining_data), desc="Processing samples") as pbar:
        assert len(remaining_data) == len(data_senna_fmt)
        # for sample in remaining_data:
        for sample, senna_sample in zip(remaining_data, data_senna_fmt):
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "query": senna_sample['question'],
                "conv_mode": 'llava_v1',
                "image_file": senna_sample['images'],
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

            output_sample = sample.copy()
            
            answer = eval_multi_img_model_wo_init(args, tokenizer, model, image_processor)
            
            output_sample['answer'] = answer
            
            output_data.append(output_sample)
            
            save_output(output_path, output_data)
            
            pbar.update(1)
    
    return output_data

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Senna Eval')
    parser.add_argument('ckpt', type=str, help='model path')
    parser.add_argument('phase', type=str, help='phase, 1 or 2', choices=[1, 2])
    args = parser.parse_args()
    print(f'eval model path: {os.path.abspath(args.ckpt)} on phase {args.phase}')
    return args

def load_senna(model_path: str):
    config = PeftConfig.from_pretrained(model_path)
    tokenizer, model, image_processor, context_len = load_senna_pretrained_model(
        config.base_model_name_or_path, None, model_name='llava', device_map=0)
    model = PeftModel.from_pretrained(model, model_path)
    return tokenizer, model, image_processor

def get_path(phase: Literal[1, 2]):
    if phase == 1:
        original_eval_data_path = 'QA_data/robosense_track1_converted.json'
        eval_data_path = 'QA_data/robosense_track1_converted2senna.json'
    elif phase == 2:
        original_eval_data_path = 'QA_data/robosense_track1_phase2_converted.json'
        eval_data_path = 'QA_data/robosense_track1_phase2_converted2senna.json'
    else:
        raise ValueError("Phase must be 1 or 2.")

    output_path = f'robosense_results/inference_results_{DATE_TIME}.json'
    print(f'eval_data_path: {os.path.abspath(eval_data_path)}')
    return original_eval_data_path, eval_data_path, output_path

if __name__ == "__main__":
    args = get_parser()
    model_path = args.ckpt
    phase = args.phase
    original_eval_data_path, eval_data_path, output_path = get_path(phase)
    
    # Load input data
    print(f"Loading input data from {original_eval_data_path}")
    with open(original_eval_data_path, 'r') as f:
        data = json.load(f)
    with open(eval_data_path, 'r') as file:
        data_senna_fmt = json.load(file)

    # Load model
    tokenizer, model, image_processor = load_senna(model_path)

    # Process data and generate answers
    print("Processing data and generating answers...")
    output_data = process_qa_data(tokenizer, model, image_processor, data, data_senna_fmt, output_path)

    print("Done!")
