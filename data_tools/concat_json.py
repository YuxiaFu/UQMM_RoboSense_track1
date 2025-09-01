import json
import os

root = 'QA_data'
json_file_list = [
    'file1.json',
    'file2.json',
    'file3.json'
]
output_file = 'merged_data.json'

merged_data = []
for file_path in json_file_list:
    file_path = os.path.join(root, file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f'len: {round(len(data))}')
        assert isinstance(data, list), f"{file_path}"
        merged_data.extend(data)

output_file = os.path.join(root, output_file)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"✅ merged {len(json_file_list)} files, in total: {len(merged_data)}, save to {output_file}")
