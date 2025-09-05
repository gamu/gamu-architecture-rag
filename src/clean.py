import json
from pathlib import Path

def replace_text_in_files(kb_origin_path, kb_clean_path, terms_map):
    with open(terms_map, 'r', encoding='utf-8') as json_file:
        replacements = json.load(json_file)
    
    directory = Path(kb_origin_path)
    
    for file_path in directory.glob('*'):
        output_file_path = kb_clean_path + '/' + file_path.name

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        for key, value in replacements.items():
            content = content.replace(key, value)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            print(f"cleaned saved to {output_file_path}")
            
    print("done")

replace_text_in_files('knowledge_base/origin', 'knowledge_base/clean', 'terms_map.json')