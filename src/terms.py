import json
from faker import Faker

# Use a valid Faker locale
fake = Faker('en_US')

def generate_random_name():
    return fake.name()

with open('terms_map_origin.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def replace_names(obj):
    if isinstance(obj, dict):
        for key, _ in obj.items():
            obj[key] = generate_random_name()

replace_names(data)

with open('terms_map.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("done")