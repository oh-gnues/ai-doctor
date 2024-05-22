import json

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_data(input_path, output_path):
    data = load_data(input_path)
    
    if isinstance(data, dict):
        # 데이터가 딕셔너리 형태로 저장된 경우
        key = list(data.keys())[0]
        processed_data = {"data": data[key]}
    elif isinstance(data, list):
        # 데이터가 리스트 형태로 저장된 경우
        processed_data = {"data": data}
    else:
        raise ValueError("데이터 형식이 올바르지 않습니다. 리스트 또는 딕셔너리 형식이어야 합니다.")
    
    save_data(processed_data, output_path)

if __name__ == "__main__":
    process_data('data/eng_train.json', 'data/processed_train.json')
    process_data('data/eng_test_unseen.json', 'data/processed_test_unseen.json')
    process_data('data/eng_test_multi.json', 'data/processed_test_multi.json')
    process_data('data/eng_test_single.json', 'data/processed_test_single.json')