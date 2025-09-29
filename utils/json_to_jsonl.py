import json

def convert_json_to_jsonl(input_json_path, output_jsonl_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data_list = data
    else:
        data_list = [data]
    
    fields_to_remove = {'transposed_table', 'sampled_indices', 'row_shuffled_table', 'row_shuffled_transposed_table', 'questions', 'answers', 'ids'}
    
    idx_counter = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for single_dict in data_list:
            questions = single_dict.get('questions', [])
            answers = single_dict.get('answers', [])
            ids = single_dict.get('ids', [])
            
            common_data = {k: v for k, v in single_dict.items() if k not in fields_to_remove}
            
            for question, answer, record_id in zip(questions, answers, ids):
                record = {
                    'idx': idx_counter,
                    **common_data,
                    'question': question,
                    'answer': answer,
                    'id': record_id
                }
                
                json_line = json.dumps(record, ensure_ascii=True, separators=(',', ':'))
                f.write(json_line + '\n')
                idx_counter += 1
    
    print(f"변환 완료: {output_jsonl_path}")

def main():
    input_json = "/home/wooo519/tableReasoningFinal/MINE_Final/MyMethod_final_one_last/datasets/wtq.json"
    output_jsonl = "/home/wooo519/tableReasoningFinal/MINE_Final/MyMethod_final_one_last/datasets/wtq.jsonl"
    convert_json_to_jsonl(input_json, output_jsonl)

if __name__ == "__main__":
    main()
