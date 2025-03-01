# coding: utf-8
import json
import os
import subprocess

from tqdm import tqdm


def get_file_line_count(file_path):
    try:
        # 使用shell命令 'wc -l' 获取文件行数
        result = subprocess.check_output(['wc', '-l', file_path], universal_newlines=True)
        
        # 解析结果，获取行数
        line_count = int(result.split()[0])
        
        return line_count
    except Exception as e:
        print(f"Error: {e}")
        return -1


def convert_to_alpaca(path: str):
    total = get_file_line_count(path)
    if total == -1:
        raise ValueError('{} error'.format(path))
    f_name = os.path.basename(path)
    dir_ = os.path.dirname(path)
    save_f = open(os.path.join(dir_, 'alpaca_{}'.format(f_name)), 'w')
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total):
            data = json.loads(line)
            instruction = data['question']
            output = data['answer']
            data['instruction'] = instruction
            data['input'] = ''
            data['output'] = output
            data.pop('question')
            data.pop('answer')
            save_f.write(json.dumps(data, ensure_ascii=False) + '\n')
    save_f.close()


if __name__ == '__main__':
    f = '/data/zhaoyuhang/data/opensource_datasets/red/15W.jsonl'
    convert_to_alpaca(f)
