# coding: utf-8
import typing as t
import random
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

random.seed(13)


def combine_with_red_data(
        red_data_path: str,
        data_paths: t.List[str],
        data_config: t.Dict,
        output_path: str):
    '''
    以红色数据为基础，将其他数据插入到红色数据中，并按照指定的概率进行插入。
    red_data_path: 红色数据的路径
    data_paths: 其他数据的路径列表
    data_config: 数据配置, save floder to interleave_probs
    output_path: 合并后的数据保存的路径
    '''
    red_data = open(red_data_path).readlines()
    red_data_num = len(red_data)
    print('Load red data successfully. Red data num: ', red_data_num)

    # 读取其他数据
    all_datas = [open(data_path).readlines() for data_path in data_paths]
    print('Load other data successfully.')

    for save_floder, interleave_probs in tqdm(data_config.items()):
        combined_data = []
        combined_data.extend(red_data)
        red_data_prob = interleave_probs['red_data_prob']
        data_prob = interleave_probs['data_prob']
        assert len(all_datas) == len(interleave_probs), 'Dataset num not match'

        total_sample_num = int(red_data_num / red_data_prob)
        for data, prob in zip(all_datas, data_prob):
            if prob == 'all':
                combined_data.extend(data)
            else:
                prob = float(prob)
                interleaved_data = random.sample(data, int(total_sample_num * prob))
                combined_data.extend(interleaved_data)
        
        random.shuffle(combined_data)

        save_floder = os.path.join(output_path, save_floder)
        os.makedirs(save_floder, exist_ok=True)
        with open(os.path.join(save_floder, 'train.jsonl'), 'w') as f:
            for line in combined_data:
                line = line.strip()
                f.write(line + '\n')

        print('Data combined successfully. Train num: ', len(combined_data), total_sample_num)


if __name__ == '__main__':
    red_data_path = '/data/zhaoyuhang/data/opensource_datasets/red/alpaca_15W.jsonl'
    data_paths = [
        '/data/zhaoyuhang/data/opensource_datasets/Identity/identity.json',
        '/data/zhaoyuhang/data/opensource_datasets/deepctrl-sft-data/sft_data.jsonl',
    ]

    save_root = '/data/zhaoyuhang/data/sft/gm-llm-v2.0'

    data_config = {}
    for red_prob, floder in zip([0.5, 0.33, 0.17, 0.09], ['one2one', 'one2two', 'one2five', 'one2ten']):
        interleave_probs = ['all', 1 - red_prob]
        data_config[floder] = {'red_data_prob': red_prob, 'data_prob': interleave_probs}
    combine_with_red_data(red_data_path, data_paths, data_config, save_root)
