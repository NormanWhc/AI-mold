import json
import re
import glob
def extract_stp_data(stp_path):
    with open(stp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取DATA部分内容
    data_section = re.search(r'DATA;(.*?)ENDSEC;', content, re.DOTALL)
    if not data_section:
        return ""
        
    data_content = data_section.group(1)
    return data_content
    # # 提取所有实体定义，保留原始换行格式
    # entities = []
    # for line in data_content.split('\n'):
    #     line = line.strip()
    #     if line.startswith('#'):
    #         entities.append(line)
    # return '\n'.join(entities)

all_entries = []
for prt_path in glob.glob('data/ai_mo/A*-PRT.stp'):
    # 提取基础编号（如A0001）
    base_number = re.search(r'A\d+', prt_path).group()
    
    # 构建对应的MOLD文件路径
    mold_path = f'data/ai_mo/{base_number}-MOLD.stp'
    
    try:
        prt_data = extract_stp_data(prt_path)
        mold_data = extract_stp_data(mold_path)
        
        all_entries.append({
            "instruction": f"以下是产品图，请根据产品图，输出模具图",
            "input": prt_data,
            "output": mold_data
        })
    except Exception as e:
        print(f"处理{base_number}时出错: {str(e)}")
        continue

# 读取并更新module.json
with open('data/ai_mo/module.json', 'r+', encoding='utf-8') as f:
    # data = json.load(f)
    # data.extend(all_entries)
    f.seek(0)
    json.dump(all_entries, f, ensure_ascii=False, indent=4)
    f.truncate()

print(f"成功处理{len(all_entries)}对STP文件")