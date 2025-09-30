import numpy as np
import os
path=os.listdir('/media/cv3/store1/postgraduate/y2023/WLF/datasets/ph_adv/newAttack/dev/')
# 1. 加载npy文件
data = np.load('dev_data.npy', allow_pickle=True).item()  # 替换为你的文件路径
print(data)

# 3. 创建新数据结构（保留prefix）
new_data = {'prefix': data['prefix']}

# 4. 过滤并重新编号有效条目
new_index = 0
for idx, entry in data.items():
    if isinstance(idx, int):  # 只处理数字索引的条目
        if entry['fileid']  in path:
            new_data[new_index] = entry
            new_index += 1

# 5. 保存处理后的数据
np.save('dev_data.npy', new_data)


