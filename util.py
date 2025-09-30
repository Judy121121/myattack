import os
import sys
import shutil


def move_folders_with_many_images(source_dir, target_dir, threshold=220):
    """移动包含超过指定数量图片的'1'子文件夹的父文件夹"""
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    moved_folders = []
    skipped_folders = []

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录下的直接子文件夹
    for parent_folder in os.listdir(source_dir):
        parent_path = os.path.join(source_dir, parent_folder)
        target_path = os.path.join(target_dir, parent_folder)

        # 跳过非目录项
        if not os.path.isdir(parent_path):
            continue

        # 检查"1"子文件夹是否存在
        folder1_path = os.path.join(parent_path, "1")
        if not os.path.exists(folder1_path) or not os.path.isdir(folder1_path):
            skipped_folders.append(parent_folder)
            continue

        # 计算"1"文件夹中的图片数量
        image_count = 0
        for filename in os.listdir(folder1_path):
            file_path = os.path.join(folder1_path, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_exts:
                    image_count += 1
                    # 如果已经超过阈值，可以提前停止计数
                    if image_count > threshold:
                        break

        # 如果图片数量超过阈值，则移动父文件夹
        if image_count > threshold:
            # 如果目标路径已存在，先删除
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                else:
                    os.remove(target_path)

            # 移动文件夹
            shutil.move(parent_path, target_path)
            moved_folders.append((parent_folder, image_count))

    return moved_folders, skipped_folders

import numpy as np
if __name__ == "__main__":
    import numpy as np

    # 1. 加载npy文件
    data = np.load('preprocess/phoenix2014/train_info.npy', allow_pickle=True).item()  # 替换为你的文件路径
    print(data)
    # 2. 定义要删除的198个fileid列表
    fileids_to_remove = [
        '01July_2009_Wednesday_tagesschau_default-2',
        '01July_2009_Wednesday_tagesschau_default-4',
        '01July_2009_Wednesday_tagesschau_default-6',
        '01June_2010_Tuesday_tagesschau_default-2',
        '01June_2011_Wednesday_heute_default-1',
        '01May_2010_Saturday_tagesschau_default-2',
        '02December_2010_Thursday_heute_default-4',
        '02July_2009_Thursday_tagesschau_default-3',
        '02September_2010_Thursday_heute_default-1',
        '03April_2010_Saturday_tagesschau_default-7',
        '03December_2009_Thursday_tagesschau_default-2',
        '03February_2010_Wednesday_tagesschau_default-10',
        '03July_2009_Friday_tagesschau_default-7',
        '03July_2009_Friday_tagesschau_default-9',
        '03November_2009_Tuesday_tagesschau_default-4',
        '04August_2011_Thursday_tagesschau_default-3',
        '04December_2009_Friday_tagesschau_default-8',
        '04December_2009_Friday_tagesschau_default-4',
        '04January_2010_Monday_tagesschau_default-5',
        '04June_2010_Friday_tagesschau_default-2',
        '04June_2010_Friday_tagesschau_default-6',
        '04May_2010_Tuesday_heute_default-5',
        '05July_2009_Sunday_tagesschau_default-2',
        '05July_2009_Sunday_tagesschau_default-3',
        '05July_2009_Sunday_tagesschau_default-7',
        '05May_2010_Wednesday_tagesschau_default-9',
        '05October_2010_Tuesday_heute_default-1',
        '05September_2009_Saturday_tagesschau_default-7',
        '06December_2009_Sunday_tagesschau_default-2',
        '06December_2010_Monday_heute_default-6',
        '06December_2010_Monday_heute_default-9',
        '06July_2009_Monday_tagesschau_default-9',
        '06May_2010_Thursday_heute_default-3',
        '06May_2010_Thursday_tagesschau_default-8',
        '07August_2009_Friday_tagesschau_default-2',
        '07December_2010_Tuesday_heute_default-9',
        '07December_2011_Wednesday_heute_default-14',
        '07December_2011_Wednesday_tagesschau_default-2',
        '07April_2010_Wednesday_heute_default-3',
        '08December_2009_Tuesday_heute_default-1',
        '08December_2009_Tuesday_heute_default-2',
        '08December_2009_Tuesday_heute_default-5',
        '08December_2009_Tuesday_heute_default-7',
        '08December_2009_Tuesday_tagesschau_default-1',
        '08July_2010_Thursday_tagesschau_default-9',
        '08November_2010_Monday_heute_default-4',
        '09August_2009_Sunday_tagesschau_default-15',
        '09August_2010_Monday_tagesschau_default-12',
        '09December_2009_Wednesday_heute_default-3',
        '09December_2009_Wednesday_heute_default-5',
        '09December_2009_Wednesday_heute_default-6',
        '09February_2011_Wednesday_heute_default-12',
        '09July_2009_Thursday_tagesschau_default-10',
        '09July_2009_Thursday_tagesschau_default-3',
        '09June_2010_Wednesday_heute_default-11',
        '09November_2010_Tuesday_tagesschau_default-17',
        '09November_2010_Tuesday_tagesschau_default-6',
        '09September_2010_Thursday_tagesschau_default-11',
        '10December_2009_Thursday_heute_default-5',
        '10December_2009_Thursday_heute_default-6',
        '10December_2009_Thursday_tagesschau_default-4',
        '10December_2009_Thursday_tagesschau_default-8',
        '10July_2009_Friday_tagesschau_default-11',
        '10July_2009_Friday_tagesschau_default-13',
        '10May_2010_Monday_tagesschau_default-2',
        '10November_2009_Tuesday_tagesschau_default-3',
        '10November_2009_Tuesday_tagesschau_default-6',
        '10November_2010_Wednesday_tagesschau_default-7',
        '11August_2009_Tuesday_tagesschau_default-3',
        '11August_2009_Tuesday_tagesschau_default-5',
        '11December_2009_Friday_tagesschau_default-2',
        '11December_2009_Friday_tagesschau_default-4',
        '11May_2010_Tuesday_heute_default-11',
        '11September_2010_Saturday_tagesschau_default-2',
        '12April_2010_Monday_heute_default-10',
        '12August_2009_Wednesday_tagesschau_default-3',
        '12August_2009_Wednesday_tagesschau_default-4',
        '12August_2009_Wednesday_tagesschau_default-7',
        '12July_2009_Sunday_tagesschau_default-14',
        '12October_2009_Monday_tagesschau_default-4',
        '12October_2009_Monday_tagesschau_default-9',
        '12September_2009_Saturday_tagesschau_default-2',
        '12September_2009_Saturday_tagesschau_default-5',
        '12November_2009_Thursday_tagesschau_default-2',
        '12September_2009_Saturday_tagesschau_default-6',
        '13December_2009_Sunday_tagesschau_default-3',
        '13December_2009_Sunday_tagesschau_default-4',
        '13December_2009_Sunday_tagesschau_default-9',
        '13February_2010_Saturday_tagesschau_default-2',
        '13July_2009_Monday_tagesschau_default-7',
        '13July_2009_Monday_tagesschau_default-8',
        '13May_2010_Thursday_tagesschau_default-7',
        '13November_2009_Friday_tagesschau_default-2',
        '13November_2009_Friday_tagesschau_default-4',
        '14December_2009_Monday_tagesschau_default-7',
        '14February_2010_Sunday_tagesschau_default-6',
        '14July_2009_Tuesday_tagesschau_default-2',
        '14July_2009_Tuesday_tagesschau_default-3',
        '14July_2009_Tuesday_tagesschau_default-5',
        '14July_2010_Wednesday_heute_default-8',
        '14May_2010_Friday_tagesschau_default-4',
        '14October_2009_Wednesday_tagesschau_default-5',
        '14September_2010_Tuesday_heute_default-10',
        '15December_2009_Tuesday_tagesschau_default-12',
        '16July_2009_Thursday_tagesschau_default-14',
        '16November_2009_Monday_tagesschau_default-4',
        '17July_2009_Friday_tagesschau_default-5',
        '17May_2010_Monday_tagesschau_default-5',
        '17September_2010_Friday_tagesschau_default-4',
        '18April_2010_Sunday_tagesschau_default-3',
        '18July_2009_Saturday_tagesschau_default-4',
        '18July_2009_Saturday_tagesschau_default-8',
        '18June_2010_Friday_tagesschau_default-2',
        '18May_2010_Tuesday_heute_default-10',
        '18May_2010_Tuesday_heute_default-4',
        '18May_2010_Tuesday_heute_default-5',
        '18November_2009_Wednesday_tagesschau_default-9',
        '18November_2011_Friday_tagesschau_default-6',
        '18October_2009_Sunday_tagesschau_default-4',
        '18October_2009_Sunday_tagesschau_default-8',
        '19February_2011_Saturday_tagesschau_default-2',
        '19July_2009_Sunday_tagesschau_default-9',
        '19May_2010_Wednesday_heute_default-2',
        '19November_2011_Saturday_tagesschau_default-17',
        '19October_2009_Monday_tagesschau_default-3',
        '19October_2010_Tuesday_heute_default-10',
        '19October_2010_Tuesday_heute_default-2',
        '20December_2010_Monday_tagesschau_default-18',
        '20November_2009_Friday_tagesschau_default-8',
        '14August_2009_Friday_tagesschau_default-3',
        '20October_2010_Wednesday_heute_default-1',
        '20October_2010_Wednesday_heute_default-2',
        '20October_2011_Thursday_tagesschau_default-13',
        '20September_2010_Monday_heute_default-0',
        '21August_2011_Sunday_tagesschau_default-12',
        '21July_2009_Tuesday_tagesschau_default-6',
        '21March_2011_Monday_tagesschau_default-14',
        '21November_2009_Saturday_tagesschau_default-8',
        '21October_2011_Friday_tagesschau_default-11',
        '22February_2010_Monday_tagesschau_default-2',
        '22July_2009_Wednesday_tagesschau_default-6',
        '22July_2009_Wednesday_tagesschau_default-7',
        '22July_2010_Thursday_heute_default-11',
        '22July_2010_Thursday_heute_default-7',
        '22July_2010_Thursday_tagesschau_default-11',
        '22March_2011_Tuesday_tagesschau_default-3',
        '22October_2009_Thursday_tagesschau_default-2',
        '22November_2011_Tuesday_tagesschau_default-9',
        '23February_2011_Wednesday_heute_default-9',
        '23February_2011_Wednesday_tagesschau_default-10',
        '23July_2009_Thursday_tagesschau_default-6',
        '23September_2009_Wednesday_tagesschau_default-6',
        '24August_2009_Monday_heute_default-6',
        '24August_2010_Tuesday_heute_default-5',
        '24February_2011_Thursday_tagesschau_default-2',
        '24September_2009_Thursday_heute_default-4',
        '25August_2009_Tuesday_heute_default-2',
        '25August_2009_Tuesday_heute_default-5',
        '25August_2009_Tuesday_tagesschau_default-5',
        '25January_2010_Monday_heute_default-12',
        '25January_2011_Tuesday_tagesschau_default-8',
        '25November_2009_Wednesday_tagesschau_default-6',
        '26April_2010_Monday_heute_default-10',
        '26August_2009_Wednesday_heute_default-10',
        '26February_2011_Saturday_tagesschau_default-16',
        '26May_2010_Wednesday_heute_default-1',
        '27April_2010_Tuesday_heute_default-8',
        '27August_2009_Thursday_tagesschau_default-9',
        '27November_2009_Friday_tagesschau_default-4',
        '27November_2009_Friday_tagesschau_default-6',
        '26September_2010_Sunday_tagesschau_default-7',
        '27February_2010_Saturday_tagesschau_default-2',
        '27September_2009_Sunday_tagesschau_default-3',
        '28August_2009_Friday_tagesschau_default-5',
        '28February_2011_Monday_tagesschau_default-5',
        '28January_2010_Thursday_heute_default-9',
        '28September_2010_Tuesday_tagesschau_default-2',
        '28September_2010_Tuesday_tagesschau_default-5',
        '29August_2009_Saturday_tagesschau_default-4',
        '29November_2011_Tuesday_heute_default-0',
        '29November_2011_Tuesday_heute_default-1',
        '29November_2011_Tuesday_heute_default-8',
        '29September_2010_Wednesday_tagesschau_default-9',
        '29September_2011_Thursday_heute_default-8',
        '29September_2011_Thursday_heute_default-9',
        '29May_2010_Saturday_tagesschau_default-13',
        '29July_2010_Thursday_heute_default-2',
        '30June_2009_Tuesday_tagesschau_default-2',
        '30March_2010_Tuesday_heute_default-11',
        '30May_2010_Sunday_tagesschau_default-2',
        '30May_2010_Sunday_tagesschau_default-5',
        '30May_2011_Monday_heute_default-3',
        '30June_2009_Tuesday_tagesschau_default-9',
        '30November_2010_Tuesday_tagesschau_default-9',
        '31August_2010_Tuesday_tagesschau_default-7',
        '30October_2009_Friday_tagesschau_default-2',
        '30September_2009_Wednesday_tagesschau_default-7',
        '31July_2010_Saturday_tagesschau_default-7'
    ]

    # 3. 创建新数据结构（保留prefix）
    new_data = {'prefix': data['prefix']}

    # 4. 过滤并重新编号有效条目
    new_index = 0
    for idx, entry in data.items():
        if isinstance(idx, int):  # 只处理数字索引的条目
            if entry['fileid'] not in fileids_to_remove:
                new_data[new_index] = entry
                new_index += 1

    # 5. 保存处理后的数据
    np.save('processed_data.npy', new_data)

    print(f"原始条目数: {len([k for k in data if isinstance(k, int)])}")
    print(f"处理后条目数: {len([k for k in new_data if isinstance(k, int)])}")
    print(f"已删除: {len(fileids_to_remove)}个条目")






    # # 设置源目录和目标目录
    # source_dir = "/media/cv3/store1/postgraduate/y2023/WLF/datasets/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px/train"
    # target_dir = "/media/cv3/store1/postgraduate/y2023/WLF/datasets/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px/temptrain"
    #
    # # 验证源目录
    # if not os.path.isdir(source_dir):
    #     print(f"错误: 源目录 '{source_dir}' 不存在或不是目录")
    #     sys.exit(1)
    #
    # print(f"源目录: {source_dir}")
    # print(f"目标目录: {target_dir}")
    # print(f"阈值: 220张图片")
    #
    # # 执行移动操作
    # moved, skipped = move_folders_with_many_images(source_dir, target_dir)
    #
    # # 输出结果
    # if moved:
    #     print("\n已移动以下文件夹 (图片数量):")
    #     for folder, count in moved:
    #         print(f" - {folder} ({count}张图片)")
    #     print(f"\n总共移动了 {len(moved)} 个文件夹")
    # else:
    #     print("\n没有符合条件的文件夹需要移动")
    #
    # if skipped:
    #     print(f"\n跳过了 {len(skipped)} 个没有'1'子文件夹的目录")
    #
    # # 统计剩余文件夹数量
    # remaining = len(os.listdir(source_dir))
    # print(f"\n源目录剩余文件夹数量: {remaining}")
    #



    '''
    /home/cv3/anaconda3/envs/attack/bin/python /home/cv3/pycharm-community-2022.3.3/plugins/python-ce/helpers/pydev/pydevd.py --multiprocess --qt-support=auto --client 127.0.0.1 --port 37467 --file /media/cv3/store1/postgraduate/y2023/WLF/EGS-TSSA-main/util.py 
已连接到 pydev 调试器(内部版本号 223.8836.43)源目录: /media/cv3/store1/postgraduate/y2023/WLF/datasets/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px/train
目标目录: /media/cv3/store1/postgraduate/y2023/WLF/datasets/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px/temptrain
阈值: 220张图片

已移动以下文件夹 (图片数量):
 - 01July_2009_Wednesday_tagesschau_default-2 (221张图片)
 - 01July_2009_Wednesday_tagesschau_default-4 (221张图片)
 - 01July_2009_Wednesday_tagesschau_default-6 (221张图片)
 - 01June_2010_Tuesday_tagesschau_default-2 (221张图片)
 - 01June_2011_Wednesday_heute_default-1 (221张图片)
 - 01May_2010_Saturday_tagesschau_default-2 (221张图片)
 - 02December_2010_Thursday_heute_default-4 (221张图片)
 - 02July_2009_Thursday_tagesschau_default-3 (221张图片)
 - 02September_2010_Thursday_heute_default-1 (221张图片)
 - 03April_2010_Saturday_tagesschau_default-7 (221张图片)
 - 03December_2009_Thursday_tagesschau_default-2 (221张图片)
 - 03February_2010_Wednesday_tagesschau_default-10 (221张图片)
 - 03July_2009_Friday_tagesschau_default-7 (221张图片)
 - 03July_2009_Friday_tagesschau_default-9 (221张图片)
 - 03November_2009_Tuesday_tagesschau_default-4 (221张图片)
 - 04August_2011_Thursday_tagesschau_default-3 (221张图片)
 - 04December_2009_Friday_tagesschau_default-8 (221张图片)
 - 04December_2009_Friday_tagesschau_default-4 (221张图片)
 - 04January_2010_Monday_tagesschau_default-5 (221张图片)
 - 04June_2010_Friday_tagesschau_default-2 (221张图片)
 - 04June_2010_Friday_tagesschau_default-6 (221张图片)
 - 04May_2010_Tuesday_heute_default-5 (221张图片)
 - 05July_2009_Sunday_tagesschau_default-2 (221张图片)
 - 05July_2009_Sunday_tagesschau_default-3 (221张图片)
 - 05July_2009_Sunday_tagesschau_default-7 (221张图片)
 - 05May_2010_Wednesday_tagesschau_default-9 (221张图片)
 - 05October_2010_Tuesday_heute_default-1 (221张图片)
 - 05September_2009_Saturday_tagesschau_default-7 (221张图片)
 - 06December_2009_Sunday_tagesschau_default-2 (221张图片)
 - 06December_2010_Monday_heute_default-6 (221张图片)
 - 06December_2010_Monday_heute_default-9 (221张图片)
 - 06July_2009_Monday_tagesschau_default-9 (221张图片)
 - 06May_2010_Thursday_heute_default-3 (221张图片)
 - 06May_2010_Thursday_tagesschau_default-8 (221张图片)
 - 07August_2009_Friday_tagesschau_default-2 (221张图片)
 - 07December_2010_Tuesday_heute_default-9 (221张图片)
 - 07December_2011_Wednesday_heute_default-14 (221张图片)
 - 07December_2011_Wednesday_tagesschau_default-2 (221张图片)
 - 07April_2010_Wednesday_heute_default-3 (221张图片)
 - 08December_2009_Tuesday_heute_default-1 (221张图片)
 - 08December_2009_Tuesday_heute_default-2 (221张图片)
 - 08December_2009_Tuesday_heute_default-5 (221张图片)
 - 08December_2009_Tuesday_heute_default-7 (221张图片)
 - 08December_2009_Tuesday_tagesschau_default-1 (221张图片)
 - 08July_2010_Thursday_tagesschau_default-9 (221张图片)
 - 08November_2010_Monday_heute_default-4 (221张图片)
 - 09August_2009_Sunday_tagesschau_default-15 (221张图片)
 - 09August_2010_Monday_tagesschau_default-12 (221张图片)
 - 09December_2009_Wednesday_heute_default-3 (221张图片)
 - 09December_2009_Wednesday_heute_default-5 (221张图片)
 - 09December_2009_Wednesday_heute_default-6 (221张图片)
 - 09February_2011_Wednesday_heute_default-12 (221张图片)
 - 09July_2009_Thursday_tagesschau_default-10 (221张图片)
 - 09July_2009_Thursday_tagesschau_default-3 (221张图片)
 - 09June_2010_Wednesday_heute_default-11 (221张图片)
 - 09November_2010_Tuesday_tagesschau_default-17 (221张图片)
 - 09November_2010_Tuesday_tagesschau_default-6 (221张图片)
 - 09September_2010_Thursday_tagesschau_default-11 (221张图片)
 - 10December_2009_Thursday_heute_default-5 (221张图片)
 - 10December_2009_Thursday_heute_default-6 (221张图片)
 - 10December_2009_Thursday_tagesschau_default-4 (221张图片)
 - 10December_2009_Thursday_tagesschau_default-8 (221张图片)
 - 10July_2009_Friday_tagesschau_default-11 (221张图片)
 - 10July_2009_Friday_tagesschau_default-13 (221张图片)
 - 10May_2010_Monday_tagesschau_default-2 (221张图片)
 - 10November_2009_Tuesday_tagesschau_default-3 (221张图片)
 - 10November_2009_Tuesday_tagesschau_default-6 (221张图片)
 - 10November_2010_Wednesday_tagesschau_default-7 (221张图片)
 - 11August_2009_Tuesday_tagesschau_default-3 (221张图片)
 - 11August_2009_Tuesday_tagesschau_default-5 (221张图片)
 - 11December_2009_Friday_tagesschau_default-2 (221张图片)
 - 11December_2009_Friday_tagesschau_default-4 (221张图片)
 - 11May_2010_Tuesday_heute_default-11 (221张图片)
 - 11September_2010_Saturday_tagesschau_default-2 (221张图片)
 - 12April_2010_Monday_heute_default-10 (221张图片)
 - 12August_2009_Wednesday_tagesschau_default-3 (221张图片)
 - 12August_2009_Wednesday_tagesschau_default-4 (221张图片)
 - 12August_2009_Wednesday_tagesschau_default-7 (221张图片)
 - 12July_2009_Sunday_tagesschau_default-14 (221张图片)
 - 12October_2009_Monday_tagesschau_default-4 (221张图片)
 - 12October_2009_Monday_tagesschau_default-9 (221张图片)
 - 12September_2009_Saturday_tagesschau_default-2 (221张图片)
 - 12September_2009_Saturday_tagesschau_default-5 (221张图片)
 - 12November_2009_Thursday_tagesschau_default-2 (221张图片)
 - 12September_2009_Saturday_tagesschau_default-6 (221张图片)
 - 13December_2009_Sunday_tagesschau_default-3 (221张图片)
 - 13December_2009_Sunday_tagesschau_default-4 (221张图片)
 - 13December_2009_Sunday_tagesschau_default-9 (221张图片)
 - 13February_2010_Saturday_tagesschau_default-2 (221张图片)
 - 13July_2009_Monday_tagesschau_default-7 (221张图片)
 - 13July_2009_Monday_tagesschau_default-8 (221张图片)
 - 13May_2010_Thursday_tagesschau_default-7 (221张图片)
 - 13November_2009_Friday_tagesschau_default-2 (221张图片)
 - 13November_2009_Friday_tagesschau_default-4 (221张图片)
 - 14December_2009_Monday_tagesschau_default-7 (221张图片)
 - 14February_2010_Sunday_tagesschau_default-6 (221张图片)
 - 14July_2009_Tuesday_tagesschau_default-2 (221张图片)
 - 14July_2009_Tuesday_tagesschau_default-3 (221张图片)
 - 14July_2009_Tuesday_tagesschau_default-5 (221张图片)
 - 14July_2010_Wednesday_heute_default-8 (221张图片)
 - 14May_2010_Friday_tagesschau_default-4 (221张图片)
 - 14October_2009_Wednesday_tagesschau_default-5 (221张图片)
 - 14September_2010_Tuesday_heute_default-10 (221张图片)
 - 15December_2009_Tuesday_tagesschau_default-12 (221张图片)
 - 16July_2009_Thursday_tagesschau_default-14 (221张图片)
 - 16November_2009_Monday_tagesschau_default-4 (221张图片)
 - 17July_2009_Friday_tagesschau_default-5 (221张图片)
 - 17May_2010_Monday_tagesschau_default-5 (221张图片)
 - 17September_2010_Friday_tagesschau_default-4 (221张图片)
 - 18April_2010_Sunday_tagesschau_default-3 (221张图片)
 - 18July_2009_Saturday_tagesschau_default-4 (221张图片)
 - 18July_2009_Saturday_tagesschau_default-8 (221张图片)
 - 18June_2010_Friday_tagesschau_default-2 (221张图片)
 - 18May_2010_Tuesday_heute_default-10 (221张图片)
 - 18May_2010_Tuesday_heute_default-4 (221张图片)
 - 18May_2010_Tuesday_heute_default-5 (221张图片)
 - 18November_2009_Wednesday_tagesschau_default-9 (221张图片)
 - 18November_2011_Friday_tagesschau_default-6 (221张图片)
 - 18October_2009_Sunday_tagesschau_default-4 (221张图片)
 - 18October_2009_Sunday_tagesschau_default-8 (221张图片)
 - 19February_2011_Saturday_tagesschau_default-2 (221张图片)
 - 19July_2009_Sunday_tagesschau_default-9 (221张图片)
 - 19May_2010_Wednesday_heute_default-2 (221张图片)
 - 19November_2011_Saturday_tagesschau_default-17 (221张图片)
 - 19October_2009_Monday_tagesschau_default-3 (221张图片)
 - 19October_2010_Tuesday_heute_default-10 (221张图片)
 - 19October_2010_Tuesday_heute_default-2 (221张图片)
 - 20December_2010_Monday_tagesschau_default-18 (221张图片)
 - 20November_2009_Friday_tagesschau_default-8 (221张图片)
 - 14August_2009_Friday_tagesschau_default-3 (221张图片)
 - 20October_2010_Wednesday_heute_default-1 (221张图片)
 - 20October_2010_Wednesday_heute_default-2 (221张图片)
 - 20October_2011_Thursday_tagesschau_default-13 (221张图片)
 - 20September_2010_Monday_heute_default-0 (221张图片)
 - 21August_2011_Sunday_tagesschau_default-12 (221张图片)
 - 21July_2009_Tuesday_tagesschau_default-6 (221张图片)
 - 21March_2011_Monday_tagesschau_default-14 (221张图片)
 - 21November_2009_Saturday_tagesschau_default-8 (221张图片)
 - 21October_2011_Friday_tagesschau_default-11 (221张图片)
 - 22February_2010_Monday_tagesschau_default-2 (221张图片)
 - 22July_2009_Wednesday_tagesschau_default-6 (221张图片)
 - 22July_2009_Wednesday_tagesschau_default-7 (221张图片)
 - 22July_2010_Thursday_heute_default-11 (221张图片)
 - 22July_2010_Thursday_heute_default-7 (221张图片)
 - 22July_2010_Thursday_tagesschau_default-11 (221张图片)
 - 22March_2011_Tuesday_tagesschau_default-3 (221张图片)
 - 22October_2009_Thursday_tagesschau_default-2 (221张图片)
 - 22November_2011_Tuesday_tagesschau_default-9 (221张图片)
 - 23February_2011_Wednesday_heute_default-9 (221张图片)
 - 23February_2011_Wednesday_tagesschau_default-10 (221张图片)
 - 23July_2009_Thursday_tagesschau_default-6 (221张图片)
 - 23September_2009_Wednesday_tagesschau_default-6 (221张图片)
 - 24August_2009_Monday_heute_default-6 (221张图片)
 - 24August_2010_Tuesday_heute_default-5 (221张图片)
 - 24February_2011_Thursday_tagesschau_default-2 (221张图片)
 - 24September_2009_Thursday_heute_default-4 (221张图片)
 - 25August_2009_Tuesday_heute_default-2 (221张图片)
 - 25August_2009_Tuesday_heute_default-5 (221张图片)
 - 25August_2009_Tuesday_tagesschau_default-5 (221张图片)
 - 25January_2010_Monday_heute_default-12 (221张图片)
 - 25January_2011_Tuesday_tagesschau_default-8 (221张图片)
 - 25November_2009_Wednesday_tagesschau_default-6 (221张图片)
 - 26April_2010_Monday_heute_default-10 (221张图片)
 - 26August_2009_Wednesday_heute_default-10 (221张图片)
 - 26February_2011_Saturday_tagesschau_default-16 (221张图片)
 - 26May_2010_Wednesday_heute_default-1 (221张图片)
 - 27April_2010_Tuesday_heute_default-8 (221张图片)
 - 27August_2009_Thursday_tagesschau_default-9 (221张图片)
 - 27November_2009_Friday_tagesschau_default-4 (221张图片)
 - 27November_2009_Friday_tagesschau_default-6 (221张图片)
 - 26September_2010_Sunday_tagesschau_default-7 (221张图片)
 - 27February_2010_Saturday_tagesschau_default-2 (221张图片)
 - 27September_2009_Sunday_tagesschau_default-3 (221张图片)
 - 28August_2009_Friday_tagesschau_default-5 (221张图片)
 - 28February_2011_Monday_tagesschau_default-5 (221张图片)
 - 28January_2010_Thursday_heute_default-9 (221张图片)
 - 28September_2010_Tuesday_tagesschau_default-2 (221张图片)
 - 28September_2010_Tuesday_tagesschau_default-5 (221张图片)
 - 29August_2009_Saturday_tagesschau_default-4 (221张图片)
 - 29November_2011_Tuesday_heute_default-0 (221张图片)
 - 29November_2011_Tuesday_heute_default-1 (221张图片)
 - 29November_2011_Tuesday_heute_default-8 (221张图片)
 - 29September_2010_Wednesday_tagesschau_default-9 (221张图片)
 - 29September_2011_Thursday_heute_default-8 (221张图片)
 - 29September_2011_Thursday_heute_default-9 (221张图片)
 - 29May_2010_Saturday_tagesschau_default-13 (221张图片)
 - 29July_2010_Thursday_heute_default-2 (221张图片)
 - 30June_2009_Tuesday_tagesschau_default-2 (221张图片)
 - 30March_2010_Tuesday_heute_default-11 (221张图片)
 - 30May_2010_Sunday_tagesschau_default-2 (221张图片)
 - 30May_2010_Sunday_tagesschau_default-5 (221张图片)
 - 30May_2011_Monday_heute_default-3 (221张图片)
 - 30June_2009_Tuesday_tagesschau_default-9 (221张图片)
 - 30November_2010_Tuesday_tagesschau_default-9 (221张图片)
 - 31August_2010_Tuesday_tagesschau_default-7 (221张图片)
 - 30October_2009_Friday_tagesschau_default-2 (221张图片)
 - 30September_2009_Wednesday_tagesschau_default-7 (221张图片)
 - 31July_2010_Saturday_tagesschau_default-7 (221张图片)

总共移动了 198 个文件夹

源目录剩余文件夹数量: 5474

进程已结束,退出代码0

    '''