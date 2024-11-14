import subprocess
import sys
import re
from pymol import cmd
import os
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def calculate_tmscore(pdb_file1, pdb_file2):
    """
    使用 US-align 计算两个蛋白质结构之间的 TM-score。

    参数：
    pdb_file1: 第一个 PDB 文件路径
    pdb_file2: 第二个 PDB 文件路径
    """

    # 构造 US-align 命令
    command = ['USalign', pdb_file1, pdb_file2]

    try:
        # 执行命令并捕获输出
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 检查是否成功执行
        if result.returncode != 0:
            print(f'Error running US-align: {result.stderr}')
            return

        # 解析输出，提取 TM-score 值
        output = result.stdout

        # 使用正则表达式提取 TM-score
        tm_score_search = re.search(r'TM-score=\s*(\d+\.\d+)', output)
        if tm_score_search:
            tm_score = float(tm_score_search.group(1))
            return round(tm_score, 4)
        else:
            print('TM-score not found in US-align output.')
            return None

    except FileNotFoundError:
        print('USalign executable not found. Please ensure USalign is installed and added to your PATH.')


def calculate_rmsd(pdb_file1, pdb_file2, selection='name CA', mobile_obj='mol1', target_obj='mol2'):
    """
    计算两个蛋白质结构之间的 RMSD。

    参数：
    pdb_file1: 第一个 PDB 文件路径（移动结构）
    pdb_file2: 第二个 PDB 文件路径（目标结构）
    selection: 对齐和计算 RMSD 的原子选择（默认只使用 Cα 原子）
    mobile_obj: 加载第一个 PDB 文件时的对象名称
    target_obj: 加载第二个 PDB 文件时的对象名称
    """

    # 加载 PDB 文件
    cmd.load(pdb_file1, mobile_obj)
    cmd.load(pdb_file2, target_obj)

    # 进行刚性对齐
    alignment_rms = cmd.align(f'{mobile_obj} and {selection}', f'{target_obj} and {selection}')

    # 获取对齐后的 RMSD 值
    rmsd_value = alignment_rms[0]
    # 可选：保存对齐后的结构
    # cmd.save('aligned_structure.pse')

    # 清除加载的对象
    cmd.delete('all')

    return round(rmsd_value, 4)


# Function to process each (pdb1, pdb2) pair
def process_pair(pdb1, pdb2):
    try:
        tmsc = calculate_tmscore(pdb1, pdb2)
        rmsd = calculate_rmsd(pdb1, pdb2)
    except:
        tmsc, rmsd = 0, 1000
    if tmsc is None:
        tmsc = 0
    if rmsd is None:
        rmsd = 1000
    return (pdb2, tmsc, rmsd)

# Function to process each pdb1 with a list of pdb2s
def process_pdb1(pdb1, pdb2_list):
    sc_list = [process_pair(pdb1, pdb2) for pdb2 in pdb2_list]
    top_1 = sorted(sc_list, key=lambda x: x[1], reverse=True)
    top_1 = [x for x in top_1 if x[2] > 2][0]
    return [os.path.basename(pdb1), top_1[1], top_1[2]]

# Main function to use ProcessPoolExecutor
def run_multiprocessing(pdb1_list, pdb2_list):
    results = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(process_pdb1, pdb1, pdb2_list) for pdb1 in pdb1_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
            with open('output.log', 'a+') as f:
                f.write(f'{len(results)}/{len(pdb1_list)}\n')
    return results


def calculate_innersim():
    with open('results/split/valid_248', 'r') as f:
        test_248 = [os.path.join('data/test_data',line.strip()) for line in f.readlines()]
    with open('results/split/valid_hme', 'r') as f:
        test_493 = [os.path.join('data/test_data',line.strip()) for line in f.readlines()]
    df_test_248 = pd.read_csv('results/simscore/test248_simscore.csv')
    df_test_493 = pd.read_csv('results/simscore/test493_simscore.csv')

    new_lines = []
    for fn in os.listdir('results/foldseek_search/test248_train7k'):
        with open(f'results/foldseek_search/test248_train7k/{fn}', 'r') as f:
            lines = f.readlines()
        if lines:
            lines = [line.strip().split('\t') for line in lines]
            lines = sorted(lines, key=lambda x: int(x[4]), reverse=True)
            new_lines.append(lines[0][1][:4])
    train_7k = []
    for l in new_lines:
        for n in os.listdir('data/cath43_filter_pdb'):
            if l in n:
                train_7k.append(os.path.join('data/cath43_filter_pdb', n))

    new_lines = []
    for fn in os.listdir('results/foldseek_search/test493_train29k'):
        with open(f'results/foldseek_search/test493_train29k/{fn}', 'r') as f:
            lines = f.readlines()
        if lines:
            lines = [line.strip().split('\t') for line in lines]
            lines = sorted(lines, key=lambda x: int(x[4]), reverse=True)
            new_lines.append(lines[0][1][:4])
    train_29k = []
    for l in new_lines:
        for n in os.listdir('data/cath43_filter_pdb'):
            if l in n:
                train_29k.append(os.path.join('data/cath43_filter_pdb', n))

    train_4m = []
    for fn in os.listdir('results/foldseek_search/test493_train4m'):
        with open(f'results/foldseek_search/test493_train4m/{fn}', 'r') as f:
            lines = f.readlines()
        if lines:
            lines = [line.strip().split('\t') for line in lines]
            lines = sorted(lines, key=lambda x: int(x[4]), reverse=True)
            train_4m.append(os.path.join('afdb_v4', lines[0][1]+'.pdb'))

    train_80k = []
    for fn in os.listdir('results/foldseek_search/test248_train80k'):
        with open(f'results/foldseek_search/test248_train80k/{fn}', 'r') as f:
            lines = f.readlines()
        if lines:
            lines = [line.strip().split('\t') for line in lines]
            lines = sorted(lines, key=lambda x: int(x[4]), reverse=True)
            train_80k.append(os.path.join('afdb_v4', lines[0][1]+'.pdb'))



    test248_train7k = run_multiprocessing(test_248, train_7k)
    df_new = pd.DataFrame(test248_train7k, columns=['file_name', 'tmscore(7k)', 'rmsd(7k)'])
    df_test_248 = pd.merge(df_test_248, df_new, on='file_name', how='left')
    df_test_248.to_csv('results/test248_simscore.csv', index=False)


    test248_train80k = run_multiprocessing(test_248, train_80k)
    df_new = pd.DataFrame(test248_train80k, columns=['file_name', 'tmscore(80k)', 'rmsd(80k)'])
    df_test_248 = pd.merge(df_test_248, df_new, on='file_name', how='left')
    df_test_248 = df_test_248[['file_name', 'seq_sim(80k)', 'tmscore(80k)', 'rmsd(80k)', 'seq_sim(7k)', 'tmscore(7k)', 'rmsd(7k)']]
    df_test_248.to_csv('results/test248_simscore.csv', index=False)


    test493_train29k = run_multiprocessing(test_493, train_29k)
    df_new = pd.DataFrame(test493_train29k, columns=['file_name', 'tmscore(29k)', 'rmsd(29k)'])
    df_test_493 = pd.merge(df_test_493, df_new, on='file_name', how='left')
    df_test_493.to_csv('results/test493_simscore.csv', index=False)


    test493_train4m = run_multiprocessing(test_493, train_4m)
    df_new = pd.DataFrame(test493_train4m, columns=['file_name', 'tmscore(4m)', 'rmsd(4m)'])
    df_test_493 = pd.merge(df_test_493, df_new, on='file_name', how='left')
    df_test_493 = df_test_493[['file_name', 'seq_sim(4m)', 'tmscore(4m)', 'rmsd(4m)', 'seq_sim(29k)', 'tmscore(29k)', 'rmsd(29k)']]
    df_test_493.to_csv('results/test493_simscore_new.csv', index=False)


def calculate_baseline():
    for tb in os.listdir('results/baseline'):
        df = pd.read_csv(os.path.join('results/baseline', tb))
        pred_folder = '/msralaphilly2/ml-la/v-yantingli/pred_struct/' + tb.replace('.csv', '.output')
        tmsc = []
        rmsd = []
        for file_name in tqdm(df['file_name']):
            pdb1 = os.path.join('data/test_data', file_name)
            pdb2_list = [os.path.join(pred_folder, nm) for nm in os.listdir(pred_folder) if file_name in nm]
            max_sc = []
            for pdb2 in pdb2_list:
                try:
                    sc1 = calculate_tmscore(pdb1, pdb2)
                    sc2 = calculate_rmsd(pdb1, pdb2)
                    if sc1 and sc2:
                        max_sc.append((sc1, sc2))
                    else:
                        max_sc.append((0, 1000))
                except:
                    max_sc.append((0, 1000))
            max_sc = sorted(max_sc, key=lambda x: x[0], reverse=True)
            tmsc.append(max_sc[0][0])
            rmsd.append(max_sc[0][1])
        df['tmscore'] = tmsc
        df['rmsd'] = rmsd
        df.to_csv(os.path.join('results', tb), index=False)


if __name__ == '__main__':
    calculate_baseline()