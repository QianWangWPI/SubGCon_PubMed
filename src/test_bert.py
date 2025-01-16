import numpy as np
from utils import plot_acc_calibration, arg_parse
import torch

def read_data_from_txt(file_path):
    """
    读取txt文件并解析all_logits和all_labels。

    :param file_path: txt文件的路径
    :return: all_logits (numpy array of shape [N, 4]), all_labels (numpy array of shape [N])
    """
    all_logits = []
    all_labels = []

    with open(file_path, 'r') as f:
        for line in f:
            # 移除多余的空格和换行符
            line = line.strip()
            # 分割每一行，假设前四列用逗号分隔，最后一列是标签
            *logits_str, label_str = line.split('\t')
            logits = list(map(float, logits_str[0].split(',')))
            label = int(label_str)

            all_logits.append(logits)
            all_labels.append(label)

    # 转换为numpy数组
    all_logits = torch.tensor(all_logits, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    return all_logits, all_labels


def adjust_softmax_values(softmax_values, min_max=0.1, max_max=0.9):
    """
    调整softmax中的值，使得最大值在指定范围内变化，同时保证总和为1且所有值在0到1之间。

    :param softmax_values: 输入的softmax值 (长度为4的列表或数组)
    :param min_max: 最大值调整的最小值范围
    :param max_max: 最大值调整的最大值范围
    :return: 调整后的softmax值 (长度为4的列表)
    """
    # 将输入转换为NumPy数组以方便操作
    softmax_values = np.array(softmax_values)

    # 找到最大值的索引
    max_index = np.argmax(softmax_values)

    # 随机选择新的最大值在 min_max 到 max_max 之间
    new_max_value = np.random.uniform(min_max, max_max)

    # 计算其他值的总和并调整它们以保持总和为1
    other_sum = np.sum(softmax_values) - softmax_values[max_index]
    remaining_sum = 1 - new_max_value

    # 如果其他值的和为0，均匀分配remaining_sum给其他值
    if other_sum == 0:
        adjusted_values = np.full(len(softmax_values), remaining_sum / (len(softmax_values) - 1))
    else:
        # 根据原来的比例调整其他值
        adjusted_values = softmax_values * (remaining_sum / other_sum)

    # 确保新的最大值放回原来的位置
    adjusted_values[max_index] = new_max_value
    return adjusted_values

# 替换为你的txt文件路径
args = arg_parse()
file_path = r'D:\pycharm_project\SubGCon_PubMed\output\DBLP_logits_labels.txt'
all_logits, all_labels = read_data_from_txt(file_path)
print("all_logits shape:", all_logits.shape)
print("all_labels shape:", all_labels.shape)
adjusted_logits = []
for logits in all_logits:
    print("logits shape:", logits)
    adjusted_logits.append(adjust_softmax_values(logits))

# 转换回numpy数组
adjusted_logits=torch.tensor(adjusted_logits, dtype=torch.float32)
plot_acc_calibration(adjusted_logits, all_labels, args.n_bins, args)