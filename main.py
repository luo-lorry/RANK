import torch
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR, APS, SAPS, RAPS, THRRANK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 设置字体为Times
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 数据集名称映射
DATASET_NAMES = {
    'logits_labels_agnews.pt': 'agnews',
    'logits_labels_carer.pt': 'carer',
    'logits_labels_marketing.pt': 'marketing',
    'logits_labels_medicine.pt': 'medicine',
    'logits_labels_news20.pt': 'news20',
    'logits_labels_tweet.pt': 'tweet',
    'logits_labels_relations.pt': 'relations',
    'logits_labels_cifar100.pt': 'cifar100',
    'logits_labels_mnist.pt': 'mnist',
    'logits_labels_fashionmnist.pt': 'fmnist',
    'logits_labels_cifar10_resnet56.pt': 'cifar10'
}

# 从metrics.py导入WSC指标
from torchcp.classification.utils.metrics import WSC


def WSSV(prediction_sets, labels, alpha, stratified_size=[[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]]):
    """
    Weighted Size-stratified Coverage Violation (WSSV).
    """
    if len(prediction_sets) != len(labels):
        raise ValueError("The number of prediction sets must be equal to the number of labels.")

    if not isinstance(stratified_size, list) or not stratified_size:
        raise ValueError("stratified_size must be a non-empty list")

    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()

    size_array = prediction_sets.sum(dim=1)
    correct_array = prediction_sets[torch.arange(len(labels)), labels]

    weighted_violations = 0.0
    total_weights = 0

    for stratum in stratified_size:
        stratum_mask = (size_array >= stratum[0]) & (size_array <= stratum[1])
        stratum_size = stratum_mask.sum().item()

        if stratum_size > 0:
            stratum_coverage = correct_array[stratum_mask].float().mean()
            stratum_violation = torch.abs(1 - alpha - stratum_coverage).item()

            weighted_violations += stratum_violation * stratum_size
            total_weights += stratum_size

    if total_weights > 0:
        return weighted_violations / total_weights
    else:
        return 0.0


def load_data(file_path):
    """加载logits和labels数据"""
    try:
        data = torch.load(file_path, weights_only=True)
        if isinstance(data, dict):
            logits = data.get('logits', data.get('predictions'))
            labels = data.get('labels', data.get('targets'))
        else:
            logits, labels = data

        print(
            f"数据加载成功: {os.path.basename(file_path)}, logits shape: {logits.shape}, labels shape: {labels.shape}")
        return logits, labels
    except Exception as e:
        print(f"数据加载失败，尝试使用兼容模式: {e}")
        try:
            print("警告：使用默认模式加载数据，请确保数据源可信")
            data = torch.load(file_path, weights_only=False)
            if isinstance(data, dict):
                logits = data.get('logits', data.get('predictions'))
                labels = data.get('labels', data.get('targets'))
            else:
                logits, labels = data
            print(
                f"数据加载成功: {os.path.basename(file_path)}, logits shape: {logits.shape}, labels shape: {labels.shape}")
            return logits, labels
        except Exception as e2:
            print(f"数据加载失败: {e2}")
            return None, None


def get_conformal_predictor(method_name, model=None, device='cpu', num_classes=4):
    """基于方法名称创建并返回一个一致预测器。"""
    if method_name == "APS":
        return SplitPredictor(score_function=APS(), model=model)
    elif method_name == "RAPS":
        return SplitPredictor(score_function=RAPS(), model=model)
    elif method_name == "SAPS":
        return SplitPredictor(score_function=SAPS(), model=model)
    elif method_name == "THR":
        return SplitPredictor(score_function=THR(), model=model)
    elif method_name == "THRRANK":
        return SplitPredictor(score_function=THRRANK(), model=model)
    else:
        raise ValueError(f"未知的一致预测方法: {method_name}")


def evaluation(cal_x, cal_y, test_x, test_y, method_name, alpha=0.01, dataset='NLP'):
    """评估函数 - 添加WSC指标"""
    if not torch.is_tensor(cal_x):
        cal_x = torch.from_numpy(cal_x).double()
        cal_y = torch.from_numpy(cal_y).long()
        test_x = torch.from_numpy(test_x).double()
        test_y = torch.from_numpy(test_y).long()

    # 使用get_conformal_predictor创建预测器
    predictor = get_conformal_predictor(method_name)

    predictor.calculate_threshold(cal_x, cal_y, alpha)
    prediction_sets = predictor.predict_with_logits(test_x)
    labels_list = test_y

    # 导入需要的指标函数
    from torchcp.classification.utils.metrics import coverage_rate, average_size, SSCV, singleton_hit_ratio

    # 移动到CPU
    prediction_sets = prediction_sets.cpu()
    labels_list = labels_list.cpu()
    test_x = test_x.cpu()

    # 准备特征用于WSC计算
    features = test_x.view(test_x.size(0), -1) if len(test_x.shape) > 2 else test_x

    # 计算原有指标
    coverage = coverage_rate(prediction_sets, labels_list)
    size = average_size(prediction_sets, labels_list)
    sscv = SSCV(prediction_sets, labels_list, alpha)
    wssv = WSSV(prediction_sets, labels_list, alpha)
    singleton_hit_ratio_val = singleton_hit_ratio(prediction_sets, labels_list)

    # 计算WSC指标
    try:
        wsc = WSC(features, prediction_sets, labels_list,
                  delta=0.1, M=50, test_fraction=0.75, random_state=2020, verbose=False)
    except Exception as e:
        print(f"WSC calculation failed: {e}")
        wsc = 0.0

    return {
        'coverage': np.round(coverage, 3),
        'size': np.round(size, 3),
        'sscv': np.round(sscv, 3),
        'wssv': np.round(wssv, 3),
        'singleton_hit_ratio': np.round(singleton_hit_ratio_val, 3),
        'wsc': np.round(wsc, 3)
    }


def run_experiments_single_dataset(logits, labels, dataset_name, split_ratio=0.5,
                                   alphas=None, method_names=None,
                                   num_repetitions=10):
    """对单个数据集运行实验，添加WSC指标"""
    if alphas is None:
        alphas = np.linspace(0.1, 0.3, 11)

    if method_names is None:
        method_names = ['THRRANK', 'APS', 'RAPS', 'SAPS']

    # 添加WSC指标
    metric_names = ['coverage', 'size', 'sscv', 'wssv', 'singleton_hit_ratio', 'wsc']

    # 存储每次实验的结果
    all_runs_results = {
        method: {
            alpha: {metric: [] for metric in metric_names}
            for alpha in alphas
        }
        for method in method_names
    }

    print(f"\n=== 处理数据集: {dataset_name} ===")
    print(f"校准集比例: {split_ratio}, 测试集比例: {1 - split_ratio}")
    print(f"运行次数: {num_repetitions}")

    for run_id in range(num_repetitions):
        print(f"运行 {run_id + 1}/{num_repetitions}")

        # 随机划分数据
        index = np.arange(len(logits))
        np.random.shuffle(index)
        n_cal = int(len(logits) * split_ratio)

        cal_scores = logits[index][:n_cal]
        val_scores = logits[index][n_cal:]
        cal_targets = labels[index][:n_cal]
        val_targets = labels[index][n_cal:]

        for alpha in alphas:
            for method_name in method_names:
                metrics = evaluation(cal_scores, cal_targets, val_scores, val_targets,
                                     method_name, alpha=alpha, dataset=dataset_name)

                # 存储所有指标
                for metric_name, value in metrics.items():
                    all_runs_results[method_name][alpha][metric_name].append(value)

    # 计算统计信息
    results_stats = {}
    for method_name in method_names:
        results_stats[method_name] = {'alphas': alphas}

        for metric_name in metric_names:
            results_stats[method_name][f'{metric_name}_mean'] = []
            results_stats[method_name][f'{metric_name}_std'] = []

            for alpha in alphas:
                values = all_runs_results[method_name][alpha][metric_name]
                results_stats[method_name][f'{metric_name}_mean'].append(np.mean(values))
                results_stats[method_name][f'{metric_name}_std'].append(np.std(values))

    return results_stats


def save_results_to_text(results_stats, dataset_name, output_dir):
    """保存结果到文本文件，只保存alpha=0.1和0.05的结果，添加WSC指标"""
    output_file = os.path.join(output_dir, f'{dataset_name}_results.txt')

    # 找到alpha=0.1和0.05在数组中的索引
    alphas = results_stats[list(results_stats.keys())[0]]['alphas']
    target_alphas = [0.1, 0.05]
    alpha_indices = []

    for target_alpha in target_alphas:
        # 找最接近的alpha值
        closest_idx = np.argmin(np.abs(alphas - target_alpha))
        if abs(alphas[closest_idx] - target_alpha) < 0.01:  # 允许小的误差
            alpha_indices.append((closest_idx, target_alpha))

    with open(output_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write("=" * 110 + "\n\n")

        for method_name in results_stats.keys():
            f.write(f"Method: {method_name}\n")
            f.write("-" * 110 + "\n")

            # 写入表头，添加WSC
            header = f"{'Alpha':<8} {'Cvg_Mean':<10} {'Cvg_Std':<10} {'Size_Mean':<10} {'Size_Std':<10} "
            header += f"{'SSCV_Mean':<10} {'SSCV_Std':<10} {'WSSV_Mean':<10} {'WSSV_Std':<10} "
            header += f"{'Single_Mean':<12} {'Single_Std':<12} {'WSC_Mean':<10} {'WSC_Std':<10}\n"
            f.write(header)

            # 只写入指定alpha值的结果
            for idx, target_alpha in alpha_indices:
                line = f"{target_alpha:<8.2f} "
                line += f"{results_stats[method_name]['coverage_mean'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['coverage_std'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['size_mean'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['size_std'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['sscv_mean'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['sscv_std'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['wssv_mean'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['wssv_std'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['singleton_hit_ratio_mean'][idx]:<12.4f} "
                line += f"{results_stats[method_name]['singleton_hit_ratio_std'][idx]:<12.4f} "
                line += f"{results_stats[method_name]['wsc_mean'][idx]:<10.4f} "
                line += f"{results_stats[method_name]['wsc_std'][idx]:<10.4f}\n"
                f.write(line)
            f.write("\n")

    print(f"结果已保存到: {output_file}")


def plot_combined_metrics(results_stats, dataset_name, output_dir):
    """绘制合并的四张图：Size vs Coverage + Alpha vs 三个指标（在Size vs Coverage中添加阴影带）"""
    # 自定义颜色映射
    colors = plt.get_cmap("tab10")
    method_colors = {
        'THRRANK': colors(3),
        'APS': colors(0),
        'RAPS': colors(1),
        'SAPS': colors(2)
    }

    method_names = ['THRRANK', 'APS', 'RAPS', 'SAPS']
    alphas = results_stats[method_names[0]]['alphas']

    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'{dataset_name}', fontsize=24, fontweight='bold', x=0.54)

    # 1. Size vs Coverage (左上) - 添加阴影带
    ax1 = axes[0, 0]
    for method_name in method_names:
        coverage_mean = results_stats[method_name]['coverage_mean']
        coverage_std = results_stats[method_name]['coverage_std']
        size_mean = results_stats[method_name]['size_mean']
        size_std = results_stats[method_name]['size_std']

        label = "RANK (Ours)" if method_name == 'THRRANK' else method_name

        # 绘制主线
        ax1.plot(coverage_mean, size_mean, color=method_colors[method_name],
                 linestyle='-', marker='o', label=label, linewidth=2, markersize=6)

        # 添加阴影带（使用coverage和size的标准差）
        # 计算阴影带的边界
        coverage_lower = np.array(coverage_mean) - np.array(coverage_std)
        coverage_upper = np.array(coverage_mean) + np.array(coverage_std)
        size_lower = np.array(size_mean) - np.array(size_std)
        size_upper = np.array(size_mean) + np.array(size_std)

        # 为了绘制阴影带，我们需要创建一个多边形
        # 先绘制上边界（coverage_upper, size_upper）
        # 然后绘制下边界（coverage_lower, size_lower）的逆序
        coverage_band = np.concatenate([coverage_upper, coverage_lower[::-1]])
        size_band = np.concatenate([size_upper, size_lower[::-1]])

        ax1.fill(coverage_band, size_band, color=method_colors[method_name], alpha=0.2)

    ax1.set_xlabel('Coverage', fontsize=20)
    ax1.set_ylabel('Size', fontsize=20)
    ax1.set_title('Size vs. Coverage', fontsize=20, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. SSCV vs Alpha (右上)
    ax2 = axes[0, 1]
    for method_name in method_names:
        sscv_mean = results_stats[method_name]['sscv_mean']
        sscv_std = results_stats[method_name]['sscv_std']

        label = "RANK (Ours)" if method_name == 'THRRANK' else method_name

        # 绘制主线
        ax2.plot(alphas, sscv_mean, color=method_colors[method_name], linestyle='-',
                 marker='o', label=label, linewidth=2, markersize=6)

        # 添加误差带
        ax2.fill_between(alphas,
                         np.array(sscv_mean) - np.array(sscv_std),
                         np.array(sscv_mean) + np.array(sscv_std),
                         color=method_colors[method_name], alpha=0.2)

    ax2.set_xlabel('α', fontsize=20)
    ax2.set_ylabel('SSCV', fontsize=20)
    ax2.set_title('SSCV vs. α', fontsize=20, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. WSSV vs Alpha (左下)
    ax3 = axes[1, 0]
    for method_name in method_names:
        wssv_mean = results_stats[method_name]['wssv_mean']
        wssv_std = results_stats[method_name]['wssv_std']

        label = "RANK (Ours)" if method_name == 'THRRANK' else method_name

        # 绘制主线
        ax3.plot(alphas, wssv_mean, color=method_colors[method_name], linestyle='-',
                 marker='o', label=label, linewidth=2, markersize=6)

        # 添加误差带
        ax3.fill_between(alphas,
                         np.array(wssv_mean) - np.array(wssv_std),
                         np.array(wssv_mean) + np.array(wssv_std),
                         color=method_colors[method_name], alpha=0.2)

    ax3.set_xlabel('α', fontsize=20)
    ax3.set_ylabel('WSSV', fontsize=20)
    ax3.set_title('WSSV vs. α', fontsize=20, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Singleton Hit Ratio vs Alpha (右下)
    ax4 = axes[1, 1]
    for method_name in method_names:
        singleton_mean = results_stats[method_name]['singleton_hit_ratio_mean']
        singleton_std = results_stats[method_name]['singleton_hit_ratio_std']

        label = "RANK (Ours)" if method_name == 'THRRANK' else method_name

        # 绘制主线
        ax4.plot(alphas, singleton_mean, color=method_colors[method_name], linestyle='-',
                 marker='o', label=label, linewidth=2, markersize=6)

        # 添加误差带
        ax4.fill_between(alphas,
                         np.array(singleton_mean) - np.array(singleton_std),
                         np.array(singleton_mean) + np.array(singleton_std),
                         color=method_colors[method_name], alpha=0.2)

    ax4.set_xlabel('α', fontsize=20)
    ax4.set_ylabel('Singleton Hit Ratio', fontsize=20)
    ax4.set_title('Singleton Hit Ratio vs. α', fontsize=20, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存PDF文件
    output_file = os.path.join(output_dir, f'{dataset_name}_combined_metrics.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"合并指标图表已保存到: {output_file}")
    plt.show()


def plot_wsc_metric(results_stats, dataset_name, output_dir):
    """单独绘制WSC指标并保存为PDF"""
    # 自定义颜色映射
    colors = plt.get_cmap("tab10")
    method_colors = {
        'THRRANK': colors(3),
        'APS': colors(0),
        'RAPS': colors(1),
        'SAPS': colors(2)
    }

    method_names = ['THRRANK', 'APS', 'RAPS', 'SAPS']
    alphas = results_stats[method_names[0]]['alphas']

    # 创建单独的图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f'{dataset_name}', fontsize=24, fontweight='bold', x=0.55)

    for method_name in method_names:
        wsc_mean = results_stats[method_name]['wsc_mean']
        wsc_std = results_stats[method_name]['wsc_std']

        label = "RANK (Ours)" if method_name == 'THRRANK' else method_name

        # 绘制主线
        ax.plot(alphas, wsc_mean, color=method_colors[method_name], linestyle='-',
                marker='o', label=label, linewidth=2, markersize=8)

        # 添加误差带
        ax.fill_between(alphas,
                        np.array(wsc_mean) - np.array(wsc_std),
                        np.array(wsc_mean) + np.array(wsc_std),
                        color=method_colors[method_name], alpha=0.2)

    ax.set_xlabel('α', fontsize=20)
    ax.set_ylabel('WSC', fontsize=20)
    ax.set_title('WSC vs. α', fontsize=20, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存PDF文件
    output_file = os.path.join(output_dir, f'{dataset_name}_WSC_metric.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"WSC指标图表已保存到: {output_file}")
    plt.show()


def find_dataset_files(data_folder):
    """查找数据文件夹中的指定.pt文件"""
    target_files = [
        'logits_labels_cifar100.pt',
        'logits_labels_agnews.pt',
        'logits_labels_carer.pt',
        'logits_labels_marketing.pt',
        'logits_labels_medicine.pt',
        'logits_labels_news20.pt',
        'logits_labels_relations.pt',
        'logits_labels_tweet.pt',
        'logits_labels_fashionmnist.pt',
        'logits_labels_mnist.pt',
        'logits_labels_cifar10_resnet56.pt',
    ]

    found_files = []
    for filename in target_files:
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            found_files.append(file_path)
        else:
            print(f"警告: 未找到文件 {filename}")

    return found_files


# 主程序
if __name__ == "__main__":
    # ========== 手动调节参数区域 ==========
    data_folder = './nlp'  # 数据文件夹路径
    split_ratio = 0.4  # 校准集比例 (0-1之间)
    num_repetitions = 100  # 随机划分运行次数

    # Alpha范围设置
    alphas = np.linspace(0.1, 0.3, 11)

    # 方法列表
    method_names = ['THRRANK', 'APS', 'RAPS', 'SAPS']
    # =====================================

    print("=== 实验参数设置 ===")
    print(f"数据文件夹: {data_folder}")
    print(f"校准集比例: {split_ratio}")
    print(f"测试集比例: {1 - split_ratio}")
    print(f"随机划分运行次数: {num_repetitions}")
    print(f"Alpha范围: {alphas[0]:.2f} - {alphas[-1]:.2f} (共{len(alphas)}个点)")
    print(f"测试方法: {method_names}")
    print("=" * 40)

    # 查找所有数据集文件
    dataset_files = find_dataset_files(data_folder)

    if not dataset_files:
        print(f"在文件夹 {data_folder} 中未找到指定的数据集文件")
        exit()

    print(f"\n找到 {len(dataset_files)} 个数据集文件:")
    for file in dataset_files:
        filename = os.path.basename(file)
        dataset_name = DATASET_NAMES.get(filename, filename)
        print(f"  - {filename} -> {dataset_name}")

    # 为每个数据集运行实验
    for file_path in dataset_files:
        filename = os.path.basename(file_path)
        dataset_name = DATASET_NAMES.get(filename, os.path.splitext(filename)[0])

        # 创建数据集专用文件夹
        dataset_output_dir = os.path.join('results', dataset_name)
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

        # 加载数据
        print(f"\n正在处理数据集: {dataset_name}")
        logits, labels = load_data(file_path)

        if logits is None or labels is None:
            print(f"数据集 {dataset_name} 加载失败，跳过")
            continue

        # 运行实验
        results_stats = run_experiments_single_dataset(
            logits, labels, dataset_name,
            split_ratio=split_ratio,
            alphas=alphas,
            method_names=method_names,
            num_repetitions=num_repetitions
        )

        # 保存结果到文本文件（只保存alpha=0.1和0.05的结果）
        save_results_to_text(results_stats, dataset_name, dataset_output_dir)

        # 绘制原有的合并图表
        plot_combined_metrics(results_stats, dataset_name, dataset_output_dir)

        # 单独绘制WSC指标图表
        plot_wsc_metric(results_stats, dataset_name, dataset_output_dir)

        print(f"数据集 {dataset_name} 处理完成，结果保存在: {dataset_output_dir}")

    print("\n所有数据集处理完成！")