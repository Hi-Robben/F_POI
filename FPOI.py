import numpy as np
from scipy.stats import pearsonr
from scipy.stats import f as f_dist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def calculate_f_test_select(data, labels, alpha=0.05):
    """
    计算每个特征的 F值，并返回 F值超过 F临界值的特征索引。

    参数:
        data: 二维数组，形状为 (n_samples, n_features)，表示数据集。
        labels: 一维数组，形状为 (n_samples,)，表示每个样本的类别标签。
        alpha: 显著性水平（默认值为0.05）。

    返回:
        significant_indices: F值超过 F临界值的特征索引。
        f_values: 每个特征的 F值。
        f_critical: 计算得到的 F临界值。
    """
    n_samples, n_features = data.shape
    f_values = np.zeros(n_features)  # 存储每个特征的 F值

    # 获取唯一标签和组数
    unique_labels = np.unique(labels)
    k = len(unique_labels)  # 组数
    N = n_samples  # 总样本量

    # 计算 F临界值
    df_between = k - 1  # 组间自由度
    df_within = N - k  # 组内自由度
    f_critical = f_dist.ppf(1 - alpha, df_between, df_within)  # F临界值

    # 遍历每个特征
    for i in range(n_features):
        feature_data = data[:, i]  # 提取第 i 个特征的所有样本

        # 计算组间方差（MS_between）
        overall_mean = np.mean(feature_data)  # 总体均值
        group_means = [np.mean(feature_data[labels == label]) for label in unique_labels]  # 每组均值
        ss_between = np.sum([np.sum(labels == label) * (mean - overall_mean) ** 2 for label, mean in zip(unique_labels, group_means)])  # 组间平方和
        ms_between = ss_between / df_between  # 组间均方

        # 计算组内方差（MS_within）
        ss_within = np.sum([np.sum((feature_data[labels == label] - mean) ** 2) for label, mean in zip(unique_labels, group_means)])  # 组内平方和
        ms_within = ss_within / df_within  # 组内均方

        # 计算F值
        if ms_within > 0:  # 避免除零错误
            f_value = ms_between / ms_within
        else:
            f_value = 0  # 如果组内方差为零，F值设为0

        f_values[i] = f_value

    # 筛选 F值超过 F临界值的特征索引
    significant_indices = np.where(f_values > f_critical)[0]
    return significant_indices, f_values, f_critical


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def Numclassify_with_xgboost(class_data_list, test_data):
    # 准备训练数据和标签
    X_train = np.vstack(class_data_list)
    y_train = np.hstack([np.full(len(data), idx) for idx, data in enumerate(class_data_list)])

    # 初始化 XGBoost 分类器
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    # 对 test_data 进行分类
    predictions = xgb.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)



def Numclassify_with_decision_tree(class_data_list, test_data):
    # 准备训练数据和标签
    X_train = np.vstack(class_data_list)
    y_train = np.hstack([np.full(len(data), idx) for idx, data in enumerate(class_data_list)])

    # 初始化决策树分类器
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # 对 test_data 进行分类
    predictions = dt.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)




def Numclassify_with_kmeans(class_data_list, test_data):
    """
    使用 K均值聚类进行多类别分类
    :param class_data_list: 列表，包含每个类别的数据，例如 [f1_tvla, f2_tvla, f3_tvla]
    :param test_data: 待分类数据，形状为 (n_samples, n_features)
    :return: 最大的 total_f{idx + 1} 的 count
    """
    # 准备训练数据
    X_train = np.vstack(class_data_list)  # 将所有类别的数据堆叠

    # 初始化 KMeans 聚类器
    kmeans = KMeans(n_clusters=len(class_data_list), random_state=42)
    kmeans.fit(X_train)

    # 对 test_data 进行分类
    predictions = kmeans.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)


def Numclassify_with_naive_bayes(class_data_list, test_data):
    """
    使用朴素贝叶斯进行多类别分类
    :param class_data_list: 列表，包含每个类别的数据，例如 [f1_tvla, f2_tvla, f3_tvla]
    :param test_data: 待分类数据，形状为 (n_samples, n_features)
    :return: 最大的 total_f{idx + 1} 的 count
    """
    # 准备训练数据和标签
    X_train = np.vstack(class_data_list)  # 将所有类别的数据堆叠
    y_train = np.hstack([np.full(len(data), idx) for idx, data in enumerate(class_data_list)])  # 生成标签

    # 初始化朴素贝叶斯分类器
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # 对 test_data 进行分类
    predictions = nb.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)


def Numclassify_with_random_forest(class_data_list, test_data, n_estimators=100):
    """
    使用随机森林进行多类别分类
    :param class_data_list: 列表，包含每个类别的数据，例如 [f1_tvla, f2_tvla, f3_tvla]
    :param test_data: 待分类数据，形状为 (n_samples, n_features)
    :param n_estimators: 随机森林中树的数量，默认为 100
    :return: 最大的 total_f{idx + 1} 的 count
    """
    # 准备训练数据和标签
    X_train = np.vstack(class_data_list)  # 将所有类别的数据堆叠
    y_train = np.hstack([np.full(len(data), idx) for idx, data in enumerate(class_data_list)])  # 生成标签

    # 初始化随机森林分类器
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # 对 test_data 进行分类
    predictions = rf.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)


def Numclassify_with_svm(class_data_list, test_data):
    """
    使用 SVM 进行多类别分类
    :param class_data_list: 列表，包含每个类别的数据，例如 [f1_tvla, f2_tvla, f3_tvla]
    :param test_data: 待分类数据，形状为 (n_samples, n_features)
    :return: 最大的 total_f{idx + 1} 的 count
    """
    # 准备训练数据和标签
    X_train = np.vstack(class_data_list)  # 将所有类别的数据堆叠
    y_train = np.hstack([np.full(len(data), idx) for idx, data in enumerate(class_data_list)])  # 生成标签

    # 初始化 SVM 分类器
    svm = SVC(kernel='linear')  # 使用线性核
    svm.fit(X_train, y_train)

    # 对 test_data 进行分类
    predictions = svm.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)


def Numclassify_with_knn(class_data_list, test_data, k=3):
    """
    使用 KNN 进行多类别分类
    :param class_data_list: 列表，包含每个类别的数据，例如 [f1_tvla, f2_tvla, f3_tvla]
    :param test_data: 待分类数据，形状为 (n_samples, n_features)
    :param k: KNN 中的 K 值，默认为 3
    :return: 最大的 total_f{idx + 1} 的 count
    """
    # 准备训练数据和标签
    X_train = np.vstack(class_data_list)  # 将所有类别的数据堆叠
    y_train = np.hstack([np.full(len(data), idx) for idx, data in enumerate(class_data_list)])  # 生成标签

    # 初始化 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 对 test_data 进行分类
    predictions = knn.predict(test_data)

    # 统计每个类别的投票数
    counts = np.bincount(predictions, minlength=len(class_data_list))

    # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)



def Numclassify_based_on_distance_multiclass(class_data_list, test_data):
    """
    支持多类别分类的版本
    :param class_data_list: 列表，包含每个类别的数据，例如 [f1_tvla, f2_tvla, f3_tvla]
    :param test_data: 待分类数据，形状为 (n_samples, n_features)
    :return: 最大的 total_f{idx + 1} 的 count
    """
    # 计算每个类别的平均值
    means = [np.mean(class_data, axis=0) for class_data in class_data_list]

    # 初始化计数器
    counts = [0] * len(class_data_list)

    # 对 test_data 中的每个样本进行分类
    for i in range(test_data.shape[0]):
        distances = []
        for mean in means:
            dist_total = 0
            for j in range(test_data.shape[1]):
                dist_total += euclidean_distance(mean[j], test_data[i][j])
            distances.append(dist_total)

        # 找到距离最小的类别
        min_index = np.argmin(distances)
        counts[min_index] += 1

    # # 打印每个类别的投票数
    # for idx, count in enumerate(counts):
    #     print(f"total_f{idx + 1}: {count}")

    # # 根据多数投票原则决定最终分类
    # result_index = np.argmax(counts)
    # print(f"test_data belongs to class f{result_index + 1}")

    # 返回最大的 total_f{idx + 1} 的 count
    return max(counts)



prefixes = ['#1_0000']
# prefixes = ['#1_0000', '#2_0001']
# prefixes = ['#1_0000', '#2_0001', '#3_0010', '#4_0011']
# prefixes = ['#1_0000', '#2_0001', '#3_0010', '#4_0011', '#5_0100', '#6_0101', '#7_0110', '#8_0111']
# prefixes = ['#1_0000', '#2_0001', '#3_0010', '#4_0011', '#5_0100', '#6_0101', '#7_0110', '#8_0111', 
#             '#9_1000', '#10_1001', '#11_1010', '#12_1011', '#13_1100', '#14_1101', '#15_1110', '#16_1111']
# classN = 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256 
classN = 2
train_sample_sizes = range(10, 101, 10)  # [10,20,...,100]
test_sample_range = (101, 201)  # 测试数据范围
# Rename the output file
output_file = f"1bits_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"  # 带时间戳的文件名



classifiers = [
    {
        'name': '基于距离的多分类',
        'func': Numclassify_based_on_distance_multiclass,
        'args': {}
    },

    {
        'name': 'KNN (k=8)',
        'func': Numclassify_with_knn,
        'args': {'k': 9}
    },
    {
        'name': 'SVM',
        'func': Numclassify_with_svm,
        'args': {}
    },
    {
        'name': '决策树',
        'func': Numclassify_with_decision_tree,
        'args': {}
    },

    {
        'name': 'XGBoost',
        'func': Numclassify_with_xgboost,
        'args': {}
    },


    {
        'name': '随机森林 (300棵树)',
        'func': Numclassify_with_random_forest,
        'args': {'n_estimators': 300}
    },
    {
        'name': '朴素贝叶斯',
        'func': Numclassify_with_naive_bayes,
        'args': {}
    }
]

# 初始化结果文件
with open(output_file, 'w') as f:
    f.write(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"训练样本量配置: {list(train_sample_sizes)}\n")
    f.write(f"测试样本范围: {test_sample_range}\n")
    f.write("-"*80 + "\n")

# 主循环流程 ---------------------------------------------------------
with open(output_file, 'a') as f:  # 追加模式打开文件
    for n_train in train_sample_sizes:
        # ================= 控制台和文件同步输出 =================
        header = f"\n{'#'*40}\n当前训练样本量: {n_train}\n{'#'*40}"
        print(header)
        f.write(header + "\n")

        # 加载训练数据（每次样本量变化时重新加载）
        train_data = [
            # np.load(f'../8bits_Lab/{prefix}/{prefix}_{i:04b}_FPOI_500.npy')[:n_train]
            np.load(f'./data/preprocessed/{prefix}/{prefix}_{i:04b}_FPOI_500.npy')[:n_train]
            for prefix in prefixes
            for i in range(2)
        ]
        
        # 生成标签和合并数据
        labels = np.concatenate([np.ones(n_train) * i for i in range(classN)])
        X1 = np.concatenate(train_data)
        
        # 特征选择（每个样本量独立进行）
        sig_idx, f_values, _ = calculate_f_test_select(X1, labels)
        top_k = min(200, sig_idx.size)
        idx_reduced = sig_idx[np.argsort(f_values[sig_idx])[-top_k:][::-1]]

        train_data_tvla = [data[:, idx_reduced] for data in train_data]

        # 遍历分类器
        for classifier in classifiers:
            # ============== 分类器头信息 ==============
            classifier_header = f"\n{'='*30}\n分类器: {classifier['name']}\n{'='*30}"
            print(classifier_header)
            f.write(classifier_header + "\n")

            result_list = []
            total_result = 0
            start_time = time.time()

            # 遍历测试数据
            for prefix in prefixes:
                for i in range(2):
                    # 加载测试数据
                    test_data = np.load(
                        f'./data/preprocessed/{prefix}/{prefix}_{i:04b}_FPOI_500.npy'
                    )[test_sample_range[0]:test_sample_range[1]]
                    test_data_tvla = test_data[:, idx_reduced]
                    
                    # 执行分类
                    result = classifier['func'](
                        train_data_tvla, 
                        test_data_tvla, 
                        **classifier['args']
                    )
                    
                    # 记录结果
                    result_str = f"{prefix}_{i:04b}: {result}"
                    print(result_str)
                    f.write(result_str + "\n")
                    
                    result_list.append(result)
                    total_result += result

            # ============== 统计结果输出 ==============
            elapsed_time = time.time() - start_time
            stats = (
                f"\n统计结果 ({classifier['name']}, n_train={n_train})"
                f"\n总耗时: {elapsed_time:.2f}s"
                f"\n总结果值: {total_result}"
                f"\n平均结果: {total_result/len(result_list):.2f}"
                f"\n{'='*30}"
            )
            print(stats)
            f.write(stats + "\n")

        # ================= 训练样本量结束分隔线 =================
        footer = f"\n{'#'*40}\n样本量 {n_train} 测试结束\n{'#'*40}"
        print(footer)
        f.write(footer + "\n")

# 最终结束标记
with open(output_file, 'a') as f:
    f.write(f"\n{'*'*80}\n")
    f.write(f"实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write('*'*80 + "\n")

print(f"\n所有结果已保存至: {output_file}")
