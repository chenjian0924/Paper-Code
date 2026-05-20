import json
import os
import pickle
import random
from copy import deepcopy
from collections import Counter
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from src.utils.tools import fix_random_seed
from data.utils.process import (
    exclude_domain,
    plot_distribution,
    prune_args,
    generate_synthetic_data,
    process_celeba,
    process_femnist,
)
from data.utils.schemes import (
    dirichlet,
    iid_partition,
    randomly_assign_classes,
    allocate_shards,
    semantic_partition,
)
from data.utils.datasets import DATASETS, BaseDataset

CURRENT_DIR = Path(__file__).parent.absolute()


def main(args):
    # 定义数据集的根目录路径
    dataset_root = CURRENT_DIR / "data" / args.dataset

    # 固定随机种子，以确保实验的可重复性
    fix_random_seed(args.seed)

    # 如果数据集的目录不存在，创建该目录
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    # 定义客户端数量
    client_num = args.client_num

    # 初始化数据分区字典，其中包含每个客户端的数据索引
    partition = {"separation": None, "data_indices": [[] for _ in range(client_num)]}

    # 初始化统计字典，包括每个客户端的样本数和标签分布情况
    # x: 样本数量，y: 标签分布
    stats = {i: {"x": 0, "y": {}} for i in range(args.client_num)}

    # 初始化数据集变量，后续将根据具体数据集类型进行赋值
    dataset: BaseDataset = None

    # 根据提供的数据集名称处理数据
    if args.dataset == "femnist":
        dataset = process_femnist(args, partition, stats)  # 处理FEMNIST数据集
        partition["val"] = []  # 初始化验证数据分区为空
    elif args.dataset == "celeba":
        dataset = process_celeba(args, partition, stats)  # 处理CelebA数据集
        partition["val"] = []  # 初始化验证数据分区为空
    elif args.dataset == "synthetic":
        dataset = generate_synthetic_data(args, partition, stats)  # 生成合成数据
    else:  # 其他数据集如MEDMNIST, COVID, MNIST, CIFAR10等
        # 如果指定了超出域数据集，FL-bench将把所有标签映射到域空间，并按新的`targets`数组进行数据分区
        dataset = DATASETS[args.dataset](dataset_root, args)
        targets = np.array(dataset.targets, dtype=np.int32)
        target_indices = np.arange(len(targets), dtype=np.int32)
        valid_label_set = set(range(len(dataset.classes)))
        if args.dataset in ["domain"] and args.ood_domains:
            metadata = json.load(open(dataset_root / "metadata.json", "r"))
            # 根据超出域数据集的定义排除特定域
            valid_label_set, targets, client_num = exclude_domain(
                client_num=client_num,
                domain_map=metadata["domain_map"],
                targets=targets,
                domain_indices_bound=metadata["domain_indices_bound"],
                ood_domains=set(args.ood_domains),
                partition=partition,
                stats=stats,
            )

        iid_data_partition = deepcopy(partition)  # 深拷贝数据分区
        iid_stats = deepcopy(stats)  # 深拷贝数据统计信息
        if 0 < args.iid <= 1.0:  # IID数据分区
            # 随机选择一定比例的数据索引进行IID分区
            sampled_indices = np.array(
                random.sample(
                    target_indices.tolist(), int(len(target_indices) * args.iid)
                )
            )

            # 如果args.iid小于1.0，剩余的索引将通过其他分区方法处理
            target_indices = np.array(
                list(set(target_indices) - set(sampled_indices)), dtype=np.int32
            )

            # 执行IID分区
            iid_partition(
                targets=targets[sampled_indices],
                target_indices=sampled_indices,
                label_set=valid_label_set,
                client_num=client_num,
                partition=iid_data_partition,
                stats=iid_stats,
            )

        if len(target_indices) > 0 and args.alpha > 0:  # 使用Dirichlet(alpha)分区
            dirichlet(
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                client_num=client_num,
                alpha=args.alpha,
                least_samples=args.least_samples,
                partition=partition,
                stats=stats,
            )
        elif len(target_indices) > 0 and args.classes != 0:  # 随机分配类别
            args.classes = max(1, min(args.classes, len(dataset.classes)))
            randomly_assign_classes(
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                client_num=client_num,
                class_num=args.classes,
                partition=partition,
                stats=stats,
            )
        elif len(target_indices) > 0 and args.shards > 0:  # 分配数据片
            allocate_shards(
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                client_num=client_num,
                shard_num=args.shards,
                partition=partition,
                stats=stats,
            )
        elif len(target_indices) > 0 and args.semantic:  # 语义分区
            semantic_partition(
                dataset=dataset,
                targets=targets[target_indices],
                target_indices=target_indices,
                label_set=valid_label_set,
                efficient_net_type=args.efficient_net_type,
                client_num=client_num,
                pca_components=args.pca_components,
                gmm_max_iter=args.gmm_max_iter,
                gmm_init_params=args.gmm_init_params,
                seed=args.seed,
                use_cuda=args.use_cuda,
                partition=partition,
                stats=stats,
            )
        elif (
                len(target_indices) > 0
                and args.dataset in ["domain"]
                and args.ood_domains is None
        ):
            with open(dataset_root / "original_partition.pkl", "rb") as f:
                partition = {}
                partition["data_indices"] = pickle.load(f)
                partition["separation"] = None
                args.client_num = len(partition["data_indices"])
        elif len(target_indices) > 0:
            raise RuntimeError(
                "部分数据索引未被处理。请至少设置以下参数之一进行分区："
                " [--alpha, --classes, --shards, --semantic]."
            )

    # 如果IID系数在0到1之间，则合并IID和non-IID数据分区结果
    if 0 < args.iid < 1.0:
        num_samples = []  # 用于存储每个客户端的样本数量
        for i in range(args.client_num):
            # 将IID数据分区的结果合并到当前客户端的数据索引中
            partition["data_indices"][i] = np.concatenate(
                [partition["data_indices"][i], iid_data_partition["data_indices"][i]]
            ).astype(np.int32)

            # 更新客户端的样本数量统计
            stats[i]["x"] += iid_stats[i]["x"]

            # 更新客户端的标签分布统计
            stats[i]["y"] = {
                cls: stats[i]["y"].get(cls, 0) + iid_stats[i]["y"].get(cls, 0)
                for cls in dataset.classes
            }
            # 记录更新后的样本数量
            num_samples.append(stats[i]["x"])

        # 将样本数量列表转换为NumPy数组
        num_samples = np.array(num_samples)

        # 计算并存储所有客户端的样本数量平均值和标准差
        stats["samples_per_client"] = {
            "mean": num_samples.mean().item(),  # 平均样本数量
            "stddev": num_samples.std().item(),  # 样本数量的标准差
        }

    # 如果当前还未设置数据分割方式，根据参数进行分割
    if partition["separation"] is None:
        # 如果分割类型是按用户（客户端）分割
        if args.split == "user":
            # 计算测试客户端的数量
            test_clients_num = int(args.client_num * args.test_ratio)
            # 计算验证客户端的数量
            val_clients_num = int(args.client_num * args.val_ratio)
            # 剩余的都是训练客户端
            train_clients_num = args.client_num - test_clients_num - val_clients_num

            # 创建训练客户端列表
            clients_4_train = list(range(train_clients_num))
            # 创建验证客户端列表
            clients_4_val = list(
                range(train_clients_num, train_clients_num + val_clients_num)
            )
            # 创建测试客户端列表
            clients_4_test = list(
                range(train_clients_num + val_clients_num, args.client_num)
            )

        # 如果分割类型是按样本分割
        elif args.split == "sample":
            # 所有客户端均用于训练、验证和测试
            clients_4_train = list(range(args.client_num))
            clients_4_val = clients_4_train
            clients_4_test = clients_4_train

        # 存储分割信息到分区字典中
        partition["separation"] = {
            "train": clients_4_train,  # 训练客户端列表
            "val": clients_4_val,  # 验证客户端列表
            "test": clients_4_test,  # 测试客户端列表
            "total": args.client_num,  # 客户端总数
        }

    if args.dataset not in ["femnist", "celeba"]:
        # 根据分割策略进一步划分数据
        if args.split == "sample":
            # 如果是按样本分割，对每个客户端的数据索引进行划分
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                np.random.shuffle(indices)  # 打乱数据索引
                testset_size = int(len(indices) * args.test_ratio)  # 测试集大小
                valset_size = int(len(indices) * args.val_ratio)  # 验证集大小
                # 分配训练集、验证集、测试集
                trainset, valset, testset = (
                    # 这表示从 indices 数组中切除了前面的测试集和验证集后的剩余部分。
                    indices[testset_size + valset_size:],  # 训练集
                    # 跳过了全部的测试集部分。
                    indices[testset_size:testset_size + valset_size],  # 验证集
                    indices[:testset_size]  # 测试集
                )
                # 更新客户端的数据索引为详细的训练、验证、测试分组
                partition["data_indices"][client_id] = {
                    "train": trainset,
                    "val": valset,
                    "test": testset,
                }

        elif args.split == "user":
            # 如果是按用户分割，对客户端指定的数据集进行分配
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                # 训练客户端所有数据为训练集，验证集和测试集为空
                partition["data_indices"][client_id] = {
                    "train": indices,
                    "val": np.array([], dtype=np.int64),
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["val"]:
                indices = partition["data_indices"][client_id]
                # 验证客户端所有数据为验证集，训练集和测试集为空
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": indices,
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["test"]:
                indices = partition["data_indices"][client_id]
                # 测试客户端所有数据为测试集，训练集和验证集为空
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": np.array([], dtype=np.int64),
                    "test": indices,
                }

    if args.dataset in ["domain"]:
        class_targets = np.array(dataset.targets, dtype=np.int32)
        metadata = json.load(open(dataset_root / "metadata.json", "r"))

        def _idx_2_domain_label(index):
            for domain, bound in metadata["domain_indices_bound"].items():
                if bound["begin"] <= index < bound["end"]:
                    return metadata["domain_map"][domain]

        domain_targets = np.vectorize(_idx_2_domain_label)(
            np.arange(len(class_targets), dtype=np.int64)
        )
        for client_id in range(args.client_num):
            indices = np.concatenate(
                [
                    partition["data_indices"][client_id]["train"],
                    partition["data_indices"][client_id]["val"],
                    partition["data_indices"][client_id]["test"],
                ]
            ).astype(np.int64)
            stats[client_id] = {
                "x": len(indices),
                "class space": Counter(class_targets[indices].tolist()),
                "domain space": Counter(domain_targets[indices].tolist()),
            }
        stats["domain_map"] = metadata["domain_map"]

    # plot
    if args.plot_distribution:
        if args.dataset in ["domain"]:
            # class distribution
            counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["class space"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/class_distribution.png",
            )
            # domain distribution
            counts = np.zeros(
                (len(metadata["domain_map"]), args.client_num), dtype=np.int64
            )
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["domain space"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/domain_distribution.png",
            )

        else:
            counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["y"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/class_distribution.png",
            )

    with open(dataset_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(dataset_root / "all_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open(dataset_root / "args.json", "w") as f:
        json.dump(prune_args(args), f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASETS.keys(), required=True
    )
    parser.add_argument("--iid", type=float, default=0.0)
    parser.add_argument("-cn", "--client_num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-sp", "--split", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("-vr", "--val_ratio", type=float, default=0.0)
    parser.add_argument("-tr", "--test_ratio", type=float, default=0.25)
    parser.add_argument("-pd", "--plot_distribution", type=int, default=1)

    # Randomly assign classes
    parser.add_argument("-c", "--classes", type=int, default=0)

    # Shards
    parser.add_argument("-s", "--shards", type=int, default=0)

    # Dirichlet
    parser.add_argument("-a", "--alpha", type=float, default=0)
    parser.add_argument("-ls", "--least_samples", type=int, default=40)

    # For synthetic data only
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--dimension", type=int, default=60)

    # For CIFAR-100 only
    parser.add_argument("--super_class", type=int, default=0)

    # For EMNIST only
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )

    # For domain generalization datasets only
    parser.add_argument("--ood_domains", nargs="+", default=None)

    # For semantic partition only
    parser.add_argument("-sm", "--semantic", type=int, default=0)
    parser.add_argument("--efficient_net_type", type=int, default=0)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument(
        "--gmm_init_params", type=str, choices=["random", "kmeans"], default="kmeans"
    )
    parser.add_argument("--pca_components", type=int, default=256)
    parser.add_argument("--use_cuda", type=int, default=1)
    args = parser.parse_args()
    main(args)
