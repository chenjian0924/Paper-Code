import numpy as np
import torch
from sklearn import metrics


#  将输入转换为 NumPy 数组。
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class Metrics:
    # 初始化方法 __init__
    def __init__(self, loss=None, predicts=None, targets=None):
        self._loss = loss if loss is not None else 0.0
        self._targets = targets if targets is not None else []
        self._predicts = predicts if predicts is not None else []

    # 这个方法用于更新当前对象的属性，添加另一个 Metrics 对象的预测结果、真实标签和损失值。
    def update(self, other):
        if other is not None:
            self._predicts.extend(to_numpy(other._predicts))
            self._targets.extend(to_numpy(other._targets))
            self._loss += other._loss

    # 计算指定的评估指标
    def _calculate(self, metric, **kwargs):
        return metric(self._targets, self._predicts, **kwargs)

    # 计算并返回平均损失值（总损失除以目标标签的数量）
    @property
    def loss(self):
        if len(self._targets) > 0:
            return self._loss / len(self._targets)
        else:
            return 0

    # 计算并返回宏平均精度（以百分比表示）。
    @property
    def macro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="macro", zero_division=0
        )
        return score * 100

    # 计算并返回宏平均召回率（以百分比表示）。
    @property
    def macro_recall(self):
        score = self._calculate(metrics.recall_score, average="macro", zero_division=0)
        return score * 100

    # 计算并返回微平均精度（以百分比表示）。
    @property
    def micro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="micro", zero_division=0
        )
        return score * 100

    # 计算并返回微平均召回率（以百分比表示）。
    @property
    def micro_recall(self):
        score = self._calculate(metrics.recall_score, average="micro", zero_division=0)
        return score * 100

    # 计算并返回准确率（以百分比表示）。
    @property
    def accuracy(self):
        if self.size == 0:
            return 0
        score = self._calculate(metrics.accuracy_score)
        return score * 100

    # 计算并返回正确分类的数量。
    @property
    def corrects(self):
        return self._calculate(metrics.accuracy_score, normalize=False)

    # 返回目标标签的数量。
    @property
    def size(self):
        return len(self._targets)
