from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

y = [0, 1, 0, 0, 1, 0, 1, 1]  # 真实值（Ground truth）
y_hat = [0, 0, 1, 0, 1, 0, 1, 0]  # 预测值（Predicted labels）

# 第一个参数是真实值，第二个参数是预测值
print(accuracy_score(y, y_hat))

# normalize=False 时，返回的是正确预测的样本数
print(accuracy_score(y, y_hat, normalize=False))

# 0.625
# 5  # 5 个样本预测正确

# 默认返回 pos_label=1 的精确率，即预测为 1 的精确率
print(precision_score(y, y_hat))

# 返回 pos_label=0 的精确率，即预测为 0 的精确率
print(precision_score(y, y_hat, pos_label=0))

# 所有类别的精确率的均值
print(precision_score(y, y_hat, average='macro'))

# 0.6666666666666666
# 0.6
# 0.6333333333333333

print(recall_score(y, y_hat))  # 默认返回 pos_label=1 的精确率
print(recall_score(y, y_hat, pos_label=0))  # 返回 pos_label=0 的精确率
print(recall_score(y, y_hat, average='macro'))  # 计算每个类别的精确率，然后计算它们的均值

# 0.5
# 0.75
# 0.625
