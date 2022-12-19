from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y = [0, 1, 0, 0, 1, 0, 1, 1]  # 真实值（Ground truth）
y_hat = [0, 0, 1, 0, 1, 0, 1, 0]  # 预测值（Predicted labels）

print(confusion_matrix(y, y_hat))  # 第一个参数是真实值，第二个参数是预测值

sns.heatmap(
    confusion_matrix(y, y_hat),
    annot=True,  # 在图中显示数字
    cmap=plt.cm.Blues,  # 颜色
    fmt='d',  # 数字格式, d: 整数
    xticklabels=['Predicted 0', 'Predicted 1'],  # x 轴标签
    yticklabels=['Actual 0', 'Actual 1']  # y 轴标签
)

plt.savefig('confusion_matrix.svg')  # 保存图片
