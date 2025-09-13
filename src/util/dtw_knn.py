import dtw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
'''
K-K近邻算法K值 train_data-训练数据,train_labels-训练数据标签,test_data-测试数据,test_labels-测试数据标签,labels_name-数字标签转换为字母
'''


class KNN_DTW:
    def __init__(self, k):
        self.k = k

    def predict(self, train_data, train_labels, test_data, test_labels, labels_name):
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        K = self.k
        i = 0
        accuracy = 0
        predict_labels = []
        for test in test_data:
            t_dis = []
            for train in train_data:
                # 计算 DTW 距离
                alignment = dtw.dtw(test.T, train.T, keep_internals=True)
                dis = alignment.distance  # 获取距离值
                t_dis.append(dis)  # 距离数组
            # KNN 算法预测标签
            nearest_indices = np.argpartition(t_dis, K)[:K]
            nearest_series_labels = np.array(train_labels[nearest_indices]).astype(int)
            predict_label_single = np.argmax(np.bincount(nearest_series_labels))
            predict_labels.append(predict_label_single)
            # 计算正确率
            if predict_label_single == test_labels[i]:
                accuracy += 1
            i += 1
        print('The accuracy is %f (%d of %d)' % ((accuracy / len(test_data)), accuracy, len(test_data)))
        cm_plot(test_labels, predict_labels, labels_name)  # 绘制混淆矩阵
        return accuracy / len(test_data)


def cm_plot(original_label, predict_label, labels_name):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    plt.imshow(cm, interpolation='nearest')
    # plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    cb = plt.colorbar()    # 颜色标签
    cb.ax.tick_params(labelsize=14)  # 设置色标刻度字体大小。
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center', fontsize=14)
    num_x = np.array(range(len(labels_name)))
    num_y = np.array(range(len(labels_name)))
    plt.xticks(num_x, labels_name, fontsize=16)    # 将标签印在x轴坐标上
    plt.yticks(num_y, labels_name, fontsize=16)
    plt.ylabel('True Area', fontsize=22)  # 坐标轴标签
    plt.xlabel('Predicted Area', fontsize=22)  # 坐标轴标签
    plt.title('LVI Confusion Matrix', fontsize=22)
    plt.ylim([-0.5, 15.5])

