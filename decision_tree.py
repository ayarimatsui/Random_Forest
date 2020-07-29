# 決定木による分類器の実装
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import mode  #最頻値を求めるのに使う
from sklearn.model_selection import train_test_split

# 決定木のノード
class Node(object):

    def __init__(self, feature_id, threshold, branch_true, branch_false):
        self.feature_id = feature_id
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false



# 決定木による分類器
class DecisionTreeClassifier():

    def __init__(self, max_features=lambda n: n, max_depth=float('inf')):
        # max_features : 使用するデータの特徴量の数
        self.max_features = max_features
        # max_depth : 決定木の深さの制限
        self.max_depth = max_depth
        self.min_split_num = 2  # さらに細かく分割するときに最低限必要なサンプルデータの数


    def fit(self, X, y):
        features_num = X.shape[1]  # 入力のデータの特徴量の数
        sub_features_num = int(self.max_features(features_num))
        feature_ids = random.sample(range(features_num), sub_features_num)
        
        self.trunk = self.build_tree(X, y, feature_ids, 0)


    def predict(self, X):
        N = X.shape[0]
        y = np.zeros(N)

        for i in range(N):
            node = self.trunk
            while isinstance(node, Node):  # nodeがNode型かどうかを判定
                if X[i][node.feature_id] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y[i] = node

        return y


    # 葉ノードのコスト(Cross-entropy)を計算
    def entropy(self, y):
        distribution = collections.Counter(y)
        s = 0
        K = len(y)
        for k, num_k in distribution.items():
            p_k = num_k / K
            s -= (p_k) * np.log(p_k)
        return s


    # information gainを計算
    def information_gain(self, y, y_true, y_false):
        IG = self.entropy(y) - (self.entropy(y_true) * len(y_true) + self.entropy(y_false) * len(y_false)) / len(y)
        return IG


    # 閾値を境界に分割
    def split(self, X, y, feature_id, threshold):
        X_true = []
        y_true = []
        X_false = []
        y_false = []

        for i in range(len(y)):
            if X[i][feature_id] <= threshold:
                X_true.append(X[i])
                y_true.append(y[i])
            else:
                X_false.append(X[i])
                y_false.append(y[i])

        # Numpy配列に変換
        X_true = np.array(X_true)
        y_true = np.array(y_true)
        X_false = np.array(X_false)
        y_false = np.array(y_false)

        return X_true, y_true, X_false, y_false



    # 分割に最適の基準を見つける
    def find_best_split(self, X, y, feature_ids):
        features_num = X.shape[1]

        best_gain = 0
        best_feature_id = 0
        best_threshold = 0

        for feature_id in feature_ids:
            values = sorted(set(X[:, feature_id])) 

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2
                X_true, y_true, X_false, y_false = self.split(X, y, feature_id, threshold)
                gain = self.information_gain(y, y_true, y_false)

                # 暫定最良の閾値よりも良いものが見つかったら更新する
                if gain > best_gain:
                    best_gain = gain
                    best_feature_id = feature_id
                    best_threshold = threshold

        return best_feature_id, best_threshold


    # 再帰で決定木を定義
    def build_tree(self, X, y, feature_ids, depth):
        if (depth == self.max_depth) or (len(y) < self.min_split_num) or self.entropy(y) == 0:
            return mode(y)[0][0]
        
        feature_id, threshold = self.find_best_split(X, y, feature_ids)

        X_true, y_true, X_false, y_false = self.split(X, y, feature_id, threshold)
        if y_true.shape[0] == 0 or y_false.shape[0] == 0:
            return mode(y)[0][0]
        
        branch_true = self.build_tree(X_true, y_true, feature_ids, depth + 1)
        branch_false = self.build_tree(X_false, y_false, feature_ids, depth + 1)

        return Node(feature_id, threshold, branch_true, branch_false)


    # 精度accuracyを計算
    def accuracy_score(self, X, y):
        y_predict = self.predict(X)
        N = len(y)
        correct = 0
        for i in range(N):
            if y_predict[i] == y[i]:
                correct = correct + 1

        accuracy = correct / N

        return accuracy


# 2次元で図示 (Petal lengthをx軸、Petal widthをy軸として表示)
# 使用する特徴量も2つになる点には注意
def visualize(max_depth=None):
    iris_dataset = datasets.load_iris()
    petal_features = iris_dataset['data'][:, 2:]
    targets = iris_dataset['target']

    if max_depth is None:
        # 決定木の最大深度は制限しない
        clf = DecisionTreeClassifier()
    else:
        clf = DecisionTreeClassifier(max_depth=max_depth)
    
    clf.fit(petal_features, targets)

    # データの取りうる範囲 +-1 を計算する
    x_min = max(0, petal_features[:, 0].min() - 1)
    y_min = max(0, petal_features[:, 1].min() - 1)
    x_max = petal_features[:, 0].max() + 1
    y_max = petal_features[:, 1].max() + 1

    # 教師データの取りうる範囲でメッシュ状の座標を作る
    grid_interval = 0.2
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_interval),
        np.arange(y_min, y_max, grid_interval))

    # メッシュの座標を学習したモデルで判定させる
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 各点の判定結果をグラフに描画する
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.rainbow, alpha=0.4)

    # データもプロット
    for c in np.unique(targets):
        plt.scatter(petal_features[targets == c, 0],
                    petal_features[targets == c, 1])

    feature_names = iris_dataset['feature_names']
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    if max_depth is None:
        plt.title('Max Depth : No Limitation')
        #plt.show()
        plt.savefig('figures/Decision_Tree_no_limit_1.png')
    else:
        plt.title('Max Depth : ' + str(max_depth))
        #plt.show()
        plt.savefig('figures/Decision_Tree_depth_{}.png'.format(max_depth))



def main():
    iris_dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.3,  random_state=0)

    # 全ての特徴量を使用したときの精度を調べる
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    accuracy = decision_tree.accuracy_score(X_test, y_test)

    print('accuracy : {:.4f}'.format(accuracy))

    # 引数は決定木の深さの制限
    visualize()



if __name__ == '__main__':
    main()
