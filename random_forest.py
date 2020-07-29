# ランダムフォレストの実装

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeClassifier


# ランダムフォレスト
class RandomForestClassifier():
   
    def __init__(self, trees_num=64, features_num=np.sqrt, max_depth=float('inf'), bootstrap=0.9):  # trees_num, features_num, bootstrapはハイパーパラメータ
        # trees_num : ランダムフォレストに含まれる決定木の数
        self.trees_num = trees_num
        # features_num : 各分岐でランダムに選択する特徴量の数
        self.features_num = features_num
        # max_depth : 決定木の深さの制限
        self.max_depth = max_depth
        self.min_split_num = 2  # さらに細かく分割するときに最低限必要なサンプルデータの数
        self.bootstrap = bootstrap  # ブートストラップ法によるデータの生成
        self.forest = []


    # 2つの配列をまとめてシャッフル
    def shuffle(self, list1, list2):
        seed_state = np.random.get_state()
        np.random.shuffle(list1)
        np.random.set_state(seed_state)
        np.random.shuffle(list2)


    def fit(self, X, y):
        self.forest = []
        N = len(y)
        N_sub_data = int(N * self.bootstrap)
        
        for i in range(self.trees_num):
            self.shuffle(X, y)
            X_sub = X[:N_sub_data]
            y_sub = y[:N_sub_data]

            decision_tree = DecisionTreeClassifier(self.features_num, self.max_depth)
            decision_tree.fit(X_sub, y_sub)
            # 得られた決定木をforestのリストに追加
            self.forest.append(decision_tree)


    def predict(self, X):
        N = X.shape[0]
        n_trees = len(self.forest)
        predicts = np.zeros([n_trees, N])
        for i in range(n_trees):
            predicts[i] = self.forest[i].predict(X)

        return mode(predicts)[0][0]


    # 精度を計算する
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
        clf = RandomForestClassifier()
    else:
        clf = RandomForestClassifier(max_depth=max_depth)
    
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
        plt.savefig('figures/Random_Forest_no_limit.png')
    else:
        plt.title('Max Depth : ' + str(max_depth))
        #plt.show()
        plt.savefig('figures/Random_Forest_depth_{}.png'.format(max_depth))



def main():
    iris_dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.3,  random_state=0)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    accuracy = random_forest.accuracy_score(X_test, y_test)

    print('accuracy : {:.4f}'.format(accuracy))

    



if __name__ == '__main__':
    main()
    visualize()