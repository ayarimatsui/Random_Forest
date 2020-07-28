# ランダムフォレストの実装

import numpy as np
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



def main():
    iris_dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25,  random_state=0)

    # 各決定木のmax_depthが小さくても、Random Forestならば、高い精度が出せることが分かる
    random_forest = RandomForestClassifier(max_depth=3)
    random_forest.fit(X_train, y_train)

    accuracy = random_forest.accuracy_score(X_test, y_test)

    print('accuracy : {:.4f}'.format(accuracy))

    



if __name__ == '__main__':
    main()