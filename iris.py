# 決定木とランダムフォレストの性能を、アヤメのデータセットで比較

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier


# 2次元で図示 (Petal lengthをx軸、Petal widthをy軸として表示)
# 使用する特徴量も2つになる点には注意
# 第一引数のmodelには、'decision_tree'または'ranfdom_forest'が入る

def visualize(model, max_depth=None):
    iris_dataset = datasets.load_iris()
    petal_features = iris_dataset['data'][:, 2:]
    targets = iris_dataset['target']

    if max_depth is None:
        # 決定木の最大深度は制限しない
        # アヤメのデータセットの場合は、データ数やクラス数が少ないため、深度を制限しなくても計算時間はあまりかからない
        if model == 'decision_tree':
            clf = DecisionTreeClassifier()
        else:
            clf = RandomForestClassifier()
    else:
        if model == 'decision_tree':
            clf = DecisionTreeClassifier(max_depth=max_depth)
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
        plt.savefig('figures/iris/{}_no_limit.png'.format(model))
    else:
        plt.title('Max Depth : ' + str(max_depth))
        plt.savefig('figures/iris/{}_depth_{}.png'.format(model, max_depth))



def main():

    dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.3,  random_state=0)

    # 決定木の深度の制限が1~3、制限なしの各場合について調べる
    depth_list = [1, 2, 3, None] 
    for depth in depth_list:
        print('######### max_depth = {} #########'.format(depth))
        # 全ての特徴量を使用したときの精度、学習時間、推論時間、汎化性能を調べる
        # 決定木
        decision_tree = DecisionTreeClassifier(max_depth=depth)
        dt_lr_start = time.time()  # 学習開始時間を記録
        decision_tree.fit(X_train, y_train)
        dt_lr_time = time.time() - dt_lr_start  # 学習時間
        dt_est_start = time.time()  # 推論開始時間を記録
        y_est = decision_tree.predict(X_test)
        dt_est_time = time.time() - dt_est_start  # 推論時間
        print('決定木       学習時間 : {:.6f} [sec]     推論時間 : {:.6f} [sec]'.format(dt_lr_time, dt_est_time))
        dt_train_accuracy = decision_tree.accuracy_score(X_train, y_train)
        dt_test_accuracy = decision_tree.accuracy_score(X_test, y_test)
        print('決定木       train accuracy : {:.4f}     test_accuracy : {:.4f}'.format(dt_train_accuracy, dt_test_accuracy))

        # ランダムフォレスト
        random_forest = RandomForestClassifier(max_depth=depth)
        rf_lr_start = time.time()  # 学習開始時間を記録
        random_forest.fit(X_train, y_train)
        rf_lr_time = time.time() - rf_lr_start  # 学習時間
        rf_est_start = time.time()  # 推論開始時間を記録
        y_est = random_forest.predict(X_test)
        rf_est_time = time.time() - rf_est_start  # 推論時間
        print('ランダムフォレスト       学習時間 : {:.6f} [sec]     推論時間 : {:.6f} [sec]'.format(rf_lr_time, rf_est_time))
        rf_train_accuracy = random_forest.accuracy_score(X_train, y_train)
        rf_test_accuracy = random_forest.accuracy_score(X_test, y_test)
        print('ランダムフォレスト       train accuracy : {:.4f}     test_accuracy : {:.4f}'.format(rf_train_accuracy, rf_test_accuracy))

        # 使用する特徴量を2つ(Petal legth, Petal width)に絞って、二次元で可視化
        visualize('decision_tree', max_depth=depth)
        visualize('random_forest', max_depth=depth)


if __name__ == '__main__':
    main()


'''
実行例

######### max_depth = 1 #########
決定木       学習時間 : 0.021221 [sec]     推論時間 : 0.000036 [sec]
決定木       train accuracy : 0.6952     test_accuracy : 0.6000
ランダムフォレスト       学習時間 : 0.612986 [sec]     推論時間 : 0.003086 [sec]
ランダムフォレスト       train accuracy : 0.6952     test_accuracy : 0.6000

######### max_depth = 2 #########
決定木       学習時間 : 0.038370 [sec]     推論時間 : 0.000053 [sec]
決定木       train accuracy : 0.9619     test_accuracy : 0.9111
ランダムフォレスト       学習時間 : 0.982143 [sec]     推論時間 : 0.003865 [sec]
ランダムフォレスト       train accuracy : 0.9714     test_accuracy : 0.9556

######### max_depth = 3 #########
決定木       学習時間 : 0.040380 [sec]     推論時間 : 0.000056 [sec]
決定木       train accuracy : 0.9810     test_accuracy : 0.9778
ランダムフォレスト       学習時間 : 1.206527 [sec]     推論時間 : 0.004530 [sec]
ランダムフォレスト       train accuracy : 0.9714     test_accuracy : 0.9778

######### max_depth = None #########
決定木       学習時間 : 0.043978 [sec]     推論時間 : 0.000058 [sec]
決定木       train accuracy : 1.0000     test_accuracy : 0.9778
ランダムフォレスト       学習時間 : 1.477691 [sec]     推論時間 : 0.005911 [sec]
ランダムフォレスト       train accuracy : 1.0000     test_accuracy : 0.9778

'''