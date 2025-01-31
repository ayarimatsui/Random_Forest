# 決定木とランダムフォレストの性能を、MNISTのデータセットで比較

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier


# ランダムフォレストのハイパーパラメータ(trees_num, bootstrap)を決める
# max_depth = 5 に固定して試す (一般的に決定木の深度は深い程良い、と言われているため、グリッドサーチでの確認は省略)
def grid_search_RF():
    dataset = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.3,  random_state=0)

    trees_num_list = [16, 32, 64, 128]  # ランダムフォレストに含まれる決定木の個数の候補
    bootstrap_list = [0.1, 0.3, 0.5, 0.7, 0.9]   # ブートストラップ法で復元するデータ量の元のデータ量に対する割合の候補

    best_acc = 0
    best_trees_num = None
    best_bootstrap = None
    with tqdm(total=len(trees_num_list)*len(bootstrap_list), desc='Progress') as pbar:
        for trees_num in trees_num_list:
            for bootstrap in bootstrap_list:
                random_forest = RandomForestClassifier(trees_num=trees_num, max_depth=5, bootstrap=bootstrap)
                random_forest.fit(X_train, y_train)
                acc = random_forest.accuracy_score(X_test, y_test)
                if acc > best_acc:
                    best_acc = acc
                    best_trees_num = trees_num
                    best_bootstrap = bootstrap
                pbar.update(1)
    
    print('best acc : {:.4f}    best trees_num : {}     best bootstrap : {}'.format(best_acc, best_trees_num, best_bootstrap))

    return best_trees_num, best_bootstrap


# 決定木、ランダムフォレストそれぞれのモデルにおいて、決定木の深さの制限を変えつつ、精度を比較
# グラフにまとめる
def compare_depth():

    dataset = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.3,  random_state=0)

    # ランダムフォレストに関して、最良のハイパーパラメータを調べる
    trees_num, bootstrap = grid_search_RF()

    # 決定木の深度の制限を変えて、調べる
    depth_list = [i for i in range(21)]  # 深さの制限0~20まで調べる

    dt_train_acc_list = []
    dt_test_acc_list = []
    rf_train_acc_list = []
    rf_test_acc_list = []

    for depth in tqdm(depth_list):
        print('***** max_depth = {} *****'.format(depth))
        # 決定木
        decision_tree = DecisionTreeClassifier(max_depth=depth)
        decision_tree.fit(X_train, y_train)
        dt_train_accuracy = decision_tree.accuracy_score(X_train, y_train)
        dt_test_accuracy = decision_tree.accuracy_score(X_test, y_test)
        # accuracyをリストに追加
        dt_train_acc_list.append(dt_train_accuracy)
        dt_test_acc_list.append(dt_test_accuracy)
        print('決定木       train accuracy : {:.4f}     test_accuracy : {:.4f}'.format(dt_train_accuracy, dt_test_accuracy))

        # ランダムフォレスト
        random_forest = RandomForestClassifier(trees_num=trees_num, max_depth=depth, bootstrap=bootstrap)
        random_forest.fit(X_train, y_train)
        rf_train_accuracy = random_forest.accuracy_score(X_train, y_train)
        rf_test_accuracy = random_forest.accuracy_score(X_test, y_test)
        # accuracyをリストに追加
        rf_train_acc_list.append(rf_train_accuracy)
        rf_test_acc_list.append(rf_test_accuracy)
        print('ランダムフォレスト       train accuracy : {:.4f}     test_accuracy : {:.4f}'.format(rf_train_accuracy, rf_test_accuracy))

    # グラフの描画
    plt.plot(depth_list, dt_train_acc_list, label='Decision Tree - train accuracy', color='r')
    plt.plot(depth_list, dt_test_acc_list, label='Decision Tree - test accuracy', color='g')
    plt.plot(depth_list, rf_train_acc_list, label='Random Forest - train accuracy', color='y')
    plt.plot(depth_list, rf_test_acc_list, label='Random Forest - test accuracy', color='b')

    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 21, 2))
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.title('Max Depth of Decision Trees and Accuracy')
    # グラフを保存
    plt.savefig('figures/mnist/max_depth_&_accuracy.png')



# 決定木とランダムフォレストの正解率、学習時間、推論時間、汎化性能を調べる
# 引数は、compare_depthの結果より決定
def compare_performance(trees_num, max_depth, bootstrap):

    dataset = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.3,  random_state=0)

    # 決定木
    print('#####　決定木の性能  #####')
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    dt_lr_start = time.time()  # 学習開始時間を記録
    decision_tree.fit(X_train, y_train)
    dt_lr_time = time.time() - dt_lr_start  # 学習時間
    dt_est_start = time.time()  # 推論開始時間を記録
    y_est = decision_tree.predict(X_test)
    dt_est_time = time.time() - dt_est_start  # 推論時間
    print('学習時間 : {:.6f} [sec]     推論時間 : {:.6f} [sec]'.format(dt_lr_time, dt_est_time))
    dt_train_accuracy = decision_tree.accuracy_score(X_train, y_train)
    dt_test_accuracy = decision_tree.accuracy_score(X_test, y_test)
    print('train accuracy : {:.4f}     test_accuracy : {:.4f}'.format(dt_train_accuracy, dt_test_accuracy))

    # ランダムフォレスト
    print('#####　ランダムフォレストの性能　#####')
    random_forest = RandomForestClassifier(trees_num=trees_num, max_depth=max_depth, bootstrap=bootstrap)
    rf_lr_start = time.time()  # 学習開始時間を記録
    random_forest.fit(X_train, y_train)
    rf_lr_time = time.time() - rf_lr_start  # 学習時間
    rf_est_start = time.time()  # 推論開始時間を記録
    y_est = random_forest.predict(X_test)
    rf_est_time = time.time() - rf_est_start  # 推論時間
    print('学習時間 : {:.6f} [sec]     推論時間 : {:.6f} [sec]'.format(rf_lr_time, rf_est_time))
    rf_train_accuracy = random_forest.accuracy_score(X_train, y_train)
    rf_test_accuracy = random_forest.accuracy_score(X_test, y_test)
    print('train accuracy : {:.4f}     test_accuracy : {:.4f}'.format(rf_train_accuracy, rf_test_accuracy))




if __name__ == '__main__':
    #compare_depth()
    # compare_depth()の実行結果より、
    # trees_num = 128, max_depth = 10, bootstrap = 0.5　に決定
    compare_performance(128, 10, 0.5)


    '''
    実行結果

    #####　決定木の性能  #####
    学習時間 : 8.753003 [sec]     推論時間 : 0.001576 [sec]
    train accuracy : 1.0000     test_accuracy : 0.8667
    
    #####　ランダムフォレストの性能　#####
    学習時間 : 72.597565 [sec]     推論時間 : 0.244916 [sec]
    train accuracy : 1.0000     test_accuracy : 0.9444

    '''