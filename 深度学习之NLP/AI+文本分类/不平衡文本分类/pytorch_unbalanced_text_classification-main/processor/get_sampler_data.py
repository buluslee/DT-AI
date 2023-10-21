from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def get_data():
    path = '../data/THUCNews/cnews.train.unbalanced.txt'
    # 首先分别得到适用文本列表及标签列表
    train_x = []
    train_y = []
    with open(path, 'r') as fp:
        data = fp.read().strip().split('\n')
        for d in data:
            d = d.split('\t')
            train_x.append([d[1]])
            train_y.append([d[0]])
    return train_x, train_y


def get_randomOverSampler_data(train_x, train_y):
    """随机过采样"""
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(train_x, train_y)
    counter_resampled = Counter(y_resampled)
    print(counter_resampled)
    data = [[i, j[0]] for i, j in zip(y_resampled, X_resampled)]
    with open('../data/THUCNews/cnews.train.randomOverSampler.txt', 'w') as fp:
        fp.write("\n".join(["\t".join(i) for i in data]))


def get_randomUnderSampler_data(train_x, train_y):
    rus = RandomUnderSampler(random_state=0)

    X_resampled, y_resampled = rus.fit_resample(train_x, train_y)
    counter_resampled = Counter(y_resampled)
    print(counter_resampled)
    data = [[i, j[0]] for i, j in zip(y_resampled, X_resampled)]
    with open('../data/THUCNews/cnews.train.randomUnderSampler.txt', 'w') as fp:
        fp.write("\n".join(["\t".join(i) for i in data]))


if __name__ == '__main__':
    """
        标签及分布情况：
        labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
        nums = [5000, 4000, 3000, 2000, 1000, 500, 400, 300, 200, 100]
    """
    train_x, train_y = get_data()
    get_randomOverSampler_data(train_x, train_y)
    get_randomUnderSampler_data(train_x, train_y)
