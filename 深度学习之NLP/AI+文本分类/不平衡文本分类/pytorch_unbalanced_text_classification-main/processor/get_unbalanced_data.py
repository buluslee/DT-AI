"""
从平衡的数据中获取不平衡的数据
"""
from random import shuffle


class GetUnbalancedData:

    def read_file(self, path):
        # 获取分类数据和标签，默认格式：标签\t文本\n
        with open(path, 'r') as fp:
            lines = fp.read().strip().split('\n')
        return lines

    def count_data(self, lines):
        # 统计数据总量以及每类数据总量及其所占比例
        data = {}
        total = len(lines)
        for line in lines:
            line = line.split("\t")
            label = line[0]
            text = line[1]
            if label not in data:
                data[label] = [text]
            else:
                data[label].append(text)

        print("总共有数据：{}".format(total))
        for k, v in data.items():
            print(k + "\t" + str(len(v)) + "\t" + "{:.2f}%".format(len(v) / total * 100))
        return data

    def get_data(self, nums, data):
        # 获得不平衡的数据集
        unbalanced_data = {}
        for i, (label, texts) in enumerate(data.items()):
            ids = list(range(len(texts)))
            shuffle(ids)
            tmp_ids = ids[:nums[i]]
            tmp_texts = [texts[i] for i in tmp_ids]
            unbalanced_data[label] = tmp_texts
        total = sum(nums)
        print("总共有数据：{}".format(total))
        for k, v in unbalanced_data.items():
            print(k + "\t" + str(len(v)) + "\t" + "{:.2f}%".format(len(v) / total * 100))
        res = []  # 用于存入数据到txt中
        for k,v in unbalanced_data.items():
            label = k
            texts = v
            for text in texts:
                res.append([label, text])
        with open('../data/THUCNews/cnews.train.unbalanced.txt', 'w') as fp:
            fp.write("\n".join(["\t".join(i) for i in res]))
        return unbalanced_data


if __name__ == '__main__':
    getUnbalancedData = GetUnbalancedData()
    lines = getUnbalancedData.read_file('../data/THUCNews/cnews.train.txt')
    data = getUnbalancedData.count_data(lines)
    # ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    nums = [5000, 4000, 3000, 2000, 1000, 500, 400, 300, 200, 100]
    unblanced_data = getUnbalancedData.get_data(nums, data)
