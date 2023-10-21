# coding:utf-8
import logging


class TrieTree(object):
    """
    Trie 树的基本方法，用途包括：
    - 词典 NER 的前向最大匹配计算
    - 繁简体词汇转换的前向最大匹配计算

    """

    def __init__(self):
        self.dict_trie = dict()
        self.depth = 0

    def add_node(self, word, typing):
        """向 Trie 树添加节点。

        Args:
            word(str): 词典中的词汇
            typing(str): 词汇类型

        Returns: None

        """
        word = word.strip()
        if word not in ['', '\t', ' ', '\r']:
            tree = self.dict_trie
            depth = len(word)
            word = word.lower()  # 将所有的字母全部转换成小写
            for char in word:
                if char in tree:
                    tree = tree[char]
                else:
                    tree[char] = dict()
                    tree = tree[char]
            if depth > self.depth:
                self.depth = depth
            if 'type' in tree and tree['type'] != typing:
                logging.warning(
                    '`{}` belongs to both `{}` and `{}`.'.format(
                        word, tree['type'], typing))
            else:
                tree['type'] = typing

    def build_trie_tree(self, dict_list, typing):
        """ 创建 trie 树 """
        for word in dict_list:
            self.add_node(word, typing)

    def search(self, word):
        """ 搜索给定 word 字符串中与词典匹配的 entity，
        返回值 None 代表字符串中没有要找的实体，
        如果返回字符串，则该字符串就是所要找的词汇的类型
        """
        tree = self.dict_trie
        res = None
        step = 0  # step 计数索引位置
        for char in word:
            if char in tree:
                tree = tree[char]
                step += 1
                if 'type' in tree:
                    res = (step, tree['type'])
            else:
                break
        if res:
            return res
        return 1, None

class LexiconNER(object):
    """ 构建基于 Trie 词典的前向最大匹配算法，做实体识别。

    Args:
        entity_dicts(dict): 每个类型对应的实体词典
            e.g.
            {
                'Person': ['张大山', '岳灵珊', '岳不群']
                'Organization': ['成都市第一人民医院', '四川省水利局']
            }
        text: str 类型，被搜索的文本内容。

    Return:
        entity_list: 基于字 token 的实体列表

    Examples:
        >>> entity_dicts = {
                'Person': ['张大山', '岳灵珊', '岳不群'],
                'Organization': ['成都市第一人民医院', '四川省水利局']}
        >>> lexicon_ner = LexiconNER(entity_dicts)
        >>> text = '岳灵珊在四川省水利局上班。'
        >>> result = lexicon_ner(text)
        >>> print(result)

        # [{'type': 'Person', 'text': '岳灵珊', 'offset': [0, 3]},
        #  {'type': 'Organization', 'text': '四川省水利局', 'offset': [4, 10]}]

    """

    def __init__(self, entity_dicts):
        self.trie_tree_obj = TrieTree()
        for typing, entity_list in entity_dicts.items():
            self.trie_tree_obj.build_trie_tree(entity_list, typing)

    def __call__(self, text):
        """
        标注数据，给定一个文本字符串，标注出所有的数据

        Args:
            text: 给定的文本 str 格式
        Return:
            entity_list: 标注的实体列表数据

        """

        record_list = list()  # 输出最终结果
        i = 0
        text_length = len(text)

        while i < text_length:
            pointer_orig = text[i: self.trie_tree_obj.depth + i]
            pointer = pointer_orig.lower()
            step, typing = self.trie_tree_obj.search(pointer)
            if typing is not None:
                record = {'type': typing,
                          'text': pointer_orig[0: step],
                          'start': i,
                          'end': step + i}
                record_list.append(record)
            i += step

        return record_list