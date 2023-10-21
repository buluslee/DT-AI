import os
from trie_tree import LexiconNER


class QuestionClassifier:
    def __init__(self):
        cur_dir = 'data'
        entity_dicts = {}
        # 　特征词路径
        self.city_names_path = os.path.join(cur_dir, 'mid_data/city_names.txt')
        self.hero_names_path = os.path.join(cur_dir, 'mid_data/hero_names.txt')
        self.hero_races_path = os.path.join(cur_dir, 'mid_data/hero_races.txt')
        self.hero_release_dates_path = os.path.join(cur_dir, 'mid_data/hero_release_dates.txt')
        self.hero_roles_path = os.path.join(cur_dir, 'mid_data/hero_roles.txt')
        self.hero_titles_path = os.path.join(cur_dir, 'mid_data/hero_titles.txt')
        self.rels_path = os.path.join(cur_dir, 'mid_data/rels.txt')
        # 加载特征词
        entity_dicts = {
            "city_names": [i.strip() for i in open(self.city_names_path, encoding="utf-8") if i.strip()],
            "hero_names": [i.strip() for i in open(self.hero_names_path, encoding="utf-8") if i.strip()],
            "hero_races": [i.strip() for i in open(self.hero_races_path, encoding="utf-8") if i.strip()],
            "hero_release_dates": [i.strip() for i in open(self.hero_release_dates_path, encoding="utf-8") if
                                   i.strip()],
            "hero_roles": [i.strip() for i in open(self.hero_roles_path, encoding="utf-8") if i.strip()],
            "hero_titles": [i.strip() for i in open(self.hero_titles_path, encoding="utf-8") if i.strip()],
            "rels": [i.strip() for i in open(self.rels_path, encoding="utf-8") if i.strip()]
        }

        self.lexicon_ner = LexiconNER(entity_dicts)
        # 问句疑问词
        self.hero_race_qwds = ["种族"]
        self.hero_role_qwds = ["角色"]
        self.hero_title_qwds = ["别称", "别名", "称号"]
        self.hero_info_qwds = ["基本信息", '简介', '介绍', '信息']
        self.city_qwds = ['区域', '城市']
        self.env_qwds = ['风景', '景色', '建筑']
        self.rel_qwds = entity_dicts['rels']

        print('model init finished ......')

        return

    '''分类主函数'''

    def classify(self, question):
        data = {}
        entities = self.lexicon_ner(question)
        entity_dict = {}
        # 收集问句当中所涉及到的实体类型
        types = set()
        for entity in entities:
            types.add(entity['type'])
            if entity['text'] not in entity_dict:
                entity_dict[entity['text']] = [entity['type']]

            else:
                if entity['type'] not in entity_dict[entity['text']]:
                    entity_dict[entity['text']].append(entity['type'])

        types = list(types)
        data['args'] = entity_dict

        uestion_type = 'others'
        question_types = []
        # 查询英雄的种族
        if self.check_words(self.hero_race_qwds, question) and "hero_names" in types:
            question_type = 'hero_race'
            question_types.append(question_type)

        # 查询英雄的角色
        if self.check_words(self.hero_role_qwds, question) and "hero_names" in types:
            question_type = 'hero_role'
            question_types.append(question_type)

        # 查询英雄的介绍
        if self.check_words(self.hero_info_qwds, question):
            if "hero_names" in types:
                question_type = 'hero_info'
                question_types.append(question_type)
            elif "city_names" in types:
                question_type = 'city_info'
                question_types.append(question_type)

        # 查询英雄的别称
        if self.check_words(self.hero_title_qwds, question) and "hero_names" in types:
            question_type = 'hero_title'
            question_types.append(question_type)

        # 查询区域的景色
        if self.check_words(self.env_qwds, question) and "city_names" in types:
            question_type = 'city_has_env'
            question_types.append(question_type)

        # 查询英雄所属的区域
        if self.check_words(self.city_qwds, question):
            if "hero_names" in types:
                question_type = 'hero_belong_city'
                question_types.append(question_type)
            else:
                # 查询某区域包含什么英雄
                if "英雄" in question:
                    question_type = 'city_has_hero'
                    question_types.append(question_type)

        # 查询某种关系的英雄
        if self.check_words(self.rel_qwds, question):
            if "rels" in types and "hero_names" in types:
                # 既出现关系，又出现英雄实体
                question_type = 'rel_hero'
                question_types.append(question_type)
            elif "rels" in types and "hero_names" not in types:
                # 只出现关系，不出现英雄实体
                question_type = 'rel_no_hero'
                question_types.append(question_type)


        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    '''基于特征词进行分类'''

    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False


if __name__ == '__main__':
    handler = QuestionClassifier()
    """
        input an question: 
        question = "德玛西亚有什么景色吗"
        {'args': {'德玛西亚': ['city_names']}, 'question_types': ['city_has_env']}
    """
    # while 1:
    #     question = input('input an question:')
    #     data = handler.classify(question)
    #     print(data)
    question = "盖伦的别称是什么"
    question = "德玛西亚区域有哪些英雄"
    print(handler.classify(question))
