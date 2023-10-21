from py2neo import Graph


class AnswerSearcher:
    def __init__(self):
        self.g = Graph("http://localhost:7474", auth=("gob", "gob"))
        self.num_limit = 20

    '''执行cypher查询，并返回相应结果'''

    def search_main(self, sqls):
        final_answers = []
        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']
            answers = []
            for query in queries:
                ress = self.g.run(query).data()
                answers += ress
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    '''根据对应的qustion_type，调用相应的回复模板'''

    def answer_prettify(self, question_type, answers):
        # print(answers)
        final_answer = []
        if not answers:
            return ''
        if question_type == 'hero_race':
            desc = [i['m.hero_race'] for i in answers]
            subject = answers[0]['m.hero_name']
            final_answer = '{0}的种族是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'hero_role':
            desc = [i['m.hero_role'] for i in answers]
            subject = answers[0]['m.hero_name']
            final_answer = '{0}的角色是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'hero_info':
            desc = [i['m.hero_info'] for i in answers]
            subject = answers[0]['m.hero_name']
            final_answer = '{0}的介绍是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'city_info':
            desc = [i['m.city_info'] for i in answers]
            subject = answers[0]['m.city_name']
            final_answer = '{0}的介绍是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'hero_title':
            desc = [i['m.hero_title'] for i in answers]
            subject = answers[0]['m.hero_name']
            final_answer = '{0}的别称是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'city_has_env':
            desc = [i['m.env_name'] for i in answers]
            subject = answers[0]['n.city_name']
            final_answer = '{0}的景色有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'hero_belong_city':
            desc = [i['n.city_name'] for i in answers]
            subject = answers[0]['m.hero_name']
            final_answer = '{0}所属区域是：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'city_has_hero':
            desc = [i['m.hero_name'] for i in answers]
            subject = answers[0]['n.city_name']
            final_answer = '{0}包含的英雄有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'rel_hero':
            desc = [i['r.name'] for i in answers]
            subject = answers[0]['m.hero_name']
            object = answers[0]['n.hero_name']
            final_answer = '{0}的{1}是：{2}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]), object)
        elif question_type == 'rel_no_hero':
            desc = [i['m.hero_name'] + "|" + i["n.hero_name"] for i in answers]
            subject = answers[0]['r.name']
            final_answer = '具有{0}关系的有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        return final_answer


if __name__ == '__main__':
    sqls = [{'question_type': 'rel_no_hero', 'sql': [
        "MATCH (m:hero)-[r:hero_rel]->(n:hero) where r.name = '徒弟' return m.hero_name, r.name, n.hero_name"]}]
    searcher = AnswerSearcher()
    final_answer = searcher.search_main(sqls)
    print(final_answer)
