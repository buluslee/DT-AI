class QuestionPaser:
    '''构建实体节点'''

    def build_entitydict(self, args):
        """
        input an question:
        question = "德玛西亚有什么景色吗"
        {'args': {'德玛西亚': ['city_names']}, 'question_types': ['env_info']}
        :param args:
        :return:
        """
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)
        # print(entity_dict)
        return entity_dict

    '''解析主函数'''

    def parser_main(self, res_classify):
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        # print(entity_dict)
        question_types = res_classify['question_types']
        sqls = []
        for question_type in question_types:
            sql_ = {}
            sql_['question_type'] = question_type
            sql = []
            if question_type == 'hero_race':
                sql = self.sql_transfer(question_type, entity_dict.get('hero_names'))
            elif question_type == 'hero_role':
                sql = self.sql_transfer(question_type, entity_dict.get('hero_names'))
            elif question_type == 'hero_info':
                sql = self.sql_transfer(question_type, entity_dict.get('hero_names'))
            elif question_type == 'hero_title':
                sql = self.sql_transfer(question_type, entity_dict.get('hero_names'))
            elif question_type == 'city_info':
                sql = self.sql_transfer(question_type, entity_dict.get('city_names'))
            elif question_type == 'city_has_env':
                sql = self.sql_transfer(question_type, entity_dict.get('city_names'))
            elif question_type == 'hero_belong_city':
                sql = self.sql_transfer(question_type, entity_dict.get('hero_names'))
            elif question_type == 'city_has_hero':
                sql = self.sql_transfer(question_type, entity_dict.get('city_names'))
            elif question_type == 'rel_hero':
                sql = self.sql_transfer(question_type, entity_dict.get('hero_names'), rels=entity_dict.get("rels"))
            elif question_type == 'rel_no_hero':
                sql = self.sql_transfer(question_type, entity_dict.get('rels'))

            if sql:
                sql_['sql'] = sql

                sqls.append(sql_)

        return sqls

    '''针对不同的问题，分开进行处理'''

    def sql_transfer(self, question_type, entities, *args, **kwargs):
        # print(args, kwargs)
        if not entities:
            return []

        # 查询语句
        sql = []
        # 查询疾病的原因
        if question_type == 'hero_race':
            sql = ["MATCH (m:hero) where m.hero_name = '{0}' return m.hero_name, m.hero_race".format(i) for i in entities]
        elif question_type == 'hero_role':
            sql = ["MATCH (m:hero) where m.hero_name = '{0}' return m.hero_name, m.hero_role".format(i) for i in entities]
        elif question_type == 'hero_info':
            sql = ["MATCH (m:hero) where m.hero_name = '{0}' return m.hero_name, m.hero_info".format(i) for i in entities]
        elif question_type == 'city_info':
            sql = ["MATCH (m:city) where m.city_name = '{0}' return m.city_name, m.city_info".format(i) for i in entities]
        elif question_type == 'hero_title':
            sql = ["MATCH (m:hero) where m.hero_name = '{0}' return m.hero_name, m.hero_title".format(i) for i in entities]
        elif question_type == 'city_has_env':
            sql = ["MATCH (m:environment)-[r:env_belong_city]->(n:city) where n.city_name = '{0}' return m.env_name, n.city_name".format(i) for i in entities]
        elif question_type == 'hero_belong_city':
            sql = [
                "MATCH (m:hero)-[r:hero_belong_city]->(n:city) where m.hero_name = '{0}' return m.hero_name, n.city_name".format(
                    i) for i in entities]
        elif question_type == 'city_has_hero':
            sql = [
                "MATCH (m:hero)-[r:hero_belong_city]->(n:city) where n.city_name = '{0}' return m.hero_name, n.city_name".format(
                    i) for i in entities]
        elif question_type == 'rel_hero':
            ent = entities[0]
            rel = kwargs["rels"][0]
            sql = [
                "MATCH (m:hero)-[r:hero_rel]->(n:hero) where m.hero_name = '{0}' and r.name = '{1}' return m.hero_name, r.name, n.hero_name".format(
                    ent, rel)]
        elif question_type == "rel_no_hero":
            sql = [
                "MATCH (m:hero)-[r:hero_rel]->(n:hero) where r.name = '{0}' return m.hero_name, r.name, n.hero_name".format(
                    i) for i in entities]
        return sql


if __name__ == '__main__':
    args = {'args': {'盖伦': ['hero_names']}, 'question_types': ['hero_race']}
    args = {'args': {'孙悟空': ['hero_names'], "徒弟": ['rels']}, 'question_types': ['rel_hero']}
    args = {'args': {"徒弟": ['rels']}, 'question_types': ['rel_no_hero']}
    handler = QuestionPaser()
    # print(handler.build_entitydict(args['args']))
    sql = handler.parser_main(args)
    print(sql)
