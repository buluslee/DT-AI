# coding=utf-8
"""
将数据存入到neo4j数据库里面
py2neo版本：py2neo-2021.2.3
neo4j版本：neo4j-4.4.5
"""
import os
from py2neo import Graph, Node, NodeMatcher
import pandas as pd
import datetime


def save_to_txt(file_path, data):
    with open(file_path, 'w', encoding="utf-8") as fp:
        fp.write("\n".join(data))


class LoLGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data')
        # auth=(用户名, 密码)
        self.g = Graph("http://localhost:7474", auth=("gob", "gob"))
        self.g.delete_all()  # 先清空数据库，按需执行

    def get_data(self):
        """获取数据"""
        hero_names = []  # 英雄名
        hero_titles = []  # 英雄名别称
        hero_races = []  # 种族
        hero_roles = []  # 角色
        hero_release_dates = []  # 发布时间
        hero_infos = []  # 英雄简介

        env_names = []  # 环境名
        env_infos = []  # 环境描述

        city_names = []  # 区域名称
        city_infos = []  # 区域描述

        hero_belong_city = []

        env_belong_city = []

        rels = {}

        """读取英雄信息"""
        hero_data = pd.read_csv(os.path.join(self.data_path, "raw_data/hero_info.csv"), header=None)
        for d in hero_data.itertuples():
            hero_names.append(d[2].strip())
            hero_titles.append(d[3].strip())
            hero_roles.append(d[5].strip())
            hero_races.append(d[6].strip())
            hero_release_dates.append(datetime.datetime.strptime(d[7].strip(), "%Y/%m/%d").strftime('%Y-%m-%d'))
            hero_infos.append(d[8].strip().replace("\n", ""))
            hero_belong_city.append((d[2].strip(), d[4].strip()))

        """读取城市信息"""
        city_data = pd.read_csv(os.path.join(self.data_path, "raw_data/city_info.csv"))
        for d in city_data.itertuples():
            city_names.append(d[1])
            city_infos.append(d[3].strip().replace("\n", ""))

        """读取环境信息"""
        env_data = pd.read_csv(os.path.join(self.data_path, "raw_data/environment_info.csv"))
        for d in env_data.itertuples():
            env_names.append(d[1])
            env_infos.append(str(d[2]).strip().replace("\n", "").replace("nan", ""))
            env_belong_city.append((d[1], d[3].strip()))

        """读取英雄关系信息"""
        rel_data = pd.read_csv(os.path.join(self.data_path, "raw_data/relation_info.csv"), header=None)
        for d in rel_data.itertuples():
            if d[2] not in rels:
                rels[d[2]] = [(d[1], d[3])]
            else:
                rels[d[2]].append((d[1], d[3]))

        rel_labels = []
        for i in rels.keys():
            if i not in rel_labels and i != '未知':
                rel_labels.append(i)

        save_to_txt(os.path.join(self.data_path, "mid_data/rels.txt"), rel_labels)

        data = {
            "hero_names": hero_names,
            "hero_titles": hero_titles,
            "hero_races": hero_races,
            "hero_roles": hero_roles,
            "hero_release_dates": hero_release_dates,
            "hero_infos": hero_infos,
            "city_names": city_names,
            "city_infos": city_infos,
            "env_names": env_names,
            "env_infos": env_infos,
            "hero_belong_city": hero_belong_city,
            "env_belong_city": env_belong_city,
            "rels": rels,
        }

        entity_cols = ["hero_names",
                       "hero_titles",
                       "hero_races",
                       "hero_roles",
                       "hero_release_dates",
                       "city_names"]
        for n in entity_cols:
            save_to_txt(os.path.join(self.data_path, "mid_data", n) + ".txt", data[n])
        return data

    '''建立英雄节点'''

    def create_hero_node(self, data):
        count = 0
        total = len(data["hero_names"])
        for i in range(total):
            node = Node("hero",
                        hero_name=data["hero_names"][i],
                        hero_title=data["hero_titles"][i],
                        hero_race=data["hero_races"][i],
                        hero_role=data["hero_roles"][i],
                        hero_release_date=data["hero_release_dates"][i],
                        hero_info=data["hero_infos"][i],
                        )
            self.g.create(node)
            count += 1
            print(count)
        return

    '''建立区域节点'''

    def create_city_node(self, data):
        count = 0
        total = len(data["city_names"])
        for i in range(total):
            node = Node("city",
                        city_name=data["city_names"][i],
                        city_info=data["city_infos"][i],
                        )
            self.g.create(node)
            count += 1
            print(count)
        return

    def create_graphrels(self, data):
        self.create_relationship(
            "hero",
            "city",
            "hero_name",
            "city_name",
            data["hero_belong_city"],
            "hero_belong_city",
            "英雄所属区域"
        )
        self.create_relationship(
            "environment",
            "city",
            "env_name",
            "city_name",
            data["env_belong_city"],
            "env_belong_city",
            "环境所属区域"
        )
        rel_data = data["rels"]
        for k, v in rel_data.items():
            if k == "未知":
                continue
            self.create_relationship(
                "hero",
                "hero",
                "hero_name",
                "hero_name",
                v,
                "hero_rel",
                k
            )

    '''创建实体关联边'''

    def create_relationship(self,
                            start_node,
                            end_node,
                            s_name,
                            e_name,
                            edges,
                            rel_type,
                            rel_name):
        """
        :param start_node: 关联起始的节点名
        :param end_node: 关联结尾的节点名
        :param s_name: 起始节点查询属性
        :param e_name: 结尾节点查询属性
        :param edges:  关联数据
        :param rel_type: 关系类型
        :param rel_name: 关系名
        :return:
        """
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0].replace('·', '-')
            q = edge[1].replace('·', '-')
            query = "match(p:%s),(q:%s) where p.%s='%s'and q.%s='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, s_name, p, e_name, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

    '''建立环境节点'''

    def create_env_node(self, data):
        count = 0
        total = len(data["env_names"])
        for i in range(total):
            node = Node("environment",
                        env_name=data["env_names"][i],
                        env_info=data["env_infos"][i],
                        )
            self.g.create(node)
            count += 1
            print(count)
        return

    def get_id(self):
        """根据节点获取节点id"""
        matcher = NodeMatcher(self.g)
        # 查询节点为city，city_name为德玛西亚的节点
        result = matcher.match('city', city_name="德玛西亚")
        for i in result.all():
            print(i.identity)

    def delete_node(self):
        """这里是删除为city的所有节点，可根据需要自己改写。
        删除节点时要先删除关系，再删除节点
        """
        self.g.run("match (n:city) detach delete n")


if __name__ == '__main__':
    lolGraph = LoLGraph()
    data = lolGraph.get_data()
    lolGraph.create_hero_node(data)
    lolGraph.create_city_node(data)
    lolGraph.create_env_node(data)
    lolGraph.create_graphrels(data)
