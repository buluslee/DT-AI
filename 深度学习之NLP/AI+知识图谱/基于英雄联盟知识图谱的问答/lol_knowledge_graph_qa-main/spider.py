"""
获取英雄联盟数据爬虫
"""
import requests
import json
import pandas as pd


def get_json(url):
    headers = {'user-agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf8'
    d_json = response.json()
    return d_json


def get_city_info():
    city_url = 'https://yz.lol.qq.com/v1/zh_cn/faction-browse/index.json'
    city_json = get_json(city_url)
    city_name = []  # 城市名
    city_slug = []  #
    city_engname = []  # 英文名
    city_context = []  # 城市介绍

    environment_name = []  # 环境
    environment_belong_city = []  # 环境所属城市
    for i in range(len(city_json['factions'])):
        """
        {   
          "type": "faction",
          "name": "巨神峰",
          "slug": "mount-targon",
          "image": {
            "title": "mount-targon-splash",
            "subtitle": "",
            "description": "",
            "uri": "https://game.gtimg.cn/images/lol/universe/v1/assets/blt1ab39bfee4a3057d-mount-targon_splash.jpg",
            "encoding": "image/jpeg",
            "width": null,
            "height": null,
            "x": null,
            "y": null,
            "featured-champions": []
          },
          "echelon": 1,
          "associated-champions": null
        },
        """
        the_url = 'https://yz.lol.qq.com/v1/zh_cn/factions/{}/index.json'.format(city_json['factions'][i]['slug'])
        the_json = get_json(the_url)
        city_name.append(the_json['faction']['name'])
        city_slug.append(the_json['faction']['slug'])
        city_context.append(the_json['faction']['overview']['short'].replace('<p>', '').replace('</p>', '\n'))
        city_engname.append(the_json['id'])
        temp = 0
        for i in the_json['modules']:
            if 'slug' in i:
                if i['slug'] == the_json['id'] + '-environment':
                    for j in i['assets']:
                        environment_name.append(j['title'])
                        environment_belong_city.append(the_json['faction']['name'])
    city_info = pd.DataFrame({'city_name': city_name,
                              'city_slug': city_slug,
                              'city_context': city_context,
                              'city_engname': city_engname})
    environment_info = pd.DataFrame({'environment_name': environment_name,
                                     'environment_belong_city': environment_belong_city})
    # print(info)
    # print(info1)
    return city_info, environment_info


def get_hero_info(city_info):
    hero_url = 'https://yz.lol.qq.com/v1/zh_cn/search/index.json'  # 主要是获取英雄名
    hero_json = get_json(hero_url)
    names = []  # 英雄姓名
    titles = []  # 英雄别称
    cities = []
    roles = []
    races = []
    release_dates = []
    intros = []

    names_related_champions = []
    related_champions = []
    relations = []
    for i in range(len(hero_json['champions'])):
        name = hero_json['champions'][i]['name']
        names.append(name)
        slug = hero_json['champions'][i]['slug']
        city_other_name = hero_json['champions'][i]['associated-faction-slug']
        if city_other_name == 'unaffiliated':
            city_name = '未知'
        elif city_other_name == 'runeterra':
            city_name = '符文之地'
        else:
            city_index = list(city_info['city_slug']).index(city_other_name)
            city_name = city_info['city_name'][city_index]
        cities.append(city_name)
        # 获取英雄的信息
        hero_detail_url = 'https://yz.lol.qq.com/v1/zh_cn/champions/{}/index.json'.format(slug)
        hero_detail_data = get_json(hero_detail_url)
        title = hero_detail_data['title']
        titles.append(title)
        release_dates.append(hero_detail_data['champion']['release-date'])
        tmp = []

        # 这里种族应该全是空的
        if 'races' in hero_detail_data['champion']:
            for i in hero_detail_data['champion']['races']:
                tmp.append(i['name'])
        if len(tmp) == 0:
            tmp.append('未知')
        races.append("#;#".join(tmp))

        tmp = []
        if 'roles' in hero_detail_data['champion']:
            for i in hero_detail_data['champion']['roles']:
                tmp.append(i['name'])
        if len(tmp) == 0:
            tmp.append('未知')
        roles.append("#;#".join(tmp))

        for i in hero_detail_data['related-champions']:
            names_related_champions.append(name)
            relations.append("朋友")
            related_champions.append(i['name'])

        intros.append(hero_detail_data['champion']['biography']['short'].replace('<p>', '').replace('</p>', '\n'))

    print(len(names),
          len(titles),
          len(cities),
          len(roles),
          len(races),
          len(release_dates),
          len(intros))
    hero_info = pd.DataFrame({'name': names,
                             'title': titles,
                             'city': cities,
                             'role': roles,
                             'race': races,
                             'release_date': release_dates,
                             'intro': intros,
                             })

    champion_relations_info = pd.DataFrame({'name1': names_related_champions,
                                       'relation': relations,
                                       'name2': related_champions,
                                       })

    print(hero_info)
    print(champion_relations_info)
    return hero_info, champion_relations_info


if __name__ == '__main__':
    city_info, environment_info = get_city_info()
    hero_info, champion_relations_info = get_hero_info(city_info)
    city_info.to_csv('data_mine/city_info.csv', encoding="utf-8")
    environment_info.to_csv('data_mine/environment_info.csv', encoding="utf-8")
    hero_info.to_csv('data_mine/hero_info.csv', encoding="utf-8")
    champion_relations_info.to_csv('data_mine/champion_relations_info.csv', encoding="utf-8")