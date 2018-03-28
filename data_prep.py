import os
import re
import json
from collections import defaultdict
from nltk import ParentedTree
from features import Mention, MentionPair

pos_data = "data/postagged-files"
parse_data = "data/parsed-files"
geo_data = "data/CountriesToCities.json"
last_name_data = "data/names/all_last_names.txt"
male_first_data = "data/names/male_first_names.txt"
female_first_data = "data/names/female_first_names.txt"
dep_data = 'data/dep-files'
chunked_data = 'data/chunked-files'


def get_pairs(data):
    pos_dict = get_pos_dict(pos_data)
    parse_dict = get_parse_dict(parse_data)
    dep_dict = get_dep_dict(dep_data)
    chunked_dict = get_chunked_dict(chunked_data)
    geo_dict = get_geo_dict(geo_data)
    names = get_name_dict([last_name_data, male_first_data, female_first_data])
    entity_pairs = []

    with open(data, 'r') as f:
        for line in f:
            fields = line.strip().split()
            fn = fields[1]
            sent_id = int(fields[2])
            tree = parse_dict[fn][sent_id]
            dep = dep_dict[fn][sent_id]
            chunks = chunked_dict[fn][sent_id]
            pos_list = [pair[-1] for pair in pos_dict[fn][sent_id]]
            word_list = [pair[0] for pair in pos_dict[fn][sent_id]]

            mention1 = wrap_mention(fields, word_list, pos_list, tree, span_id1=3, span_id2=4, word_id=7, ent_id=5)
            mention2 = wrap_mention(fields, word_list, pos_list, tree, span_id1=9, span_id2=10, word_id=13, ent_id=11)

            entity_pairs.append(MentionPair(mention1, mention2, fields[0], tree, dep, chunks, word_list, pos_list, geo_dict, names))

    return entity_pairs


def wrap_mention(fields, word_list, pos_list, tree, span_id1=0, span_id2=0, word_id=0, ent_id=0):
    span = [int(fields[span_id1]), int(fields[span_id2])]
    pos = " ".join(pos_list[span[0]: span[1]])
    tree_pos = tree.leaf_treeposition(span[0])[:-1]
    mention = Mention(fields[word_id], fields[ent_id], pos, span, word_list, pos_list, tree_pos)
    return mention


def get_pos_dict(pos_data):
    pos_dict = {}
    fns = os.listdir(pos_data)
    for fn in fns:
        pos_dict[fn[:21]] = []
        with open(os.path.join(pos_data, fn)) as f:
            for line in f:
                line = line.strip()
                if line:
                    # the pos_list contains word and its pos tag pair as one element
                    pos_list = [w_pos.split("_") for w_pos in line.split()]
                    pos_dict[fn[:21]].append(pos_list)

    return pos_dict


def get_parse_dict(parse_data):
    fns = os.listdir(parse_data)
    parse_dict = {}
    for fn in fns:
        parse_dict[fn[:21]] = []
        with open(os.path.join(parse_data, fn)) as f:
            for line in f:
                if not re.findall("^\s",line):
                    line = ParentedTree.fromstring(line)
                    parse_dict[fn[:21]].append(line)
    return parse_dict


def get_dep_dict(dep_data):
    fns = os.listdir(dep_data)
    dep_dict = {}
    for fn in fns:
        dep_dict[fn[:21]] = []
        with open(os.path.join(dep_data, fn)) as f:
            sent = []
            for line in f:
                if not line.split():
                    dep_dict[fn[:21]].append(sent)
                    sent = []
                else:
                    line = line.split()
                    if int(line[0]) == 1:
                        sent.append((0, 0, 'NONE'))
                    # (word#, dep_word#, relation)
                    sent.append((int(line[0]), int(line[6]), line[7]))
    return dep_dict


def get_chunked_dict(chunked_data):
    fns = os.listdir(chunked_data)
    chunked_dict = {}
    for fn in fns:
        chunked_dict[fn[:-4]] = []
        with open(os.path.join(chunked_data, fn)) as f:
            sent = []
            for line in f:
                if not line.startswith('#') and line.split():
                    line = line.split()
                    # word#, CHUNK, word, head_word, head#, chunk_chain
                    line = [line[2], line[3], line[5], line[7], line[8], line[9]]
                    sent.append(line)
                elif not line.split():

                    chunked_dict[fn[:-4]].append(sent)
                    sent = []
    return chunked_dict


def get_geo_dict(geo_data):
    geo_dict = defaultdict(set)
    mapping = json.load(open(geo_data), encoding="utf8")
    for country, cities in mapping.items():
        country = country.strip().replace("-", " ")
        for city in cities:
            geo_dict[country].add(city.strip().replace("-", " "))

        # make "United States" to "US", "U.S.", "the US" and "the U.S."
        if country.istitle():
            country_short = "".join([word[0] for word in country.split()])
            if len(country_short) > 1:
                country_short_dot = ".".join(list(country_short)) + "."
                det_country = "the " + country_short
                det_country_dot = "the " + country_short_dot

                geo_dict[country_short] = geo_dict[country]
                geo_dict[country_short_dot] = geo_dict[country]
                geo_dict[det_country] = geo_dict[country]
                geo_dict[det_country_dot] = geo_dict[country]

    return geo_dict


def get_name_dict(data_files):
    names = set()
    for data_file in data_files:
        read_name_file(data_file, names)
    return names


def read_name_file(data_file, names):
    with open(data_file, "r") as f:
        for line in f:
            name = line.strip().split()[0].title()
            names.add(name)


# pos_dict = get_pos_dict(pos_data)
# print(pos_dict["APW20001001.2021.0521"][3])

# ner = defaultdict(int)
# with open("data/rel-trainset.gold", 'r') as f:
#     for line in f:
#         fields = line.strip().split()
#         ner[fields[0]] += 1
#         # ner[fields[11]] += 1
# print(sorted(ner.items(), key=lambda x: x[1], reverse=True))

# pairs = get_pairs("data/rel-testset.gold")
# print(pairs[0].mention2.pos)

# for p in pairs:
#     if "ORG" in p.rel:
#         print(p.rel)
#         print(p.mention1.word)
#         print(p.mention2.word)
#         print()

def convert_to_dep(parse_data, dep_data):
    fns = os.listdir(parse_data)
    os.chdir('stanford-parser')
    for fn in fns:
        fn_out = fn[:21]
        os.system(
            "java -cp stanford-parser.jar edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile {0} -enhanced++ > {1}.conllu".format("../"+os.path.join(parse_data, fn), "../"+os.path.join(dep_data, fn_out)))
