import os
from features import Mention, MentionPair
from nltk import ParentedTree
import re


pos_data = "data/postagged-files"
parse_data = "data/parsed-files"


def get_pairs(data):
    pos_dict = get_pos_dict(pos_data)
    entity_pairs = []
    parse_dict = get_parse_dict(parse_data)
    with open(data, 'r') as f:
        for line in f:
            fields = line.strip().split()
            fn = fields[1]
            sent_id = int(fields[2])
            assert int(fields[2]) == int(fields[8])

            span1 = [int(fields[3]), int(fields[4])]
            pos1 = "_".join(pos_dict[fn][sent_id][span1[0]: span1[1]])
            tree_pos1 = parse_dict[fn][sent_id].leaf_treeposition(span1[0])[:-1]
            mention1 = Mention(fields[7], fields[5], pos1, span1, tree_pos1)

            span2 = [int(fields[9]), int(fields[10])]
            pos2 = "_".join(pos_dict[fn][sent_id][span2[0]: span2[1]])
            tree_pos2 = parse_dict[fn][sent_id].leaf_treeposition(span2[0])[:-1]
            mention2 = Mention(fields[13], fields[11], pos2, span2, tree_pos2)

            tree = parse_dict[fn][sent_id]

            entity_pairs.append(MentionPair(mention1, mention2, fields[0], tree))

    return entity_pairs


def get_pos_dict(pos_data):
    pos_dict = {}
    fns = os.listdir(pos_data)
    for fn in fns:
        pos_dict[fn[:21]] = []
        with open(os.path.join(pos_data, fn)) as f:
            for line in f:
                line = line.strip()
                if line:
                    pos_list = [w_pos.split("_")[-1] for w_pos in line.split()]
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

# pos_dict = get_pos_dict(pos_data)
# print(pos_dict["APW20001001.2021.0521"][3])

# with open("data/rel-trainset.gold", 'r') as f:
#     for line in f:
#         fields = line.strip().split()
#         print(fields)

# pairs = get_pairs("data/rel-trainset.gold")
# print(pairs[0].mention2.pos)