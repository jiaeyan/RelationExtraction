import os
from features import Mention, MentionPair
from nltk import ParentedTree
import re


pos_data = "data/postagged-files"
parse_data = "data/parsed-files"


def get_pairs(data):
    pos_dict = get_pos_dict(pos_data)
    parse_dict = get_parse_dict(parse_data)
    entity_pairs = []
    with open(data, 'r') as f:
        for line in f:
            fields = line.strip().split()
            fn = fields[1]
            sent_id = int(fields[2])
            tree = parse_dict[fn][sent_id]
            pos_list = [pair[-1] for pair in pos_dict[fn][sent_id]]
            word_list = [pair[0] for pair in pos_dict[fn][sent_id]]

            mention1 = wrap_mention(fields, word_list, pos_list, tree, span_id1=3, span_id2=4, word_id=7, ent_id=5)
            mention2 = wrap_mention(fields, word_list, pos_list, tree, span_id1=9, span_id2=10, word_id=13, ent_id=11)

            entity_pairs.append(MentionPair(mention1, mention2, fields[0], tree, word_list))

    return entity_pairs


def wrap_mention(fields, word_list, pos_list, tree, span_id1=0, span_id2=0, word_id=0, ent_id=0):
    span = [int(fields[span_id1]), int(fields[span_id2])]
    pos = " ".join(pos_list[span[0]: span[1]])
    tree_pos = tree.leaf_treeposition(span[0])[:-1]
    mention = Mention(fields[word_id], fields[ent_id], pos, span, word_list, tree_pos)
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

# pos_dict = get_pos_dict(pos_data)
# print(pos_dict["APW20001001.2021.0521"][3])

# with open("data/rel-trainset.gold", 'r') as f:
#     for line in f:
#         fields = line.strip().split()
#         print(fields)

# pairs = get_pairs("data/rel-trainset.gold")
# print(pairs[0].mention2.pos)