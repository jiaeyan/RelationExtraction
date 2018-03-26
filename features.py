from nltk import ParentedTree


class Mention:

    def __init__(self, word, entity, pos, span, word_list, tree_pos):
        self.word = word.strip().replace("_", " ")
        self.entity = entity
        self.pos = pos
        self.span = span
        self.word_list = word_list
        self.tree_pos = tree_pos
        self.features = self.get_features()

    def get_features(self):
        features = {}

        self.get_head_pos(features)
        self.get_words_before(features)
        self.get_words_after(features)

        return features

    def get_head_pos(self, features):
        words = self.word.split(" ")
        poss = self.pos.split(" ")
        index = poss.index("IN") - 1 if "IN" in poss else -1
        features["head"] = words[index]
        features["head_pos"] = poss[index]

    def get_words_before(self, features):
        mention_start = self.span[0]
        first_w = self.word_list[mention_start - 1] if mention_start > 0 else "None"
        second_w = self.word_list[mention_start - 2] if mention_start > 1 else "None"
        features["first_word_before"] = first_w
        features["second_word_before"] = second_w

    def get_words_after(self, features):
        # end of span already points to the first word that is not in mention
        mention_end = self.span[1]
        sent_len = len(self.word_list)
        first_w = self.word_list[mention_end] if mention_end < sent_len else "None"
        second_w = self.word_list[mention_end + 1] if mention_end < sent_len - 1 else "None"
        features["first_word_after"] = first_w
        features["second_word_after"] = second_w


class MentionPair:

    def __init__(self, mention1, mention2, rel, tree, word_list, geo_dict):
        self.mention1 = mention1
        self.mention2 = mention2
        self.rel = rel
        self.tree = tree
        self.word_list = word_list
        # self.headed_tree = self.get_heads()
        self.features = self.get_features(geo_dict)

    def get_heads(self):

        return 0

    def get_features(self, geo_dict):
        features = {}

        # word bag of the mention
        features["WM1"] = self.mention1.word
        # head of the mention
        features["HM1"] = self.mention1.features["head"]
        # pos tag of head of the mention
        # features["HPM1"] = self.mention1.features["head_pos"]

        features["WM2"] = self.mention2.word
        features["HM2"] = self.mention2.features["head"]
        # features["HPM2"] = self.mention2.features["head_pos"]

        # combinations of head words and their pos tags of mention 1 and 2
        features["HM12"] = features["HM1"] + " " + features["HM2"]
        # features["HPM12"] = features["HPM1"] + " " + features["HPM2"]

        # first and second words before mention1
        features["BM1F"] = self.mention1.features["first_word_before"]
        features["BM1S"] = self.mention1.features["second_word_before"]

        # first and second words after mention2'
        features["AM1F"] = self.mention2.features["first_word_after"]
        features["AM1S"] = self.mention2.features["second_word_after"]

        # combination of entity types
        features["ET12"] = self.mention1.entity + " " + self.mention2.entity

        # words between mention1 and mention2, features["#WB"]
        self.get_words_between(features)

        # geo checking between mentions
        self.check_geo_info(features, geo_dict)

        # mention level relation
        self.get_mention_level(features, geo_dict)

        # mention inclusions
        # features["M1>M2"], features["M2>M1"] = self.check_mention_inclusion()


        # features["pos1"] = self.mention1.pos
        # features["pos2"] = self.mention2.pos
        # features["comb_pos"] = self.mention1.pos + " " + self.mention2.pos
        # issue in sents with parentheses where POS / word index are not equivalent to word index in tree
        # features["siblings"] = str(self.tree[self.mention1.tree_pos].parent() == self.tree[self.mention2.tree_pos].parent())
        # probably not useful only occurs positively 53 times in training mostly no_rel
        # features["modifies"] = str(features['siblings'] == "True" and self.head(self.mention1.pos) and not self.head(self.mention2.pos))
        # features["depth_diff"] = str(abs(len(self.mention1.tree_pos) - len(self.mention2.tree_pos)))

        # features["ancestor"] = self.tree[self.mention2.tree_pos] in self.tree[self.mention1.tree_pos].subtrees()
        # features["descendant"] =


        return features

    def get_words_between(self, features):
        start = self.mention1.span[1]
        end = self.mention2.span[0] - 1
        between_range = end - start + 1

        features["WBF"] = "None"
        features["WBL"] = "None"
        features["WBO"] = "None"
        features["WBFL"] = "None"
        features["WBNULL"] = False

        if between_range > 2:
            features["WBF"] = self.word_list[start]
            features["WBL"] = self.word_list[end]
            features["WBO"] = " ".join(self.word_list[start + 1: end])
        elif between_range == 2:
            features["WBF"] = self.word_list[start]
            features["WBL"] = self.word_list[end]
        elif between_range == 1:
            features["WBFL"] = self.word_list[start]
        else:
            features["WBNULL"] = True

    def check_mention_inclusion(self):
        span1 = self.mention1.span
        span2 = self.mention2.span
        m1hasm2 = True if span1[0] <= span2[0] and span1[1] >= span2[1] else False
        m2hasm1 = True if span2[0] <= span1[0] and span2[1] >= span1[1] else False
        return m1hasm2, m2hasm1

    def check_geo_info(self, features, geo_dict):
        w1 = self.clean_word(self.mention1.word, geo_dict)
        w2 = self.clean_word(self.mention2.word, geo_dict)

        features["GHAS"] = False
        features["GIN"] = False
        features["CountryET2"] = "None"
        features["ET1Country"] = "None"

        if w1 in geo_dict and w2 in geo_dict[w1]:
            features["GHAS"] = True
        elif w2 in geo_dict and w1 in geo_dict[w2]:
            features["GIN"] = True
        elif w1 in geo_dict or any([w1 in cities for cities in geo_dict.values()]):
            features["CountryET2"] = self.mention2.entity
        elif w2 in geo_dict or any([w2 in cities for cities in geo_dict.values()]):
            features["ET1Country"] = self.mention1.entity

    def get_mention_level(self, features, geo_dict):
        mt1 = self.check_mention_type(self.mention1, geo_dict)
        mt2 = self.check_mention_type(self.mention2, geo_dict)

        features["ML12"] = mt1 + " " + mt2

    def check_mention_type(self, mention, geo_dict):
        words = self.clean_word(mention.word, geo_dict).split()
        pos = mention.pos.split()

        if len(words) == 1 and pos[0].startswith("PRP"):
            return "PRONOUN"
        elif words[0].istitle() or (len(words) > 1 and words[1].istitle()):
            return "NAME"
        else:
            return "NOMIAL"

    def clean_word(self, word, geo_dict):
        return word.title() if word.isupper() and word not in geo_dict else word

    def head(self, pos):
        return pos.startswith("N") or pos.startswith("VB") or pos is "IN"