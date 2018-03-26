from nltk import ParentedTree


class Mention:

    def __init__(self, word, entity, pos, span, word_list, pos_list, tree_pos):
        self.word = word.strip().replace("_", " ")
        self.entity = entity
        self.pos = pos
        self.span = span
        self.word_list = word_list
        self.pos_list = pos_list
        self.tree_pos = tree_pos
        self.features = self.get_features()

    def get_features(self):
        features = {}

        self.get_head_pos(features)
        self.get_words_pos_before(features)
        self.get_words_pos_after(features)

        return features

    def get_head_pos(self, features):
        words = self.word.split(" ")
        poss = self.pos.split(" ")
        index = poss.index("IN") - 1 if "IN" in poss else -1
        features["head"] = words[index]
        features["head_pos"] = poss[index]

    def get_words_pos_before(self, features):
        mention_start = self.span[0]

        first_w = self.word_list[mention_start - 1] if mention_start > 0 else "None"
        second_w = self.word_list[mention_start - 2] if mention_start > 1 else "None"
        features["first_word_before"] = first_w
        features["second_word_before"] = second_w

        first_p = self.pos_list[mention_start - 1] if mention_start > 0 else "None"
        second_p = self.pos_list[mention_start - 2] if mention_start > 1 else "None"
        features["first_pos_before"] = first_p
        features["second_pos_before"] = second_p

    def get_words_pos_after(self, features):
        # end of span already points to the first word that is not in mention
        mention_end = self.span[1]
        sent_len = len(self.word_list)

        first_w = self.word_list[mention_end] if mention_end < sent_len else "None"
        second_w = self.word_list[mention_end + 1] if mention_end < sent_len - 1 else "None"
        features["first_word_after"] = first_w
        features["second_word_after"] = second_w

        first_p = self.pos_list[mention_end] if mention_end < sent_len else "None"
        second_p = self.pos_list[mention_end + 1] if mention_end < sent_len - 1 else "None"
        features["first_pos_after"] = first_p
        features["second_pos_after"] = second_p


class MentionPair:

    def __init__(self, mention1, mention2, rel, tree, dep, word_list, pos_list, geo_dict, names):
        self.mention1 = mention1
        self.mention2 = mention2
        self.rel = rel
        self.tree = tree
        self.dep = dep
        self.word_list = word_list
        self.pos_list = pos_list
        self.mid_words = word_list[mention1.span[1]: mention2.span[0]]
        self.mid_poss = pos_list[mention1.span[1]: mention2.span[0]]
        # self.headed_tree = self.get_heads()
        self.features = self.get_features(geo_dict, names)


    def get_features(self, geo_dict, names):
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

        # first and second pos before mention1
        # features["PBM1F"] = self.mention1.features["first_pos_before"]
        # features["PBM1S"] = self.mention1.features["second_pos_before"]

        # first and second words after mention2'
        features["AM1F"] = self.mention2.features["first_word_after"]
        features["AM1S"] = self.mention2.features["second_word_after"]

        # first and second pos after mention2'
        # features["PAM1F"] = self.mention2.features["first_pos_after"]
        # features["PAM1S"] = self.mention2.features["second_pos_after"]

        # combination of entity types
        features["ET1"] = self.mention1.entity
        features["ET2"] = self.mention2.entity
        features["ET12"] = self.mention1.entity + " " + self.mention2.entity

        # combination of entity type and dependent word ---> DECREASE PERFORMANCE
        # features["ET1DW1"] = self.mention1.entity + " " + self.tree.leaves()[self.dep[self.mention1.span[0]][1]]
        # features["ET2DW2"] = self.mention2.entity + " " + self.tree.leaves()[self.dep[self.mention2.span[0]][1]]

        # combination of head word and dependent word ---> DECREASE PERFORMANCE
        # features["H1DW1"] = self.mention1.features["head"] + " " + self.tree.leaves()[self.dep[self.mention1.span[0]][1]]
        # features["H2DW2"] = self.mention2.features["head"] + " " + self.tree.leaves()[self.dep[self.mention2.span[0]][1]]

        # combination of ET12 and if they are in the same (type of) phrase
        # features['ET12SameNP'] = 0
        # features['ET12SamePP'] = 0
        # features['ET12SameVP'] = 0

        # words between mention1 and mention2, features["#WB"]
        self.get_words_pos_between(features)

        # geo checking between mentions
        self.check_geo_info(features, geo_dict)

        # check family relation
        # self.check_family(features, geo_dict, names)

        # check invent relation
        # self.check_create(features)

        # check onwereship relation --> DECREASE PERFORMANCE
        # self.check_own(features)

        # mention level relation --> DECREASE PERFORMANCE
        # self.get_mention_level(features, geo_dict)

        # mention inclusions --> DECREASE PERFORMANCE
        # features["M1>M2"], features["M2>M1"] = self.check_mention_inclusion()

        # POS information  --> DECREASE PERFORMANCE
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


    def get_words_pos_between(self, features):
        start = self.mention1.span[1]
        end = self.mention2.span[0] - 1
        between_range = end - start + 1

        features["WBF"] = "None"
        features["WBL"] = "None"
        features["WBO"] = "None"
        features["WBFL"] = "None"
        features["WBNULL"] = False

        # features["PBF"] = "None"
        # features["PBL"] = "None"
        # features["PBO"] = "None"
        # features["PBFL"] = "None"
        # features["PBNULL"] = False

        if between_range > 2:
            features["WBF"] = self.word_list[start]
            features["WBL"] = self.word_list[end]
            features["WBO"] = " ".join(self.word_list[start + 1: end])

            # features["PBF"] = self.pos_list[start]
            # features["PBL"] = self.pos_list[end]
            # features["PBO"] = " ".join(self.pos_list[start + 1: end])

        elif between_range == 2:
            features["WBF"] = self.word_list[start]
            features["WBL"] = self.word_list[end]

            # features["PBF"] = self.pos_list[start]
            # features["PBL"] = self.pos_list[end]

        elif between_range == 1:
            features["WBFL"] = self.word_list[start]
            # features["PBFL"] = self.pos_list[start]

        else:
            features["WBNULL"] = True
            # features["PBNULL"] = True

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

        ents = features["ET12"].split()

        if ents == "GPE GPE":
            # if one mention is located at another
            if w1 in geo_dict and w2 in geo_dict[w1]:
                features["GHAS"] = True
            elif w2 in geo_dict and w1 in geo_dict[w2]:
                features["GIN"] = True
        elif ents[0] == "GPE":
            # country and residence
            if w1 in geo_dict or any([w1 in cities for cities in geo_dict.values()]):
                features["CountryET2"] = ents[1]
        elif ents[1] == "GPE":
            # country and residence reverse
            if w2 in geo_dict or any([w2 in cities for cities in geo_dict.values()]):
                features["ET1Country"] = ents[0]

    def get_mention_level(self, features, geo_dict):
        mt1 = self.check_mention_type(self.mention1, geo_dict)
        mt2 = self.check_mention_type(self.mention2, geo_dict)

        features["ML12"] = mt1 + " " + mt2

    def check_mention_type(self, mention, geo_dict):
        # words = self.clean_word(mention.word, geo_dict).split()
        pos = mention.pos.split()

        if "PRP" in pos:
            return "PRONOUN"
        elif "NNP" in pos or "NNPS" in pos:
            return "NAME"
        else:
            return "NOMIAL"

    def check_family(self, features, geo_dict, names):
        w1 = set(self.clean_word(self.mention1.word, geo_dict).split())
        w2 = set(self.clean_word(self.mention2.word, geo_dict).split())

        features["RELATIVE"] = False
        features["FAMNAME"] = False

        # some relative trigger nouns between mentions
        triggers = {"wife", "husband", "daughter", "son", "father", "mother", "grandfather", "grandmother", "uncle",
                    "aunt", "nephew", "niece", "sister", "brother", "cousin"}
        for word in self.mid_words:
            if word.lower() in triggers:
                features["RELATIVE"] = True
                break

        # if two mentions share the same family name
        w = w1 & w2
        for word in w:
            if word in names:
                features["FAMNAME"] = True
                break

    def check_create(self, features):
        features["CREATE"] = False
        # features["CREATENUM"] = 0
        features["CREATOR"] = "None"
        features["CREATEE"] = "None"
        features["CREATEBY"] = "None"

        triggers = {"creat", "buil", "made", "make", "mak", "develop", "construct", "draw", "drew",
                    "coin", "writ", "wrote", "invent", "manufactur", "generat", "produc", "foster", "fabricat"}

        # some relative trigger verbs between mentions
        for i, word in enumerate(self.mid_words):
            # if self.mid_poss[i].startswith("VB"):
            for trigger in triggers:
                if trigger in word.lower():
                    features["CREATE"] = True
                    # features["CREATENUM"] += 1
                    break

        if features["CREATE"]:
            ents = features["ET12"].split()
            features["CREATOR"] = ents[0]
            features["CREATEE"] = ents[1]
            features["CREATEBY"] = features["ET12"]

    def check_own(self, features):
        features["OWN"] = False
        features["OWNER"] = "None"
        features["OWNEE"] = "None"
        features["OWNBY"] = "None"
        features["BYOWN"] = "None"

        triggers = {"found", "run", "ran", "start", "own", "use", "consume", "buil", "manage", "support", "control",
                    "acquir", "buy", "bought", "possess", "lead", "led", "govern", "oversee", "supervise", "administer",
                    "utiliz", "direct"}

        # some relative trigger verbs between mentions
        for i, word in enumerate(self.mid_words):
            if self.mid_poss[i].startswith("VB"):
                for trigger in triggers:
                    if trigger in word.lower():
                        features["OWN"] = True
                        break

        if features["OWN"]:
            ents = features["ET12"].split()
            if "by" in self.mid_words:
                features["OWNER"] = ents[1]
                features["OWNEE"] = ents[0]
                features["OWNBY"] = features["ET12"]
            else:
                features["OWNER"] = ents[0]
                features["OWNEE"] = ents[1]
                features["BYOWN"] = features["ET12"]

    def clean_word(self, word, geo_dict):
        return word.title() if word.isupper() and word not in geo_dict else word

    def head(self, pos):
        return pos.startswith("N") or pos.startswith("VB") or pos is "IN"
