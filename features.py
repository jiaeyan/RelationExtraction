from itertools import product
from nltk import ParentedTree
from nltk.corpus import wordnet as wn
import re

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

    def __init__(self, mention1, mention2, rel, tree, dep, chunks, word_list, pos_list, geo_dict, names):
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
        self.chunks = chunks
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

        # first and second pos before mention1 ---> DECREASE PERFORMANCE
        # features["PBM1F"] = self.mention1.features["first_pos_before"]
        # features["PBM1S"] = self.mention1.features["second_pos_before"]

        # first and second words after mention2'
        features["AM1F"] = self.mention2.features["first_word_after"]
        features["AM1S"] = self.mention2.features["second_word_after"]

        # first and second pos after mention2' ---> DECREASE PERFORMANCE
        # features["PAM1F"] = self.mention2.features["first_pos_after"]
        # features["PAM1S"] = self.mention2.features["second_pos_after"]

        # combination of entity types
        features["ET1"] = self.mention1.entity
        features["ET2"] = self.mention2.entity
        features["ET12"] = self.mention1.entity + " " + self.mention2.entity

        # combinations of dep relations and entity types ---> DECREASE PERFORMANCE
        #  decrease performance
        # features["DR1"] = self.dep[self.mention1.span[0]][2]
        # features["DR2"] = self.dep[self.mention2.span[0]][2]

        #  dep relation 1 & dep relation 2
        features['DR1DR2'] = self.dep[self.mention1.span[0]][2] + " " + self.dep[self.mention2.span[0]][2]

        # DECREASE PERFORMANCE
        # features["DR1ET1"] = self.dep[self.mention1.span[0]][2] + " " + self.mention1.entity
        # features["DR2ET2"] = self.dep[self.mention2.span[0]][2] + " " + self.mention2.entity

        # combination of entity type and dependent word ---> DECREASE PERFORMANCE
        # features["ET1DW1"] = self.mention1.entity + " " + self.tree.leaves()[self.dep[self.mention1.span[0]][1]]
        # features["ET2DW2"] = self.mention2.entity + " " + self.tree.leaves()[self.dep[self.mention2.span[0]][1]]

        # combination of head word and dependent word ---> DECREASE PERFORMANCE
        # features["H1DW1"] = self.mention1.features["head"] + " " + self.tree.leaves()[self.dep[self.mention1.span[0]][1]]
        # features["H2DW2"] = self.mention2.features["head"] + " " + self.tree.leaves()[self.dep[self.mention2.span[0]][1]]

        # words between mention1 and mention2, features["#WB"]
        self.get_words_pos_between(features)

        # geo checking between mentions
        self.check_geo_info(features, geo_dict)

        # check family relation ---> DECREASE PERFORMANCE
        # self.check_family(features, geo_dict, names)

        # get wordnet information ---> DECREASE PERFORMANCE
        # self.get_wordnet_info(features)

        # check if words are in the same phrases
        self.check_shared_phrase(features)

        # check invent relation --> DECREASE PERFORMANCE
        # self.check_create(features)

        # check onwereship relation --> DECREASE PERFORMANCE
        # self.check_own(features)

        # check name info
        self.get_name_info(features, geo_dict, names)

        # check org info
        # self.get_organization_info(features)

        # check employment info
        self.get_employ_info(features)

        # check social info
        self.get_social_info(features)

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

        # self.get_chunk_features(features)


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

        features["PBF"] = "None"
        features["PBL"] = "None"
        features["PBO"] = "None"
        features["PBFL"] = "None"
        features["PBNULL"] = False

        if between_range > 2:
            features["WBF"] = self.word_list[start]
            features["WBL"] = self.word_list[end]
            features["WBO"] = " ".join(self.word_list[start + 1: end])

            features["PBF"] = self.pos_list[start]
            features["PBL"] = self.pos_list[end]
            features["PBO"] = " ".join(self.pos_list[start + 1: end])

        elif between_range == 2:
            features["WBF"] = self.word_list[start]
            features["WBL"] = self.word_list[end]

            features["PBF"] = self.pos_list[start]
            features["PBL"] = self.pos_list[end]

        elif between_range == 1:
            features["WBFL"] = self.word_list[start]
            features["PBFL"] = self.pos_list[start]

        else:
            features["WBNULL"] = True
            features["PBNULL"] = True

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

    def get_wordnet_info(self, features):
        hm1 = self.wn_get_stem(features["HM1"].lower())
        hm2 = self.wn_get_stem(features["HM2"].lower())

        hyper1 = self.get_hypernyms(hm1)
        hyper2 = self.get_hypernyms(hm2)
        # syn1 = self.get_synonyms(hm1)
        # syn2 = self.get_synonyms(hm2)

        features["HAS"] = False
        features["IS"] = False
        # features["SAME"] = False
        # features["SIMILARITY"] = self.get_similarity(hm1, hm2)

        # if hm1 in syn2 or hm2 in syn1:
        #     features["SAME"] = True
        if hm1 in hyper2:
            features["HAS"] = True
        elif hm2 in hyper1:
            features["IS"] = True

    def wn_get_stem(self, word):
        for s in wn.synsets(word):
            for lemma in s.lemmas():
                return lemma.name()
        return word

    def get_similarity(self, word1, word2):
        synset1 = wn.synsets(word1)
        synset2 = wn.synsets(word2)

        if len(synset1) == 0 or len(synset2) == 0:
            return 0

        best = max(wn.path_similarity(s1, s2) for s1, s2 in
                   product(synset1, synset2))
        return best

    def get_hypernyms(self, word):
        hypers = set()
        for ss in wn.synsets(word):
            for hyper in ss.hypernyms():
                for lemma in hyper.lemmas():
                    hypers.add(lemma.name())
        return hypers

    def get_synonyms(self, word):
        syns = set()
        for ss in wn.synsets(word):
            for syn in ss.hypernyms():
                for lemma in syn.lemmas():
                    syns.add(lemma.name())
        return syns

    def get_name_info(self, features, geo_dict, names):
        w1 = self.mention1.word
        w2 = self.mention2.word

        features["NameET2"] = "None"
        features["ET1Name"] = "None"

        if w1.istitle():
            for word in w1.split():
                if word in names and word not in geo_dict:
                    features["NameET2"] = self.mention2.entity
                    break
        elif w2.istitle():
            for word in w2.split():
                if word in names and word not in geo_dict:
                    features["ET1Name"] = self.mention1.entity
                    break

    def get_organization_info(self, features):
        w1 = self.omit_stopwords(self.mention1.word)
        w2 = self.omit_stopwords(self.mention2.word)

        features["ORGET2"] = "None"
        features["ET1ORG"] = "None"
        # features["ORGORG"] = "None"

        triggers = {"University", "College", "Association", "Department", "Division", "Bank", "Hospital",
                    "Union", "Centre", "Center", "Central", "Organization", "Authority", "Agency", "Organisation",
                    "Institute", "Court", "Assembly", "Commission", "Committee", "Board", "International", "National",
                    "Inc", "Laboratory", "Lab", "Office", "Company", "Congress", "Microsoft", "Facebook", "Apple", "Uber",
                    "Linkedin", "Yahoo", "Amazon", "Google", "Ltd"}

        for w in w1:
            for t in triggers:
                if t in w or t.lower() in w:
                    features["ORGET2"] = self.mention2.entity
                    break
        for w in w2:
            for t in triggers:
                if t in w or t.lower() in w:
                    features["ET1ORG"] = self.mention1.entity
                    break

        # if features["ORGET2"] != "None" and features["ET1ORG"] != "None":
        #     features["ORGORG"] = features["ET12"]

    def get_employ_info(self, features):
        w1 = self.omit_stopwords(self.mention1.word)
        w2 = self.omit_stopwords(self.mention2.word)

        features["EMPET2"] = "None"
        features["ET1EMP"] = "None"

        triggers = {"lead", "official", "officer", "administrator", "manager", "director", "executive", "president",
                    "chief", "boss", "chair", "supervisor", "governor", "head", "doctor", "professor", "student",
                    "analyst", "journalist", "scientist", "police", "teacher", "assistant", "accountant", "actor",
                    "agent", "technician", "controller", "specialist", "expert", "driver", "trainer", "instructor",
                    "operator", "counsellor", "consultant", "adviser", "engineer", "researcher", "mayor",
                    "CEO", "CTO", "lawyer", "representative"}

        for w in w1:
            for t in triggers:
                if t in w or t.title() in w:
                    features["EMPET2"] = self.mention2.entity
                    break
        for w in w2:
            for t in triggers:
                if t in w or t.title() in w:
                    features["ET1EMP"] = self.mention1.entity
                    break

    def get_social_info(self, features):
        features["FAM"] = False
        features["BUS"] = False

        if features["ET12"] == "PER PER":
            w1 = self.omit_stopwords(self.mention1.word)
            w2 = self.omit_stopwords(self.mention2.word)
            w3 = set(w1 + w2)

            fam = {"wife", "husband", "son", "daughter", "father", "mother", "grandfather", "grandmother", "aunt", "uncle",
                   "brother", "sister", "niece", "nephew", "cousin", "parent", "relative", "child", "kid"}
            bus = {"boss", "employer", "employee", "colleague", "teacher", "student", "lawyer", "client",
                   "spokesm", "lead", "administrator", "manager", "director", "executive", "president",
                   "chief", "chair", "supervisor", "governor", "head", "police", "doctor", "professor", "student",
                   "assistant", "accountant", "agent", "controller", "sponsor", "driver", "trainer", "instructor",
                   "counsellor", "consultant", "adviser", "mentor", "team", "candidate", "secretary",
                   "surrogate", "representative", "colleague"}

            for w in w3:
                for f in fam:
                    if f in w:
                        features["FAM"] = True
                        break
                for b in bus:
                    if b in w:
                        features["BUS"] = True
                        break
                if features["FAM"] is True and features["BUS"] is True:
                    break

    def omit_stopwords(self, word):
        words = word.split()
        stopwords = {"of", "and", "the", "on", "for"}
        neo_word = [w for w in words if w not in stopwords]
        return neo_word

    def clean_word(self, word, geo_dict):
        return word.title() if word.isupper() and word not in geo_dict else word

    def head(self, pos):
        return pos.startswith("N") or pos.startswith("VB") or pos is "IN"

    def check_shared_phrase(self, features):
        sameNP = 'False'
        min_tree = self.tree[self.tree.treeposition_spanning_leaves(self.mention1.span[0], self.mention2.span[0])]
        while min_tree is not None and not isinstance(min_tree, str):
            if min_tree.label() == "NP" and sameNP == 'False':
                sameNP = "True " + self.mention1.entity + " " + self.mention2.entity
                break
            min_tree = min_tree.parent()

        # Only having NPs helped performance
        # features['sameVP'] = sameVP
        features['sameNP'] = sameNP
        # features['samePP'] = samePP

    def get_chunk_features(self, features):
        chunk_words = [word[2].replace('COMMA',',') for word in self.chunks]
        chunk_words = [word.split(' ')[-1] if len(word.split(' ')) > 1 else word for word in chunk_words]
        if 'O\'' in chunk_words and 'O\'' not in self.word_list:
            chunk_words[chunk_words.index('O\'')+1] = 'O\'' + chunk_words[chunk_words.index('O\'')+1]
            chunk_words.pop(chunk_words.index('O\''))
        if '\'s' in chunk_words and '\'s' not in self.word_list:
            chunk_words[chunk_words.index('\'s')-1] = chunk_words[chunk_words.index('\'s')-1] + '\'s'
            chunk_words.pop(chunk_words.index('\'s'))

        word1 = re.sub('`|\(|\)', '', self.mention1.word)
        word1 = word1.split(' ')[-1].split('\'')[-1] if 'O\'' not in word1 and '\'s' not in word1 else word1.split(' ')[
            -1]
        word2 = re.sub('`|\(|\)', '', self.mention2.word)
        word2 = word2.split(' ')[-1].split('\'')[-1] if 'O\'' not in word2 and '\'s' not in word2 else word2.split(' ')[
            -1]

        #features['CPHBNULL'] = False
        #features['CPHBFL'] = False
        #features['CPHBF'] = False
        #features['CPHBL'] = False
        #features['CPHBO'] = False
        try:
            id_1, id_2 = self.chunks[chunk_words.index(word1)][-2], self.chunks[chunk_words.index(word2)][-2]
            chunk_span = self.chunks[chunk_words.index(word1):chunk_words.index(word2) + 1]
            # chunk_span = self.chunks[self.mention1.span[0]:self.mention2.span[0] + 1]
            # chunk phrase heads in between ---> DECREASES PERFORMANCE
            '''chunks_in_between = [word for word in chunk_span[1:-1] if word[-2] != id_1 and word[-2] != id_2]
            if len(chunks_in_between) == 0:
                features['CPHBNULL'] = True
            elif len(chunks_in_between) == 1:
                features['CPHBFL'] = chunks_in_between[0][3]
            elif len(chunks_in_between) > 1:
                features['CPHBF'] = chunks_in_between[0][3]
                features['CPHBL'] = chunks_in_between[-1][3]
                if len(chunks_in_between) > 2:
                    features['CPHBO'] = ' '.join([chunk[3] for chunk in chunks_in_between[1:-1]])'''
            # phrase chain --> DECREASES PERFORMANCE
            """
            chain = []
            for word in chunks_in_between:
                if word[1].startswith('B'):
                chain.append(word[1].split('-')[-1])
            features['phrase_chain'] = ' '.join(chain) if chain else 'None'"""
            '''chunk previous --> DECREASES PERFORMANCE
            features['pre_chunk1'] = "None"
            features['pre_chunk2'] = "None"
            pre_chunk = self.chunks[:chunk_words.index(word1)]
            if len(pre_chunk) > 0:
                features['pre_chunk1'] = pre_chunk[0][2]
            if len(pre_chunk) > 1:
                features['pre_chunk2'] = pre_chunk[1][2]'''

            # chunk after

        except ValueError:
            print(word1,word2, " --- mismatch in tokenization")

