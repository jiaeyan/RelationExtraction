from nltk import ParentedTree
import re


class Mention:

    def __init__(self, word, entity, pos, span, word_list, tree_pos):
        self.word = word.replace("_", " ")
        self.entity = entity
        self.pos = pos
        self.span = span
        self.word_list = word_list
        self.tree_pos = tree_pos
        self.features = self.get_features()

    def get_features(self):
        features = {}

        features["head"], features["head_pos"] = self.get_head_pos()
        features["first_word_before"], features["second_word_before"] = self.get_words_before()
        features["first_word_after"], features["second_word_after"] = self.get_words_after()

        return features

    def clean_word(self, word):
        # handle case like "Mercer_Bullard"
        words = word.replace("_", " ")

        # convert case like "CAIRO HAHA" to "Cairo Haha", but make "UK" to "Uk"
        words = self.titlecase(words) if words.isupper() else words
        return words

    def titlecase(self, word):
        return re.sub(r"[A-Za-z]+('[A-Za-z]+)?",
                        lambda mo: mo.group(0)[0].upper() +
                                   mo.group(0)[1:].lower(),
                        word)

    def get_head_pos(self):
        words = self.word.split(" ")
        poss = self.pos.split(" ")
        if "IN" in poss:
            prep_index = poss.index("IN")
            return words[prep_index-1], poss[prep_index-1]
        return words[-1], poss[-1]

    def get_words_before(self):
        mention_start = self.span[0]
        first_w = self.word_list[mention_start - 1] if mention_start > 0 else "None"
        second_w = self.word_list[mention_start - 2] if mention_start > 1 else "None"
        return first_w, second_w

    def get_words_after(self):
        # end of span already points to the first word that is not in mention
        mention_end = self.span[1]
        sent_len = len(self.word_list)
        first_w = self.word_list[mention_end] if mention_end < sent_len else "None"
        second_w = self.word_list[mention_end + 1] if mention_end < sent_len - 1 else "None"
        return first_w, second_w




class MentionPair:

    def __init__(self, mention1, mention2, rel, tree, word_list):
        self.mention1 = mention1
        self.mention2 = mention2
        self.rel = rel
        self.tree = tree
        self.word_list = word_list
        # self.headed_tree = self.get_heads()
        self.features = self.get_features()

    def get_heads(self):

        return 0

    def get_features(self):
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
        features["HPM12"] = self.mention1.features["head_pos"] + " " + self.mention2.features["head_pos"]

        # features["HPM12"] = features["HPM1"] + " " + features["HPM2"]

        # first and second words before mention1
        features["BM1F"] = self.mention1.features["first_word_before"]
        features["BM1S"] = self.mention1.features["second_word_before"]

        # first and second words after mention2'
        features["AM1F"] = self.mention2.features["first_word_after"]
        features["AM1S"] = self.mention2.features["second_word_after"]

        # words between mention1 and mention2
        features["WBF"], features["WBL"], features["WBO"], features["WBFL"], features["WBNULL"], features["#WB"] = \
            self.get_words_between()

        # combination of entity types
        features["ET12"] = self.mention1.entity + " " + self.mention2.entity


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

    def get_words_between(self):
        start = self.mention1.span[1]
        end = self.mention2.span[0] - 1
        between_range = end - start + 1

        first_w = "None"
        last_w = "None"
        other_w = "None"
        only_w = "None"
        no_w = False

        if between_range > 2:
            first_w = self.word_list[start]
            last_w = self.word_list[end]
            other_w = " ".join(self.word_list[start + 1: end])
        elif between_range == 2:
            first_w = self.word_list[start]
            last_w = self.word_list[end]
        elif between_range == 1:
            only_w = self.word_list[start]
        else:
            no_w = True

        return first_w, last_w, other_w, only_w, no_w, between_range

    def check_mention_inclusion(self):
        span1 = self.mention1.span
        span2 = self.mention2.span
        m1hasm2 = True if span1[0] <= span2[0] and span1[1] >= span2[1] else False
        m2hasm1 = True if span1[0] <= span2[0] and span1[1] >= span2[1] else False

    def head(self, pos):
        return pos.startswith("N") or pos.startswith("VB") or pos is "IN"