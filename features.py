from nltk import ParentedTree

class Mention:

    def __init__(self, word, entity, pos, span, tree_pos):
        self.word = word
        self.entity = entity
        self.pos = pos
        self.span = span
        self.tree_pos = tree_pos
        self.features = self.get_features()

    def get_features(self):
        features = {}

        return features


class MentionPair:

    def __init__(self, mention1, mention2, rel, tree):
        self.mention1 = mention1
        self.mention2 = mention2
        self.rel = rel
        self.tree = tree
        # self.headed_tree = self.get_heads()
        self.features = self.get_features()

    def get_heads(self):

        return 0

    def get_features(self):
        features = {}
        features["comb_entity"] = self.mention1.entity + "-" + self.mention2.entity
        features["pos1"] = self.mention1.pos
        features["pos2"] = self.mention2.pos
        features["comb_pos"] = self.mention1.pos + "-" + self.mention2.pos
        # issue in sents with parentheses where POS / word index are not equivalent to word index in tree
        features["siblings"] = str(self.tree[self.mention1.tree_pos].parent() == self.tree[self.mention2.tree_pos].parent())
        # probably not useful only occurs positively 53 times in training mostly no_rel
        # features["modifies"] = str(features['siblings'] == "True" and self.head(self.mention1.pos) and not self.head(self.mention2.pos))
        features["depth_diff"] = str(abs(len(self.mention1.tree_pos) - len(self.mention2.tree_pos)))

        # features["ancestor"] = self.tree[self.mention2.tree_pos] in self.tree[self.mention1.tree_pos].subtrees()
        # features["descendant"] =


        return features

    def head(self, pos):
        return pos.startswith("N") or pos.startswith("VB") or pos is "IN"