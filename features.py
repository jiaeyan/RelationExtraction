

class Mention:

    def __init__(self, word, entity, pos, span):
        self.word = word
        self.entity = entity
        self.pos = pos
        self.span = span
        self.features = self.get_features()

    def get_features(self):
        features = {}

        return features


class MentionPair:

    def __init__(self, mention1, mention2, rel):
        self.mention1 = mention1
        self.mention2 = mention2
        self.rel = rel
        self.features = self.get_features()

    def get_features(self):
        features = {}
        features["comb_entity"] = self.mention1.entity + "-" + self.mention2.entity
        features["pos1"] = self.mention1.pos
        features["pos2"] = self.mention2.pos
        features["comb_pos"] = self.mention1.pos + "-" + self.mention2.pos

        return features
