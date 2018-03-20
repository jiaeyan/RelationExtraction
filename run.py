from data_prep import get_pairs


def generate_feature_file(pairs, out_f, train=True):
    with open(out_f, "w") as f:
        for pair in pairs:
            line = pair.rel+" " if train else ""
            line += " ".join(["{}={}".format(k, v) for k, v in pair.features.items()]) + "\n"
            f.write(line)


pairs = get_pairs("data/rel-trainset.gold")
generate_feature_file(pairs, "outout.txt")