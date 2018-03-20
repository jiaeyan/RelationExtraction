from data_prep import get_pairs
import argparse



def generate_feature_file(pairs, out_f, train=True):
    with open(out_f, "w") as f:
        for pair in pairs:
            line = pair.rel+" " if train else ""
            line += " ".join(["{}={}".format(k, v) for k, v in pair.features.items()]) + "\n"
            f.write(line)

def main():
    parser = argparse.ArgumentParser(description='Run relation extraction')
    parser.add_argument('--gold', type=str, help='the input gold file to generate train feature file')
    parser.add_argument('--train', type=bool, help='decide if this procedure is train or not')
    args = parser.parse_args()

    if args.train:
        pairs = get_pairs("data/rel-trainset.gold")
    else:
        pairs = get_pairs("data/rel-testset.gold")
    if args.train:
        generate_feature_file(pairs, "train.txt", train=args.train)
    else:
        generate_feature_file(pairs, "test.txt", train=args.train)


if __name__ == '__main__':
    main()