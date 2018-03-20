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
    parser.set_defaults(train=True)
    parser.add_argument('--train', dest='train', action='store_true', help='decide if this procedure is train')
    parser.add_argument('--test', dest='train', action='store_false', help='decide if this procedure is test')
    args = parser.parse_args()

    if args.train:
        pairs = get_pairs("data/rel-trainset.gold")
    else:
        pairs = get_pairs("data/rel-testset.gold")
    if args.train:
        print("yes trianing")
        generate_feature_file(pairs, "train.txt", train=args.train)
    else:
        print("yes testing")
        generate_feature_file(pairs, "test.txt", train=args.train)


if __name__ == '__main__':
    main()