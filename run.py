from data_prep import get_pairs
import argparse


def generate_feature_file(pairs, out_f, task="train"):
    with open(out_f, "w") as f:
        for pair in pairs:
            line = pair.rel+" " if task == "train" else ""
            line += " ".join(["{}={}".format(k, v) for k, v in pair.features.items()]) + "\n"
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description='Run relation extraction')
    parser.add_argument('--task', type=str, help='specify the task, use train/test/dev')
    args = parser.parse_args()

    print("Generating {} file...".format(args.task))

    if args.task == "train":
        pairs = get_pairs("data/rel-trainset.gold")
        generate_feature_file(pairs, "train.txt", task=args.task)

    elif args.task == "test":
        pairs = get_pairs("data/rel-testset.gold")
        generate_feature_file(pairs, "test.txt", task=args.task)

    elif args.task == "dev":
        pairs = get_pairs("data/rel-devset.gold")
        generate_feature_file(pairs, "dev.txt", task=args.task)


if __name__ == '__main__':
    main()
