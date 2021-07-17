from main_utils          import load_json, dict2dotdict
import argparse
import pickle
import os

def main(configs):
    feature_files = [filename.split(".")[0] for filename in os.listdir(configs.FEATURE_PATH)]
    for feature_file in feature_files:
        features = f"{configs.FEATURE_PATH}{feature_file}.csv"
        print(f"\nProcessing feature file {features}")
        output_features = f"./data/{configs.dataset}/{configs.attackType}_{configs.attackPlatform}_{feature_file}.features"

        words = {}
        # read nodes
        f = open(configs.node_file, 'rb')
        fl = pickle.load(f)
        for i, l in enumerate(fl):
            # repoID => nodeID mapping
            words[l.strip()] = str(i)
        f.close()

        # read lines
        f = open(features, 'r', encoding='utf8')
        g = open(output_features, 'w')
        for l in f.readlines():
            # convert word to wordID
            try:
                vec = l.strip().split(",")
                g.write("{} ".format(words[vec[0]]))
                g.write(" ".join(vec[1:]))
                g.write("\n")
            except:
                vec = l.strip().split(",")
                print(f"node {vec[0]} not in the graph")
        g.close()
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser feature file generator")
    parser.add_argument("--json_path", type=str, required=True,help="Path to the json config file")
    args = parser.parse_args()

    configs = load_json(args.json_path)
    main(dict2dotdict(configs))
