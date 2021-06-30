from gva_utils          import load_json, dict2dotdict
import argparse

def main(configs):
    for feature_file in configs.feature_files:
        features = f"{configs.FEATURE_PATH}/{feature_file}.csv"
        print(f"\nProcessing feature file {features}")
        output_features = f"./data/{configs.org}/{configs.dataset}_{feature_file}.features"

        nodes = {}
        # read nodes
        f = open(configs.node_file, 'r')
        for i, l in enumerate(f.readlines()):
            # repoID => nodeID mapping
            nodes[l.strip()] = str(i)
        f.close()

        # read lines
        f = open(features, 'r')
        g = open(output_features, 'w')
        # skip csv header
        f.readline()
        for l in f.readlines():
            # convert repoID to nodeID
            try:
                vec = l.strip().split(",")
                g.write("{} ".format(nodes[vec[0]]))
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
