from main_utils         import load_json, dict2dotdict
from pandas             import DataFrame
import numpy            as np
import argparse
import pickle

def main(configs):
    attackType = configs.attackType
    attackPlatform = configs.attackPlatform
    wordlist = configs.node_file

    print(f"\nGenerating one-hot embeddings for {configs.attackType}_{configs.attackPlatform}")
    output_csv_features = f"{configs.FEATURE_PATH}/identity.csv"

    # read full word list
    f = open(configs.node_file, 'rb')
    fl = pickle.load(f)
    f.close()

    # generate identity matrix
    iden = np.identity(len(fl))
    df = DataFrame(data=iden, index=fl)
    df.to_csv(output_csv_features, header=False, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fasttext feature csv generator")
    parser.add_argument("--json_path", type=str, required=True,help="Path to the json config file")
    args = parser.parse_args()

    configs = load_json(args.json_path)
    main(dict2dotdict(configs))
