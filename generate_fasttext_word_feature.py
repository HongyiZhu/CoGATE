from main_utils         import load_json, dict2dotdict
from pandas             import DataFrame
import argparse
import pickle
import fasttext
import fasttext.util

def main(configs, dim):
    attackType = configs.attackType
    attackPlatform = configs.attackPlatform
    wordlist = configs.node_file

    ft = fasttext.load_model('./data/cc.en.300.bin')

    if dim != 300:
        fasttext.util.reduce_model(ft, dim)
    
    print(f"\nGenerating fasttext embeddings for {configs.attackType}_{configs.attackPlatform} with {dim} dimensions")
    output_csv_features = f"{configs.FEATURE_PATH}/fasttext_{dim}.csv"

    # read full word list
    f = open(configs.node_file, 'rb')
    fl = pickle.load(f)
    f.close()

    # compile fasttext embedding dictionary
    fl_dict = {}
    for word in fl:
        fl_dict[word] = ft.get_word_vector(word)
    
    # use pandas DF to create output csv
    df = DataFrame.from_dict(fl_dict, orient='index')
    df.to_csv(output_csv_features, header=False, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fasttext feature csv generator")
    parser.add_argument("--json_path", type=str, required=True,help="Path to the json config file")
    parser.add_argument("--fastText_dim", type=int, required=False, help="fastText word embedding dimension, default 300")
    args = parser.parse_args()

    configs = load_json(args.json_path)
    dim = args.fastText_dim
    main(dict2dotdict(configs), dim)
