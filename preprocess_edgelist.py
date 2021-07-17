import pickle

datasets = {"title only", "title and description"}
testbeds = {
    ("local", "windows"),
    ("local", "linux"),
    ("remote", "windows"),
    ("remote", "linux"),
    ("both", "both")
}

for dataset in datasets:
    for (attackType, platform) in testbeds:
        f = open("./data/{}/{}_{}_preprocessed_sent_document.pkl".format(dataset, attackType, platform), 'rb')
        _, docs = pickle.load(f)
        f.close()
        fullwordlist = []
        for doc in docs:
            wordlist = set(doc[0])
            for word in wordlist:
                if word not in fullwordlist:
                    fullwordlist.append(word)
        f = open("./data/{}/{}_{}_fullwordlist.pkl".format(dataset, attackType, platform), "wb")
        pickle.dump(fullwordlist, f)
        f.close()
        edgelist = []
        edgelistfilename = "./data/{}/{}_{}.edgelist".format(dataset, attackType, platform)
        edgelistfile = open(edgelistfilename, "w")
        for doc in docs:
            wordlist = doc[0]
            for j in range(len(wordlist) - 1):
                for k in range(j + 1, len(wordlist)):
                    if wordlist[j] != wordlist[k]:
                        edge = str(fullwordlist.index(wordlist[j])) + " " + (str(fullwordlist.index(wordlist[k])) + "\n")
                        edgelistfile.write(edge)
        edgelistfile.close()