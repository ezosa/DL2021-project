from data import parse_reuters_articles
import pandas as pd
import pickle

articles = parse_reuters_articles("REUTERS_CORPUS_2/")


df = pd.DataFrame(articles)
df = df.convert_dtypes()
df["date"] = pd.to_datetime(df["date"])

# convert list of topic codes to space-separated string
codes = [" ".join(articles[i]['codes']) for i in range(len(articles))]
df["codes"] = codes

# create code2id dictionaru
codes_list = [articles[i]['codes'] for i in range(len(articles))]
codes_list = sorted(list(set([c for codes in codes_list for c in codes])))
print("all codes:", len(codes_list))
codes_dictionary = dict([(codes_list[index], index) for index in range(len(codes_list))])
labels_filename = "labels2id.pkl"
with open(labels_filename, "wb") as pf:
    pickle.dump(codes_dictionary, pf)
    pf.close()
    print("Saved labels dictionary to", labels_filename)

# save dataframe as csv
dataset = df[["date", "headline", "text", "codes"]]
del df
dataset.to_csv("data.csv", index=False)
print("dataset:", dataset.shape)
del dataset

