from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from data_utils import clean_document
import pandas as pd
import numpy as np
import time

def encode(df, name=""):
    try:
        df["X"] = df["headline"] + " " + df["text"]
        df["X"] = df["X"].apply(clean_document)
        df = df.reset_index().drop(columns=["index"])
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        now = time.time()
        enc = model.encode(df["X"])
        encdf = pd.DataFrame(enc)
        encdf.columns = [f"dim{c}" for c in range(encdf.shape[1])]
        print(f"[!] Took {time.time() - now}s")
        filename=f"datafiles/{name}_enc_{int(time.time())}"
        encdf.to_csv(filename+".csv")
        encdf.to_feather(filename+".ft")
        print(f"[+] Written to {filename}")
    except:
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    dataset = pd.read_csv("datafiles/data.csv")

    train, test = train_test_split(dataset, test_size=0.1, random_state=0, shuffle=True)
    val = test[:int(test.shape[0]/2)]
    test = test[int(test.shape[0]/2):]

    del dataset

    print("Encoding Test!")
    encode(test, name="test")
    print("Encoding Train!")
    encode(train, name="train")
    # Encoding Test!
    # [!] Took 713.1582820415497s
    # [+] Written to datafiles/test_enc_1616137520
    # Encoding Train!
    # [!] Took 12454.442389965057s
    # [+] Written to datafiles/train_enc_1616150036