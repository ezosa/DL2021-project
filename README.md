# Deep Learning Project [dtrizna, ezosa]

Dataset, overall ~300k articles.  
Takes ~150s to read & parse XML data.

```bash
% python3 Repository/data.py REUTERS_CORPUS_2
 [*] Reading articles: 299770
 [!] Took 150.61s
```

ToDo:

- ~~need to read labels~~ DONE
- ~~take a look on errors~~ SOLVED, encoding errors
- investigate BERT + pytorch + multi-label options

### Preliminary results 

#### LSTM (300D GloVe embeddings, 15 epochs, Adam opt, 128 hidden size):
 - macro F1: 57.78%
 - macro Recall: 52.65%
 - macro Precision: 69.58%
