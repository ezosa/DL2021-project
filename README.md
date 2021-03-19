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

### Dataset stats
#### 103 classes
#### Median labels per article: 3
#### Mean article length: 103

### Preliminary results 

#### LSTM (300D GloVe embeddings; 20 epochs; 128 LSTM hidden size; dropout=0.5; Adam opt; early stop):
 - macro F1: 57.78%
 - macro Recall: 52.65%
 - macro Precision: 69.58%

#### GRU (300D GloVe embeddings; 2 layers; 20 epochs; 128 LSTM hidden size; 128 hidden MLP; dropout=0.25; Adam opt; early stop):
 - macro F1: 60.65%
 - macro Recall: 57.58%
 - macro Precision: 68.90%

#### CNN (300D GloVe embeddings; 20 epochs; 100 kernels of sizes 2, 3, 4, 5; stride 1; dropout=0.25; Adam opt; early stop):
 - macro F1: 68.51%
 - macro Recall: 62.49%
 - macro Precision: 79.49%
