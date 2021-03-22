# Deep Learning Project [dtrizna, ezosa]

Dataset, overall ~300k articles.  
Takes ~150s to read & parse XML data.

```bash
% python3 Repository/data.py REUTERS_CORPUS_2
 [*] Reading articles: 299770
 [!] Took 150.61s
```

### Dataset stats
- 103 classes
- Median/mean labels per article: 3/3.2
- Mean article length: 203

### Preliminary results 

#### LSTM (300D GloVe embeddings; 2 layers; 20 epochs; 128 LSTM hidden size; 128 hidden MLP; dropout=0.25; Adam opt; early stop):
 - macro F1: 60.91%
 - macro Recall: 56.57%
 - macro Precision: 71.90%

#### GRU (300D GloVe embeddings; 2 layers; 20 epochs; 128 LSTM hidden size; 128 hidden MLP; dropout=0.25; Adam opt; early stop):
 - macro F1: 60.65%
 - macro Recall: 57.58%
 - macro Precision: 68.90%

#### CNN (300D GloVe embeddings; 20 epochs; 100 kernels of sizes 2, 3, 4, 5; stride 1; dropout=0.25; Adam opt; early stop):
 - macro F1: 68.51%
 - macro Recall: 62.49%
 - macro Precision: 79.49%

#### FFNN (S-BERT, 100 epochs, dropout=0.5, Adam opt; early stop):
 - macro-F1: 52.43%
 - macro Recall: 46.55%
 - macro Precision: 67.09%