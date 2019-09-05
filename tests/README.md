Tests in DaNLP
==============
In order to make the CI more efficient, some of the models have been shrunk when running the tests.
Currently the static word embeddings as well as the subword embeddings (fastText) has been made smaller.

## Smaller static word embeddings
To shrink the static word embeddings the number of word vectors has simply been reduced to the 5000 most frequent 
words. This has been done with the `wiki.da.wv` embeddings using the following code. The code is inspired
from an answer on [stack overflow](https://stackoverflow.com/a/53899885).

```python
words_to_trim = wv.index2word[5000:]
ids_to_trim = [wv.vocab[w].index for w in words_to_trim]

for w in words_to_trim:
    del wv.vocab[w]

wv.vectors = np.delete(wv.vectors, ids_to_trim, axis=0)
wv.init_sims(replace=True)

for i in sorted(ids_to_trim, reverse=True):
    del(wv.index2word[i])
```

## Smaller subword embeddings
To make smaller subword embeddings we have trained a new fastText model on the train part of the [Danish Dependency Treebank](<https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master>) dataset.
The training has been done using the official [fastText implementation](https://github.com/facebookresearch/fastText/) 
with the following command.

```bash
./fasttext skipgram -input ddt_train.txt -output ddt.swv -cutoff 5000
```