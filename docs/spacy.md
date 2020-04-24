# SpaCy model in Danish 

spaCy is an industrial friendly open source framework for doing NLP, and you can read more about it on there [homesite](https://spacy.io/) or [gitHub](https://github.com/explosion/spaCy).

This project support a Danish spaCy model that can easily be loaded with the DaNLP package. 

Supporting Danish directly in the spaCy framework is under development  and the progress can be follow here [issue #3056](https://github.com/explosion/spaCy/issues/3056). 

Note that the two model is not the same, e.g. the spaCy model in DaNLP performers better on Name Entity Recognition due to more training data.  However the extra training data is not open source and can therefore not be included in the spaCy framework itself as it contravenes the guidelines. 

The spaCy model comes with **tokenization**, **dependency parsing**, **part of speech tagging** , **word vectors** and **name entity recognition**. 

The model is trained on the [Danish Dependency Treebansk (DaNe)](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>), and with additional data for NER  which originates from News article form a collaboration with InfoMedia. 

For comparison to other models and additional information of the tasks, check out the task individual pages for [part of speech tagging](<https://github.com/alexandrainst/danlp/blob/master/docs/models/pos.md>) , [word embedding](<https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md>) and [name entity recognition](<https://github.com/alexandrainst/danlp/blob/master/docs/models/ner.md>).

#### Performance on spaCy Model

The following lists the  performance scores of the spaCy model provided in DaNLP pakage. The scores and elaborating scores can be found in the file meta.json that is shipped with the model when it is downloaded. 

| Task                    | Measures | Scores |
| ----------------------- | -------- | :----- |
| Dependence parsing      | uas      | 81.63  |
| Dependence parsing      | las      | 77.22  |
| Part of speech tags     | accuracy | 96.40  |
| Name entity recognition | f1       | 80.50  |
|                         |          |        |





## :hatching_chick: Getting started with the spaCy model

Below is some small snippets to get started using the spaCy model within the DaNLP package. More information about using spaCy can be found on spaCy own [page](https://spacy.io/).  

**First load the libraries and the model**

```python
# Import libaries
from danlp.models import load_spacy_model
from spacy.gold import docs_to_json
from spacy import displacy

#Downoad and load the spacy model using the daNLP wrapper fuction
nlp = load_spacy_model()
```

**Use the model to determined linguistic features**

```python
# Construkt the text to a container "Doc" obejct
doc = nlp("Spacy er et godt værtøj,og det virker på dansk")

# prepare some pretty printing
lingvistisk_features=['Text','Lemma','POS', 'Dep', 'Form', 'Bogstaver', 'stop ord']
head_format ="\033[1m{!s:>11}\033[0m" * (len(lingvistisk_features) )
row_format ="{!s:>11}" * (len(lingvistisk_features) )

print(head_format.format(*lingvistisk_features))
# printing for each token in det docs the coresponding linguistic features
for token in doc:
    print(row_format.format(token.text, token.lemma_, token.pos_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop))
    
```

![](/imgs/ling_feat.png)

**Visualizing the dependency tree:**

```python
# the spacy framework provides a nice visualizatio tool!
# This is run in a terminal, but if run in jupyter use instead display.render 
displacy.serve(doc, style='dep')
```



![](/imgs/dep.png)

## :hatching_chick: Start ​training you own text classification model

The spaCy framework provide and easy command line tool for training an existing model, for example by adding a text classifier.  This short example shows how to do so using your own annotated data. It is also possible to use any static embedding provided in the DaNLP wrapper. 

As an example we will use a small dataset for sentiment classification on twitter. The dataset is under development at will be added in the daNLP package when ready, and the spacy model will be updated with the classification model as well.  

 **The first thing is to convert the annotated data into a data format readable by spaCy**

Imagine you have the data in a e.g csv format and have it split in development and training part.  Our  twitter data have  (in time of creating this snippet)  973 training examples and 400 evaluation examples, with the following labels : 'positive' marked by 0, 'neutral' marked by 1, and 'negative' by 2. Loaded with pandas dataFrame it look like this:  

![](/imgs/data_head.png)

It need to be convert into the format expected by spaCy for training the model, which can be done as follows:

```python
#import libaries
import srsly
import pandas as pd
from danlp.models import load_spacy_model
from spacy.gold import docs_to_json

# load the spacy model 
nlp = load_spacy_model()
nlp.disable_pipes(*nlp.pipe_names)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, first=True)

# function to read pandas dataFrame and save as json format expected by spaCy
def prepare_data(df, outputfile):
    # choose the name of the columns containg the text and labels
    label='polarity'
    text = 'text'
    def put_cat(x):
        # adapt the name and amount of labels
        return {'positiv': bool(x==0), 'neutral': bool(x==1), 'negativ': bool(x==2)} 
    
    cat = list(df[label].map(put_cat))
    texts, cats= (list(df[text]), cat)
    
    #Create the container doc object
    docs = []
    for i, doc in enumerate(nlp.pipe(texts)):
        doc.cats = cats[i]
        docs.append(doc)
    # write the data to json file
    srsly.write_json(outputfile,[docs_to_json(docs)])
    
    
# prepare both the training data and the evaluation data from pandas dataframe (df_train and df_dev) and choose the name of outputfile
prepare_data(df_train, 'train_sent.json')
prepare_data(df_dev, 'eval_dev.json')

```

The data now look like this cuted snippet:

![](/imgs/snippet_json.png)

**Ensure you have the models and embeddings download**

The spacy model and the embeddings most be download. If you have done so it can be done by running the following commands. It will by default be placed in the cache directory of danlp, eg. "/home/USERNAME/.danlp/".

```python
from danlp.models import load_spacy_model
from danlp.models.embeddings  import load_wv_with_spacy

#download the sapcy model
nlp = load_spacy_model()

# download the static (non subword) embedding of your choice
word_embeddings = load_wv_with_spacy('cc.da.wv')
```



**Now train the model through the terminal**

Now train the model thought the terminal by pointing to the path of  the desired output directory, the converted trainings data, the converted development data, the base model and the embedding. The specify that the trainings pipe.  See more parameter setting [here](https://spacy.io/api/cli#train) . 

```
python -m spacy train da spacy_sent train.json test.json  -b '/home/USERNAME/.danlp/spacy' -v '/home/USERNAME/.danlp/cc.da.wv.spacy' --pipeline textcat
```



**Using the trained model**

Load the model from the specified output directory. 

```python
import spacy

#load the trained model
output_dir ='spacy_sent/model-best'
nlp2 = spacy.load(output_dir)

#use the model for prediction
def predict(x):
    doc = nlp2(x)
    return max(doc.cats.items(), key=operator.itemgetter(1))[0]
predict('Vi er glade for spacy!')

#'positiv'

```

