coverage run -m unittest tests.test_download
coverage run -a -m unittest tests.test_datasets
coverage run -a  -m unittest tests.test_embeddings
coverage run -a  -m unittest tests.test_bert_models
coverage run -a  -m unittest tests.test_flair_models
coverage run -a  -m unittest tests.test_spacy_models
coverage run -a -m unittest tests.test_xlmr_models
coverage run -a -m unittest tests.test_electra_models