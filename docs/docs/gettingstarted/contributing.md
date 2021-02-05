How do I contribute?
====================

If you want to contribute to the DaNLP project, your help is very welcome. 
Please follow these guidelines for contributing to the [repository](https://github.com/alexandrainst/danlp).


## Contributing to the DaNLP repository

If you don't know how to contribute to an open source repository (i.e. open a pull request), 
[here](https://github.com/firstcontributions/first-contributions) is an example tutorial. 

Please, follow the commit message style described [here](https://chris.beams.io/posts/git-commit/). 

Before pushing your pull request, make sure that the tests pass running `python -m unittest discover` (or using `coverage run -m unittest discover`).

### Adding new models or datasets


When you add a new model or a new dataset to the danlp repository, you should also: 

1. test it
2. benchmark it
3. document it


#### Test 

Add a test for your model or dataset: 

- in `tests/test_datasets.py` for a new dataset,
- in `tests/test_{framework}_models.py` for a new model (where `framework` is `spacy`, `flair` or `bert`) or create a new test file if you introduce a new framework.

Then run the test(s) (e.g. `python -m unittest tests/test_dataset.md`).

#### Benchmark

Benchmark your model or dataset.
In `examples/benchmarks`, add the code for benchmarking in the corresponding task file. 
If your model introduce a new task, create a new benchmark file for this task. 
Don't forget to add any potential required packages in `requirements_benchmarks.txt`. 

#### Document

Add (markdown) documentation in `docs`: 

- in `datasets.md` for a new dataset
- in `tasks/{task}.md` for a new model for a specific task (e.g. in `tasks/pos.md` for a POS-tagger); create a new file if introducing a new task. You can also add `frameworks` documentation with examples of how to use the model. 
