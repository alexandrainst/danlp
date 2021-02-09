How do I contribute?
====================

If you want to contribute to the DaNLP project, your help is very welcome. 
These guidelines will help you contributing to the [repository](https://github.com/alexandrainst/danlp) for: 
* adding new [tutorials](#tutorials)
* adding new [benchmarks](#benchmarking)
* adding new [models and datasets](#models-datasets)


## About Pull Requests

If you don't know how to contribute to an open source repository (i.e. open a pull request), 
[here](https://github.com/firstcontributions/first-contributions) is an example tutorial. 

Please, follow the **commit message style** described [here](https://chris.beams.io/posts/git-commit/). 

Before pushing a pull request, make sure that the tests pass running `python -m unittest discover` (or using `coverage run -m unittest discover`).

## Adding new tutorials {#tutorials}

You're welcome to help us writing new jupyter notebook tutorials (in English) about how to use and apply Danish NLP. 

For instance, you can write about: 
* how to use DaNLP tools for a specific applicative case, e.g.:
   * NER for recommendation system, customer support
   * Dependency parsing for question answering, building a knowledge graph
   * ...
* how to train a model for a specific task
* ...

For inspiration, you can have a look at our current tutorials [here](https://github.com/alexandrainst/danlp/tree/master/examples/tutorials).

Keep in mind that DaNLP focuses on industry-friendly Danish NLP.
So the examples should be using tools and data that are freely available for commercial purpose.

Steps for adding a tutorial to the DaNLP repo: 
* add the jupyter notebook tutorial to the `examples/tutorials` folder
* update the [README.md](examples/tutorials/README.md) in the same folder with adding to the list of tutorials: the name of the tutorial (with a brief description)


## Benchmarking external models {#benchmarking}
 
We update our benchmark [scripts](https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks) when :
* we add a new dataset to DaNLP ;
* we add a new model to DaNLP ;
* new (commercialy available) tools are realeased for Danish NLP and we want to compare it to the DaNLP models and evaluate it against our data.

Refer to the specific [benchmark](#benchmark) subsection if you are adding a new model or dataset. 

If you want to benchmark a model that is not part of the DaNLP package, it is possible (we evaluated, for example, the NERDA models on our NER dataset, see [script](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/ner_benchmarks.py)). 

Steps when evaluating a new model :
* add the code to the corresponding script in the benchmarks folder: in `examples/benchmarks/ner_benchmarks.py` for example when benchmarking a new NER model
* add required packages in `requirements_benchmarks.txt`
* update our documentation with the results from benchmarking.


## Adding new models or datasets {#models-datasets}

If you want to add a new model or dataset to the DaNLP repository, contact us (danlp@alexandra.dk) first in order to:
* make sure that the model or data is in line with our focus (i.e. industry friendly Danish NLP) ;
* send us your model or dataset, so that we can upload it to our server.

Code for loading/using models and datasets should be added to, respectively, the [models](https://github.com/alexandrainst/danlp/tree/master/danlp/models) and [datasets](https://github.com/alexandrainst/danlp/tree/master/danlp/datasets) folders. 
Each model or dataset should be provided with at least one `load` function. 

When you add code for a new model or a new dataset to the danlp repository, you should also: 

1. add code for [testing](#test) it (**required**)
2. add code for evaluating it against our [benchmarks](#benchmark) (optional but that would greatly help us, and the community using it)
3. add some [documention](#documentation) about it (optional but that would greatly help us, and the community using it)

Include all of this in your pull request. 
Following are more details about these 3 steps.

### Test 

Add a test for your model or dataset to the [test](https://github.com/alexandrainst/danlp/tree/master/tests) folder:

- in `tests/test_datasets.py` for a new dataset,
- in `tests/test_{framework}_models.py` for a new model (where `framework` is `spacy`, `flair` or `bert` -- or create a new test file if you introduce a new framework).

Then run the test(s) (e.g. `python -m unittest tests/test_dataset.md`) to check that your code doesn't fail. 

### Benchmark

Benchmark your model on our dataset(s) or our model(s) on your dataset.

Update or create a new [benchmark script](https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks): 
in `examples/benchmarks`, add the code for benchmarking in the corresponding task file. 
If your model introduces a new task, create a new benchmark file for this task. 

Don't forget to add any potential required packages in `requirements_benchmarks.txt`. 

Update our [documentation](https://github.com/alexandrainst/danlp/tree/master/docs/docs/tasks) with the results from benchmarking -- in the corresponding `{task}.md` file or create a new one for a new task. 

If you are adding a new task (hence a new benchmark script), update the list of scripts in the [README](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/README.md) file.

### Documentation

Add (markdown) documentation in `docs`: 

- in `datasets.md` for a new dataset
- in `tasks/{task}.md` for a new model for a specific task (e.g. in `tasks/pos.md` for a POS-tagger); create a new file if introducing a new task. You can also add `frameworks` documentation with examples of how to use the model. 
