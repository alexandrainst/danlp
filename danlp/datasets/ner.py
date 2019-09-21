import os

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func

AVAILABLE_NER_DATASETS = ['wikiann', 'ddt']

# TODO: Load it as a corpus for flair
# TODO: Make a pull request to flair similar to this:
# https://github.com/zalandoresearch/flair/issues/383

# TODO: Load it as jsonlines with spacy


def load_ner_as_conllu(dataset: str, predefined_splits: bool = False, cache_dir: str = DEFAULT_CACHE_DIR):
    """
    Load it with IOB

    :param cache_dir:
    :param predefined_splits:
    :param dataset:
    """
    assert dataset in AVAILABLE_NER_DATASETS, "Only " + ", ".join(AVAILABLE_NER_DATASETS) + " datasets are available"

    import pyconll

    download_dataset(dataset, process_func=_unzip_process_func)

    file_extension = 'conllu'
    dataset_dir = 'dataset'
    dataset_path = os.path.join(cache_dir, dataset)

    parts = [None, None, None]  # Placeholder list to put predefined parts of dataset [train, dev, test]
    for i, part in enumerate(['train', 'dev', 'test']):
        file_name = "{}.{}.{}".format(dataset, part, file_extension)
        file_path = os.path.join(dataset_path, file_name)

        if not os.path.isfile(file_path):
            continue

        parts[i] = pyconll.load_from_file(file_path)

    if not _any_part_exist(parts):  # Then there is no predefined splits for the dataset
        file_name = "{}.{}".format(dataset, file_extension)  # ddt.conllu
        file_path = os.path.join(dataset_path, file_name)
        assert os.path.isfile(file_path), "The file {} should have existed".format(file_path)

        conll_obj = pyconll.load_from_file(file_path)

        if predefined_splits:  # If user wants predefined splits we assume the dataset part is the train set
            return [conll_obj, None, None]
        return conll_obj

    # if predefined_splits: then we should return three files
    if predefined_splits:
        return parts  # At least one of the part (ie. a train, dev or test set) are defined

    # Merge the splits to one single dataset
    parts[0].extend(parts[1])
    parts[0].extend(parts[2])

    return parts[0]


def _any_part_exist(dfs: list):
    for df in dfs:
        if df is not None:
            return True
    return False

