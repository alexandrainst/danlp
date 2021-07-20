import os
from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "danlp", "about.py"), encoding="utf8") as f:
    about = {}
    exec(f.read(), about)

setup(
    name='danlp',
    description=about['__summary__'],
    author=about["__author__"],
    author_email=about["__email__"],
    version=about["__version__"],
    url=about["__url__"],
    license=about["__license__"],
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'conllu',
        'pandas',
        'pyconll', 
        'tqdm', 
        'tweepy'
    ],
    extras_require={
        'all' : [
            'transformers<=4.3.3',
            'gensim<=3.8.1',
            'torch<=1.7.1',
            'flair<=0.8',
            'spacy<=2.2.3',
            'allennlp<=2.5.0'
        ]
    },
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
    ]
)
