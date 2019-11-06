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
    install_requires=['tqdm', 'pyconll'],
    scripts=['danlp/datasets/wiki_downloader.sh'],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
    ]
)
