from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='DaNLP',
    version='0.0.1',
    author="Alexandra Institute",
    author_email="dansknlp@alexandra.dk",
    description="NLP in Danish",
    packages=find_packages(),
    install_requires=['tqdm'],
    license='BSD 3-Clause License',
    long_description=open('README.md').read(),
)
