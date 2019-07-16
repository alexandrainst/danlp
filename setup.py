from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='danlp',
    version='0.0.1',
    author="Alexandra Institute",
    author_email="dansknlp@alexandra.dk",
    description="DaNLP: NLP in Danish",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['tqdm'],
    license='BSD 3-Clause License',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
    ]
)
