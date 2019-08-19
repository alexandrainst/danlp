FROM jupyter/minimal-notebook

COPY requirements.txt /tmp/.
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install danlp

# Download notebook examples to image
RUN wget -O ~/work/POS_tagger_exampel.ipynb https://raw.githubusercontent.com/alexandrainst/danlp/master/examples/POS_tagger_example.ipynb
