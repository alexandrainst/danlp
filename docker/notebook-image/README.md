DaNLP Jupyter Notebook Image
============================
This Docker image is build on top of the [Minimal Jupyter Notebook Stack](https://hub.docker.com/r/jupyter/minimal-notebook).
The image includes the DaNLP [notebook tutorials](/examples).

The image build can be found on [Docker Hub](https://hub.docker.com/r/alexandrainst/danlp) and you can pull the image 

```bash
docker pull alexandrainst/danlp:latest-notebook
``` 

To start a jupyter notebook simply run 
```bash
docker run -p 8888:8888 alexandrainst/danlp:latest-notebook
```
