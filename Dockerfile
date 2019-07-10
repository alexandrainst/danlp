FROM ubuntu:18.04

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -yq \
                        python3 python3-pip htop nano git wget \
                        libglib2.0-0 autoconf automake \
                        libtool build-essential unzip \
                        libarchive-dev vim \
                        libicu-dev

# Install Python dependencies.
ADD requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

# Welcome message
ADD /docs/imgs/message.txt /
RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' \
        >> /etc/bash.bashrc \
        ; cat message.txt > /etc/motd

WORKDIR /root