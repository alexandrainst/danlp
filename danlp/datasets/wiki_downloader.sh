#! /bin/sh -

# This shell script download and extracts a dump of the danish wikipedia site.
# Notice that the script uses WikiExtractor (https://github.com/attardi/wikiextractor).
# The script is inspired by https://github.com/stefan-it/flair-lms 

cache_dir=$1

FILE="$1/dawiki/dawiki.txt"

if [ -f "$FILE" ]; then
    echo "$FILE exists - exiting";
else
    mkdir -p "$1/dawiki/" && cd "$1/dawiki/"

    echo "Fetching Wikipedia"

    mkdir -p extractor && cd extractor

    # install the wikiextractor
    git clone https://github.com/attardi/wikiextractor
    cd wikiextractor

    # download
    curl -L -O https://dumps.wikimedia.org/dawiki/20191020/dawiki-20191020-pages-articles.xml.bz2

    # extract
    echo "Extracting files with WikiExtractor..."
    python3 WikiExtractor.py -o extracted -b 25M -c -q dawiki-20191020-pages-articles.xml.bz2

    # make chemsum test
    CHECKSUM=$(head -c $((2**20)) dawiki-20191020-pages-articles.xml.bz2 | md5)
    if [ "$CHECKSUM" != "728d5bedcaef9bc7123a514e11bce6b8" ]; then
        echo "CHECKSUM failed - try again";
        exit 1
    fi

    # combine into one file
    find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dawiki.xml

    # clean for xml tags and rename to txt
    cat dawiki.xml | grep -v "<" > dawiki.txt

    # delete dump and move
    yes | rm dawiki-20191020-pages-articles.xml.bz2

    cd ../../
    mv extractor/wikiextractor/dawiki.txt dawiki.txt

    yes | rm -r extractor
    echo "The corpus is fetched. The Danish wikipedia is fetch from https://dumps.wikimedia.org"
fi
