#! /bin/sh -

cache_dir=$1

mkdir -p "$1/wikidata/wikidata" && cd "$1/wikidata/"

echo "Fetching Wikipedia"

mkdir -p extractor && cd extractor

# install the wikiextractor
git clone https://github.com/attardi/wikiextractor
cd wikiextractor

# download
curl -L -O https://dumps.wikimedia.org/dawiki/20191020/dawiki-20191020-pages-articles.xml.bz2

# extract
echo -e "\e[96 Extract the files\e[0m"
python3 WikiExtractor.py -o extracted -b 25M -c -q dawiki-20191020-pages-articles.xml.bz2

# combine into one file
find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dawiki.xml

# clean for xml tags and rename to txt
cat dawiki.xml | grep -v "<" > dawiki.txt


head -c $((2**20)) dawiki-20191020-pages-articles.xml.bz2 | md5

# delete dump and move
yes | rm dawiki-20191020-pages-articles.xml.bz2

cd ../../
mv extractor/wikiextractor/dawiki.txt wikidata/dawiki.txt

yes | rm -r extractor
echo -e "The corpus is fetched. The Danish wikipedia is fetch from \e[96mhttps://dumps.wikimedia.org/\e[0m"
;;
