#! /bin/sh -
'''
This shell script download and extrat different corpora.
Notice the for the wikipedia, it glone a github repository https://github.com/attardi/wikiextractor
The script is inspired by https://github.com/stefan-it/flair-lms 
'''
# list of posible names
declare -a datasets=("wiki" "euparl" "opensub" "UD_danish")
# creat a folder for corpus and navigate to it 
mkdir -p .corpus && cd .corpus

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
        --wiki) echo "Fetching Wikipedia"
        mkdir -p wikipedia && cd wikipedia
        # install the wikiextractor
        git clone https://github.com/attardi/wikiextractor
        cd wikiextractor
        # download
        curl -L -O https://dumps.wikimedia.org/dawiki/latest/dawiki-latest-pages-articles.xml.bz2
        # extract
        echo -e "\e[96 Extract the files\e[0m"
        python3 WikiExtractor.py -c -b -q 25M -o extracted dawiki-latest-pages-articles.xml.bz2
        # combine into one file
        find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dawiki.xml
        # clean for xml tags and rename to txt
        cat dawiki.xml | grep -v "<" > dawiki.txt
        # delete dump and move
        rm "dawiki-latest-pages-articles.xml.bz2"
        cd ..
        mv wikiextractor/dawiki.txt dawiki.tx
        echo -e "The corpus is fetched. The Danish wikipedia is fetch from \e[96mhttps://dumps.wikimedia.org/\e[0m"
            ;;
            
        --euparl) echo -e "Fetching EuroParl from http://opus.nlpl.eu/Europarl.php"
        mkdir -p europarl && cd europarl
        wget "http://opus.nlpl.eu/download.php?f=Europarl%2Fda.raw.tar.gz"
        unzip -q "download.php?f=Europarl%2Fda.raw.tar.gz" 
        echo -e "\e[96 Extract the files\e[0m"
        # combine all small xlm files
        find Europarl/raw/da/ -name *.xml -exec cat {} + > europarl-combined.xml 
        # remove some xml tags
        sed -i 's/<[^>]*>//g' europarl-combined.xml
        # remove empty lines
        sed -i '/^\s*$/d' europarl-combined.xml
        # rename and delete
        mv europarl-combined.xml europarl-combined.txt
        rm "download.php?f=Europarl%2Fda.raw.tar.gz"
        echo -e "The corpus is fetched. The data  originates from the European Parlemant \e[96mhttp://www.statmt.org/europarl/\e[0m and is gatered by and downloaded from  \e[96mhttp://opus.nlpl.eu/Europarl.php\e[0m"
            ;;
            
        --opensub) echo "Fetching Open subtitels 2018"
        mkdir -p opensub && cd opensub
        wget "http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fda.raw.tar.gz"
        unzip -q "download.php?f=OpenSubtitles2018%2Fda.raw.tar.gz"
        # combine in one xml file
        find OpenSubtitles/raw/da/ -name *.xml -exec cat {} + > opensubtitles-combined.xml
        # remove some xml tags, amd rename tp txt, and delete
        cat opensubtitles-combined.xml | grep -v "<" > opensubtitles-combined.txt
        rm "download.php?f=OpenSubtitles2018%2Fda.raw.tar.gz"
        echo -e "The corpus is fetched. The data  originates from Open subtitels 2018 \e[96mhttps://www.opensubtitles.org/da\e[0m and is gatered by and downloaded from  \e[96mhttp://opus.nlpl.eu/OpenSubtitles2018.php\e[0m"
            ;;
       
        --*) echo -e "bad option $1, your options are: \e[96m${datasets[@]}\e[0m"
            ;;
        *) echo "argument $1"
            ;;
    esac
    shift
done

exit 0