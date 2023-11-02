#!/bin/bash

# Initialize variables for author and model_name
author=""
modelName=""


while getopts "a:m:" opt; do
  case $opt in
    a)
      author="$OPTARG"
      ;;
    m)
      modelName="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

#do -a and -m exist
if [ -z "$author" ] || [ -z "$modelName" ]; then
  echo "Usage: $0 -a <author> -m <modelName>"
  exit 1
fi

#construct complete URL
url="https://huggingface.co/$author/$modelName/raw/main/README.md"

folder="/Users/karolinaryzka/Documents/Samples/identifyGeneratedFromTrainer/readMeFiles"

echo "Constructed URL: $url"

#download the content from the URL
wget -P $folder $url

#run identifyTrainer with full url
python3 identifyTrainer.py "$url"

