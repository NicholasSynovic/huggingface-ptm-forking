#!/bin/bash

modelName=""

while getopts "m:" opt; do
    case $opt in
        m)
        modelName="$OPTARG"
        ;;
        \?)
        echo "Invalid option: -$OPTARG" >&1
        exit 1
        ;;
    esac
done

if [ -z "$modelName" ]; then
  echo "Usage: $0 -m <modelName>"
  exit 1
fi

fullURL="https://huggingface.co/$modelName"

response=$(curl -s "$fullURL") 

targetString=$(echo "$response" | sed -n -E 's/.*<a rel="noopener nofollow" href="([^"]+)">[^<]+<\/a>.*/\1/p')

if [ -n "$targetString" ]; then
    nonURL=$(echo "$targetString" | awk -F 'https://huggingface.co/' '{print $2}')
    echo "Base model for $modelName:"
    echo "$nonURL"
else
    echo "Error: Target string not found in the model information for $modelName."
fi
