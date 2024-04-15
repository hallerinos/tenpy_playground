#!/bin/bash
skip_headers=0
while IFS=, read -r col1 col2 # col3 col4
do
    if ((skip_headers))
    then
        ((skip_headers--))
    else
        echo "$col1 | $col2 | $col3 | $col4"
    fi
done < input.csv