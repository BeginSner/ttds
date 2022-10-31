#! bin/sh
cat $1 | tr " " "\n" | tr "A-Z" "a-z" | sort | uniq -c | sort -n > $1.csv
# cat $1 | perl -p -e "s/[^\w]+/\n/g" | tr "A-Z" "a-z" | sort | uniq -c | sort -n > $1.csv