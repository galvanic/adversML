
###
### to follow running experiment
###

LOG_FILEPATH=$1
export LOG_FILEPATH

TOTAL_NUM=`cat $LOG_FILEPATH 2>/dev/null | head -n 100 | grep -o -P -e '\b\w+ of specifications: \d+\b' | grep -o -P -e '\d+'`
NUM_COMPLETED=`cat $LOG_FILEPATH | cut -d' ' -f10 | grep -P '^performance' | wc -l`
echo "$NUM_COMPLETED of $TOTAL_NUM completed"
echo

## overview of experiments run and still running
cat $LOG_FILEPATH | grep '>>>' | cut -d' ' -f6 | grep -P '^\d' | sort | uniq -c | sort -k1,1n -r

## TODO for the watch command, monitor whether python Error

