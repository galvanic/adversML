
###
### to follow running experiment
###

LOG_FILEPATH=$1
export LOG_FILEPATH

TOTAL_NUM=`cat $LOG_FILEPATH | grep -o -P -e '\b\w+ of specifications: \d+\b' | grep -o -P -e '\d+'`
NUM_COMPLETED=`cat $LOG_FILEPATH | grep experiment | grep  ': performance:' | grep -o -P -e '_\d{1,3}+' | sed 's/_//g' | wc -l`
echo "$NUM_COMPLETED of $TOTAL_NUM completed"
echo

## overview of experiments run and still running
cat $LOG_FILEPATH | grep -o -P -e '_\d{1,3}+' | sed 's/_//g' | sort | uniq -c | sort -n -r | cat -n | sort -n -r
#cat $LOG_FILEPATH | cut -c38-65 | grep experiment | cut -c26- | sed 's/]//g' | sort -n | uniq -c | cat -n
## TODO for the watch command, monitor whether python Error

## overview of functions run per experiment
#cat $LOG_FILEPATH | grep -o -P -e '_\d{1,3}\] - \w+\.\w+ - ' | sed 's/_//g' | sed 's/]//g' | cut -d'-' -f1-2 | sort | grep -v performexp | uniq -c | sed 's/- \w*\.//g'

