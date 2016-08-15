
###
### ONE-LINERS to analyse log files
###

## TODO turn into functions

LOG_FILEPATH=./1608151500.log
export LOG_FILEPATH

## experiment number,number of epochs, error at end (to get idea of convergence) - only works if experiments were run serially, not using threads
cat $LOG_FILEPATH | grep -P -e 'gradientdescent| performance' | grep -P -e '(error|epoch|INFO: performance).*' -o | grep -o -P -e '\d\.?\d{0,2}|: perf' | xargs -n2 |  grep perf -B 1 | xargs -n3 -d'\n' | cut -d' ' -f1-2 | sed 's/ /\t/g' | awk '{print NR-1 "\t" $0}'

## overview of experiments run and still running
cat $LOG_FILEPATH | grep -o -P -e '_\d{1,3}+' | sed 's/_//g' | sort | uniq -c | sort -n -r | cat -n | sort -n -r
#cat $LOG_FILEPATH | cut -c38-65 | grep experiment | cut -c26- | sed 's/]//g' | sort -n | uniq -c | cat -n

## overview of functions run per experiment
#cat $LOG_FILEPATH | grep -o -P -e '_\d{1,3}\] - \w+\.\w+ - ' | sed 's/_//g' | sed 's/]//g' | cut -d'-' -f1-2 | sort | grep -v performexp | uniq -c | sed 's/- \w*\.//g'

## get epoch, cost and error for one experiment
EXPERIMENT_NUM=0
#cat $LOG_FILEPATH 2>/dev/null | grep "_$EXPERIMENT_NUM]" 2>/dev/null | grep -P -e 'cost|error' 2>/dev/null | cut -d'=' -f2 | xargs -n2 | cat -n

