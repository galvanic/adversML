
###
### ONE-LINERS to analyse log files
###

## TODO turn into functions

LOG_FILEPATH=./1608161201.log
export LOG_FILEPATH

## get epoch, cost and error for one experiment
EXPERIMENT_NUM=0
cat $LOG_FILEPATH 2>/dev/null | grep "_$EXPERIMENT_NUM]" 2>/dev/null | grep -P -e 'cost|error' 2>/dev/null | cut -d'=' -f2 | xargs -n2 | cat -n

## to start debugging a certain experiment
EXPERIMENT_NUM=0
cat 1608161201.log | grep -P -e "experiment #\d{10}_$EXPERIMENT_NUM\]" | cut -c69-174 | less

## get results with experiment numbers and a pattern highlighted to delimit parts visually
PATTERN=0.5
cat $LOG_FILEPATH | grep metrics -A 100000 |  awk '{print NR-3 "\t" $0}' | sed 's/-[0-9]\b/  /g' | grep -P -e "$PATTERN|$"

## to see if mini-batch, batch, stochastic are working
cat 1608171206.log | grep -P -e "experiment_id': |gradient_descent_method': | samples" | sed "s/samples/\'samples/g" | grep -o -P -e "'.*" | cut -c2-210 | less

## experiment number, number of epochs, error at end (to get idea of convergence)
## /!\ only works if experiments were run serially, not using threads /!\
cat $LOG_FILEPATH | grep -P -e 'gradientdescent| performance' | grep -P -e '(error|epoch|INFO: performance).*' -o | grep -o -P -e '\d\.?\d{0,2}|: perf' | xargs -n2 |  grep perf -B 1 | xargs -n3 -d'\n' | cut -d' ' -f1-2 | sed 's/ /\t/g' | awk '{print NR-1 "\t" $0}'

