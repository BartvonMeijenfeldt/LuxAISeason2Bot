counter=1
while [ $counter -le 20 ]
do
    echo "Runnning match $counter"
    luxai-s2 agent/main.py agent/main.py -s $counter -v 3 -o data/replay_$counter.json > data/cli_output_$counter.txt
    python parse_cli_output.py --cli_output_path data/cli_output_$counter.txt
    ((counter++))
done
