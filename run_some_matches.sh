counter=1
while [ $counter -le 20 ]
do
    echo "Runnning match $counter"
    luxai-s2 main.py main.py -s $counter -v 3 -o data/"$counter"_replay.json > data/"$counter"_cli_output.txt
    python parse_cli_output.py --cli_output_path data/"$counter"_cli_output.txt
    ((counter++))
done
