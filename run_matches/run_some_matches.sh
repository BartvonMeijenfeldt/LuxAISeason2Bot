counter=1
while [ $counter -le 20 ]
do
    echo "Runnning match $counter"

    replay_file="data/"$counter"_replay.json"
    cli_output_file="data/"$counter"_cli_output.txt"

    luxai-s2 main.py main.py -s $counter -v 3 -o $replay_file > $cli_output_file
    python run_matches/parse_cli_output.py --cli_output_path=$cli_output_file
    ((counter++))
done
