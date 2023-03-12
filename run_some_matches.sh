counter=1
while [ $counter -le 20 ]
do
    luxai-s2 agent/main.py agent/main.py -s $counter -v 2 -o replay_$counter.json
    ((counter++))
done
