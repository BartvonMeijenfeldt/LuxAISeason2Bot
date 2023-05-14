# Lux AI Season 2 Competition Bot

## Overview

Logic-based bot built for the [Lux AI Challenge Season 2](https://www.kaggle.com/competitions/lux-ai-season-2/overview) to reach 15th position out of 743 competitors. A summary of the bot can be found on [Kaggle](https://www.kaggle.com/competitions/lux-ai-season-2/discussion/407723). My final 3 submissions for the competition were the more experimental #119 and #118 and the better tested #111. #118 proved best on the leaderboard, obtaining the 15th position. #111 and #119 would have ended up in 17th place.

## Installation

After cloning the project, install the Conda environment by running the following commands in the cloned directory:

```
conda env create -n luxai_s2 -f environment.yaml
conda activate luxai_s2
```

Now you are able to locall run self-play matches using the following command:
```
luxai-s2 main.py main.py -s 1 -v 3 -o main.json
```

The -s argument defines the seed of the map and can be changed to a different number to play matches on different maps. The output of this match will be stored in main.json and this file can be dragged and dropped into the [Lux Visualizer](https://s2vis.lux-ai.org/#/) to see a replay of the match.


## Structure
### Directories
- objects: functionality which also has a meaning in the game, e.g. Coordinates, Units and Actions
- lux: starter files from the [LuxAI Python starter kit](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/python), slightly adapted
- logic: functionality purely for directing the units and factories, e.g. goals
- search: [A*](https://en.wikipedia.org/wiki/A*_search_algorithm) implementation to find optimal paths from one coordinate to another
- utils: utils
- tests: tests
- run_matches: functionality to run matches or parse results of match outputs

### Files
- agent.py: agent
- config.py: settings controlling the behavior of the agent
- environment.yaml: [The LuxAI S2 environment](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/environment.yml) that runs in the competition VMs. luxai_s2 package added for convenience.
- exceptions.py: custom exceptions
- main.py: main script from LuxAI starter kit
