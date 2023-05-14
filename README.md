# Lux AI Season 2 Competition Bot

## Overview

Logic-based bot built for the [Lux AI Challenge Season 2](https://www.kaggle.com/competitions/lux-ai-season-2/overview). A summary of the bot can be found on [Kaggle](https://www.kaggle.com/competitions/lux-ai-season-2/discussion/407723).

## Installation

After cloning the environment install the Conda environment by running the following commands in the cloned directory

```
conda env create -n luxai_s2 -f environment.yaml
conda activate luxai_s2
```

Now you are able to self-play matches locally using the following command:
```
luxai-s2 main.py main.py -o main.json
```

The output of this match will be stored in main.json and this file can be dragged into the [Lux Visualizer](https://s2vis.lux-ai.org/#/) to see a replay of the match.


## Structure
### Directories
- objects: functionality which also has a meaning in the game, e.g. Coordinates, Units and Actions
- lux: starter files from the [LuxAI Python starter kit](https://github.com/Lux-AI-Challenge/Lux-Design-S2/tree/main/kits/python), slightly adapted
- logic: functionality purely for directing the units and factories, e.g. goals
- search: A* [link] implementation to find optimal paths from one coordinate to another
- utils: utils
- tests: tests
- run_matches: functionality to run matches or parse results of match output

### Files
- agent.py: agent
- config.py: settings controlling the behavior of the agent
- environment.yaml: conda environment that is installed in the competition environment. luxai_s2 packaged added for convenience.
- exceptions.py: exceptions
- main.py: main script from LuxAI starter kit
