This project is a tool that uses a 4 stage approach to enriching large json files of newspaper articles with the goal of identifying:

- Does the article describe a flood? (Stages 1 and 2)
- Does the flood take place in Ontario (Stage 2)
- What was the date and exact location of the flood (Stage 3)
- What were the impacts of the flood? (Stage 4)

Each stage works in a separate subdirectory.

- stage1-bert: Stage 1 uses BERT to conduct a high-recall (>95%), relatively lower-precision first pass at excluding articles that are clearly not about floods
- stage2: Stage 2 uses a local LLM to conduct a higher-precision evaluation of whether the article is describing a flood and whether it took place in Ontario
- stage3: Stage 3 uses a local LLM to identify the exact data and location of the flood and uses a geocoding API (OSM nominatum) to provide latitude/longitude coordinates for location
- stage4: Stage 4 uses an external LLM to extract relevant flood impact information

The project directory is structured as follows:

annotation/ - a flexible utility tool for labeling newspaper articles
annotation/annotation_tool.html - locally hosted browser-based tool which provides simple interface for annotating articles
annotation/schema.json - customizable schema for annotations
annotation/split-json.js - utility tool for susetting a random number of articles from the main dataset for annotating

data/ - put your initial dataset plus any labeled data here. store stage-specific data in stage-specific directories

results/ - 

mlruns/ - created by BERT during Stage 1
modesl/ - created by BERT during Stage 1

stage1-bert/
stage1-bert/bert-data-splitter.py - used for creating necessary training and testing data for BERT training
stage1-bert/bert-train-high-recall.py - training script for BERT
stage1-bert/bert-inference.py - executes the BERT model against the primary dataset once trained
stage1-bert/config.py - redundant file until I can implement proper _init.py_ setup to allow a single config.py for all stages
stage1-bert/utils.py -  - redundant file until I can implement proper _init.py_ setup to allow a single util.py for all stages


config.py - various model config and filepath settings
requirements.txt - dependencies, run pip install -r requirements.txt to set up
utils.py - utility functions that are shared across various stages
