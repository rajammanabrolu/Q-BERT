# Q*BERT
![Q*BERT](/images/qbertpixel.png)

Code accompanying paper [_How to Avoid Being Eaten by a Grue:  Structured Exploration Strategies for Textual Worlds_](https://arxiv.org/abs/2006.07409)
 by Prithviraj Ammanabrolu, Ethan Tien, Matthew Hausknecht, and Mark O. Riedl
 
Please use this Bibtex to cite us:
```
@article{ammanabrolu20how,
  title={How to Avoid Being Eaten by a Grue: Structured Exploration Strategies for Textual Worlds},
  author={Ammanabrolu, Prithviraj and Tien, Ethan and Hausknecht, Matthew and Riedl, Mark O.},
  journal={CoRR},
  year={2020},
  url={http://arxiv.org/abs/2006.07409},
  volume={abs/2006.07409}
}
```
 
Structured exploration using knowledge graph A2C agents.
Overall and architecture one-step knowledge graph extraction is seen below: in the Jericho-QA format
 architecture at time step _t_. At each step the ALBERT-QA model extracts a relevant highlighted entity set _V_t_ by
  answering questions based on the observation, which is used to update the knowledge graph.
  
![arch](/images/qbertall.png)

 
 Underlying A2C code is adapted from https://github.com/rajammanabrolu/KG-A2C.
Go-Explore code adapted from https://github.com/uber-research/go-explore.

# Quickstart
**Step 1:** Install Dependencies: Jericho==2.4.2, Redis, Pytorch >= 1.2  
Full list of dependencies in conda environment file `environment.yml` 
```bash
conda env create -f environment.yml
source activate qbert
python -m spacy download en
```

**Step 2:** Download ROM files for games from [https://github.com/BYU-PCCL/z-machine-games/archive/master.zip](https://github.com/BYU-PCCL/z-machine-games/archive/master.zip)

**Step 3:** Train BERT model.  
**Jericho-QA** Datafiles can be downloaded [here](https://1drv.ms/u/s!Ajlo4u0ek6Wha43AjKWKf-2aaJg?e=Vawpjp).
```bash
cd qbert/extraction
python run_squad.py --model_type albert --model_name_or_path albert-large-v2 --do_train  --train_file data/cleaned_qa_train.json --predict_file data/cleaned_qa_dev.json --per_gpu_eval_batch_size 8 --learning_rate 3e-5 --max_seq_length 512 --doc_stride 128 --output_dir ./models/ --warmup_steps 814 --max_steps 8144 --version_2_with_negative --gradient_accumulation_steps 24 --overwrite_output_dir
```

(Optional) Evaluate BERT model:
```bash
cd qbert/extraction
python run_squad.py --model_type albert --model_name_or_path model_name_here --do_eval --train_file data/cleaned_qa_train.json --predict_file data/cleaned_qa_dev.json --per_gpu_eval_batch_size 8 --learning_rate 3e-5 --max_seq_length 512 --doc_stride 128 --output_dir ./models/ --warmup_steps 814 --max_steps 8144 --version_2_with_negative --gradient_accumulation_steps 24 --overwrite_output_dir
```

**Step 4:** Train Q*BERT
```bash
cd qbert
mkdir models && mkdir models/checkpoints
python train.py --rom_file_path path_to_your_rom  --tsv_file ../data/rom_name_here --attr_file attrs/rom_name_here --training_type trainingtype --reward_type rew
```

For example, to run the game _zork1_ with MC!Q*BERT, with reward type Game+IM:
```bash
cd qbert
mkdir models && mkdir models/checkpoints
python train.py --rom_file_path roms/zork1.z5 --tsv_file ../data/zork1_entity2id.tsv --attr_file attrs/zork1_attr.txt --training_type chained --reward_type game_and_IM
```

This will produce a number of files, including progress.csv listing averaged scores and other metrics during agent exploration, to be used for evaluation and analysis.

# Q*BERT flags
`--training_type` can be ['base', 'chained', 'goexplore'], which will train base Q\*BERT, MC!Q\*BERT, or GO!Q\*BERT respectively  
`--reward_type` can be ['game_only', 'IM_only', 'game_and_IM'], IM meaning Intrinsic Motivation (calculated by size set of all edges seen before in KG)  
`--intrinsic_motivation_factor` a float constant multiplied to IM reward (only used in IM_only reward. game_and_IM reward = base_score + IM * (base_score + episilon) / max_game_score)  
`--goexplore_logger` goexplore logger logging each cell exploration and its obs  
`--extraction` confidence threshold for Albert-QA entity extraction 

## MC!Q*BERT only flags
`--patience` is the max number of steps taken before we trigger a 'bottleneck', and begin refreshing training from the previously best seen state  
`--buffer_size` the max number of valid steps we keep track of up until the current state to begin stepping back from  
`--patience_valid_only` an option to only count towards patience when a valid action is taken  
`--patience_batch_factor` if *patience_valid_only* is True, a 'bottleneck' is triggered when this percentage of a batch has valid steps over *patience*  
`--chained_logger` chained logger location that logs the steps where bottlenecks are triggered and the obs at that state  
`--clear_kg_on_reset` boolean to clear KG upon refresh

