import os
import json
import random
import numpy as np


"""
Update on Sep 9, 2024: This doesn't change the dataset size. Hence not running.  
"""


ROOT_DR = f"/checkpoint/sagnikmjr2002/code/ActiveAVSepMovingSource/data/passive_datasets/v1"
assert os.path.isdir(ROOT_DR)

TRAIN_FP = f"{ROOT_DR}/train_locationPredictor/allepisodesPerScene.json"
assert os.path.isfile(TRAIN_FP)

VAL_FP = f"{ROOT_DR}/val_locationPredictor/allepisodesPerScene.json"
assert os.path.isfile(VAL_FP)

# Read the JSON file
with open(TRAIN_FP, 'r') as file:
    data_train = json.load(file)

with open(VAL_FP, 'r') as file:
    data_val = json.load(file)

# Print the length

# random_numbers = random.sample(range(2924973), 100)

#for i in random_numbers:

# we have to write a script for filtering the json based on the condition that max(delx, dely) > MAX_VAL are to be removed. Please note that MAX_VAL when divided by 20 is 2.9.
# Therefore, original value is 2.9 * 20, i.e, 58.

# We filter both train wavs and val wavs file and store them in filtered train wavs and filtered val wavs respectively.

MAX_VAL = 58.0 # global max

for data_json_str in ['val','train']:
    if 'train' in data_json_str:
        data_json = data_train
        dump_fp = f"{TRAIN_FP.split('.json')[0]}_wavs_filtered.json"
    else:
        data_json = data_val
        dump_fp = f"{VAL_FP.split('.json')[0]}_wavs_filtered.json"

    # print('0: ', len(data_json[data_json_str]))
    new_data_list = []
    for data in data_json[data_json_str]:
        abs_del_x, abs_del_y = np.abs(data['target'][0]), np.abs(data['target'][1])
        if np.maximum(abs_del_x, abs_del_y) <= MAX_VAL:
            new_data_list.append(data)
    
    dict_wavs_filtered = {data_json_str:new_data_list}

    # print("1: ", dump_fp, len(dict_wavs_filtered[data_json_str]))

    # import pdb; pdb.set_trace()
    with open(dump_fp, "w") as outfile:    # str(data_json_str) + "_wavs_filtered.json", dump_fp
        # json_data refers to the above JSON
        json.dump(dict_wavs_filtered, outfile)

