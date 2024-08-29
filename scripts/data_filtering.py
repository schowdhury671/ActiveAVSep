import json
import random
import numpy as np

# Read the JSON file
with open('train_wavs.json', 'r') as file:
    data_train = json.load(file)

with open('val_wavs.json', 'r') as file:
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
    else:
        data_json = data_val
    new_data_list = []
    for data in data_json[data_json_str]:
        abs_del_x, abs_del_y = np.abs(data['target'][0]), np.abs(data['target'][1])
        if np.maximum(abs_del_x, abs_del_y) <= MAX_VAL:
            new_data_list.append(data)
    
    dict_wavs_filtered = {data_json_str:new_data_list}

    # import pdb; pdb.set_trace()
    with open(str(data_json_str) + "_wavs_filtered.json", "w") as outfile:
        # json_data refers to the above JSON
        json.dump(dict_wavs_filtered, outfile)

