

import json

name = 'logs/UVG/HDNeRV3/ConvNeXt_2_256x320_fc_4_5_Dims160,160,160,160,92_Blocks3,3,9,3,3_exp2_f8_k5_e400_warm80_b2_lr0.0005_Fusion6_Strd4,2,2,2,2_eval_normal/results.json'
# Thay 'file.json' bằng đường dẫn tới file JSON của bạn
with open(name, 'r') as f:
    data = json.load(f)


values_without_slash = {key: value for key, value in data.items() if '/' not in key}
print(values_without_slash)