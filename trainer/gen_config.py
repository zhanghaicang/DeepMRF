import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)

for i in range(10):
    config['train']=config['train']+'_'+str(i)
    with open(sys.argv[2]+'_'+str(i)+'.json', 'w') as f:
        json.dump(config, f)

