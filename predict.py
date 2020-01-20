import argparse
import json
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', dest = 'image_path')
parser.add_argument('--checkpoint', dest = 'checkpoint', default = 'script_checkpoint.pth')
parser.add_argument('--topk', dest = 'topk', default = 3)
parser.add_argument('--category_names', dest = 'category_names', default = 'cat_to_name.json')
parser.add_argument('--gpu', dest = 'gpu', default = True)

args = parser.parse_args()
image_path = args.image_path
checkpoint = args.checkpoint
topk = args.topk
category_names = args.category_names
gpu = args.gpu

model = utils.load_checkpoint(checkpoint)
probs, classes = utils.predict(image_path, model, topk, gpu)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
class_names = [cat_to_name[x] for x in classes]

print('Predictions:')
for x, y in zip(probs, class_names):
    print(f'{y} with probability {(x):.2%}')