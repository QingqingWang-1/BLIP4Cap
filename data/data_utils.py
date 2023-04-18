import re 
import json 
import os 
import torch
import torch.distributed as dist
import utils 

def pre_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower(),)
    caption = re.sub(r"\s{2,}", ' ', caption,)
    caption = caption.rstrip('\n') #remove \n at the end of strings
    caption = caption.strip(' ')#remove ' ' at the begining and the end of strings
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

def save_result(result, result_dir, filename): #only save file in main process
    final_result_file = os.path.join(result_dir, "%s.json"%filename)
    if utils.is_main_process():
        json.dump(result, open(final_result_file, 'w'))

from pycocotools.coco import COCO 
from pycocoevalcap.eval import COCOEvalCap 

def coco_caption_eval(coco_ann_json, results_file):
    coco = COCO(coco_ann_json)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    return coco_eval

    


