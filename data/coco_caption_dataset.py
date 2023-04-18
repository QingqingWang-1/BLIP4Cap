import os 
import json 
from torch.utils.data import Dataset 
from PIL import Image 
from data.data_utils import pre_caption 

class coco_train(Dataset):
    def __init__(self, transform, image_root, ann_json, max_words=30, prompt=''):
        self.annotation = json.load(open(ann_json, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {} #for retrieval task
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]
    
class coco_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_json):
        self.annotation = json.load(open(ann_json, 'r'))
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image'].split('/')[-1].strip(.jpg).split('_')[-1] #to save predicted results with the same name as image itself

        return image, int(img_id)
        
    