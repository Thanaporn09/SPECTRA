import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob

class NormalizeCNV(object):
    def __init__(self, mean, std):
        """
        Args:
            mean: A scalar (or 1D tensor) representing the global mean of CNV features.
            std: A scalar (or 1D tensor) representing the global std of CNV features.
        """
        self.mean = mean
        self.std = std

    def __call__(self, cnv):
        # Convert the CNV list/array to a tensor of type float32
        cnv_tensor = torch.tensor(cnv, dtype=torch.float32)
        # Apply standardization: (x - mean) / std
        return (cnv_tensor - self.mean) / self.std
        

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.model = args.model

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids']) 
            self.examples[i]['dataset'] = args.dataset_name

    def __len__(self):
        return len(self.examples)
    
class PathologySingleImageDataset(BaseDataset):

    def find_files_with_substring(self, directory, substring):
        pattern = os.path.join(directory, f"*{substring}*")
        matching_files = glob.glob(pattern)
        return matching_files

    def __getitem__(self, idx):
        example = self.examples[idx]
        if example['dataset'] in ["BRCA","BRCA_R"]:
            image_id = self.find_files_with_substring(self.image_dir, example['id'])[0]
            image_path = os.path.join(self.image_dir, image_id)
        else:
            image_id = example['id']
            image_path = os.path.join(self.image_dir, image_id + '.pt')
        image = torch.load(image_path)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)


        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class PathologyNGenomicsDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None, cnv_transform=None):
        super().__init__(args, tokenizer, split, transform)
        self.cnv_transform = cnv_transform  

    def find_files_with_substring(self, directory, substring):
        pattern = os.path.join(directory, f"*{substring}*")
        matching_files = glob.glob(pattern)
        return matching_files

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = self.find_files_with_substring(self.image_dir, example['id'])[0]
        image_path = os.path.join(self.image_dir, image_id)
        
        image = torch.load(image_path)
        report_ids = example['ids']
        report_masks = example['mask']
        cnv = example['CNV']  

        if self.cnv_transform is not None:
            cnv = self.cnv_transform(cnv)
            
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length, cnv)
        return sample
    