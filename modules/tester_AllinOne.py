import logging
import os
from abc import abstractmethod

import cv2
import pandas as pd
import torch

from modules.utils import generate_heatmap
from tqdm import tqdm
import logging

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images_id, images, reports_ids, reports_masks = images_id[0], images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_ids.append(images_id)
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            # Convert to pandas DataFrame
            test_res_df = pd.DataFrame(test_res, columns=['Generated Reports'])
            test_gts_df = pd.DataFrame(test_gts, columns=['Ground Truths'])

            # Create DataFrame for IDs
            test_ids_df = pd.DataFrame(test_ids, columns=['Case ID'])

            # Merge the DataFrames
            merged_df = pd.concat([test_ids_df, test_res_df, test_gts_df], axis=1)

            # Save the merged DataFrame to a CSV file
            merged_df.to_csv(os.path.join(self.save_dir, "gen_vs_gt.csv"), index=False)
            test_res_df.to_csv(os.path.join(self.save_dir, "res.csv"), index=False)
            test_gts_df.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False)

        return log
    

class Tester_SPECTRA(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester_SPECTRA, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, cnv) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks, cnv = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), cnv.to(self.device)
                output = self.model(images, cnv, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_ids.append(images_id)
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            # Convert to pandas DataFrame
            test_res_df = pd.DataFrame(test_res, columns=['Generated Reports'])
            test_gts_df = pd.DataFrame(test_gts, columns=['Ground Truths'])

            # Create DataFrame for IDs
            test_ids_df = pd.DataFrame(test_ids, columns=['Case ID'])

            # Merge the DataFrames
            merged_df = pd.concat([test_ids_df, test_res_df, test_gts_df], axis=1)

            # Save the merged DataFrame to a CSV file
            merged_df.to_csv(os.path.join(self.save_dir, "gen_vs_gt.csv"), index=False)
            test_res_df.to_csv(os.path.join(self.save_dir, "res.csv"), index=False)
            test_gts_df.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False)

        return log