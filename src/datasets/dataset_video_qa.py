import torchvision
import os
import cv2
import torch
import random
import numpy as np
import copy
import pysrt
import ftfy
from unidecode import unidecode
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import TWBertBaseDataset
from src.datasets.randaugment import TemporalConsistentRandomAugment


class TWBertVideoQADataset(TWBertBaseDataset):
    open_ended_qa_names = ["frameqa", "msrvtt_qa", "msvd_qa"]

    def __init__(self, task_type, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, ans2label=None,
                 ensemble_n_clips=1, return_label=True, is_train=False, random_sample_clips=True, 
                 video_fmt='.mp4', img_db_type='lmdb'):
        super(TWBertVideoQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, img_db_type=img_db_type,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.task_type = task_type
        self.ans2label = ans2label
        self.num_labels = len(ans2label)
        self.random_sample_clips = random_sample_clips
        self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

        self.video_fmt = video_fmt

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                raise NotImplementedError('Do not support multiple clips for now.')
            else:
                video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt) 
                vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            if self.randaug:
                vid_frm_array = self.randaug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            examples = [self._get_single_example(e) for e in examples]
            #print(vid_frm_array, vid_frm_array.shape)
            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in self.open_ended_qa_names:
            if self.return_label:
                example["label"] = self.ans2label[example["label"]]
        if not self.return_label:
            example["label"] = None
        return example

    def evaluate_qa(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "question_id": int,
                    "answer": int or float, either answer_idx (int)
                }
        Returns:
            TGIF-QA score
        """
        preds = []
        gts = []
        # for frameQA
        answer_types = []
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            msvd_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])}
        )

        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        if self.task_type in self.open_ended_qa_names:  # convert ans_idx, int --> str
            qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            if self.task_type in self.open_ended_qa_names:
                answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        metrics["overall_acc"] = float(np.mean(preds == gts))
        if self.task_type in self.open_ended_qa_names:
            answer_types = np.array(answer_types)
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (
                        preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(
                    np.mean(answer_type_corrects)) if len(answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [
                    1. * len(answer_type_corrects) / len(answer_types),
                    len(answer_type_corrects)]
            metrics["ratios"] = ratios
        return metrics


class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20, task_type="action", n_options=5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        return dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )


class TWBertTVQADataset(TWBertBaseDataset):
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, num_labels=5,
                 ensemble_n_clips=1, return_label=True, is_train=False, random_sample_clips=True, 
                 video_fmt='.mp4', img_db_type='lmdb'):
        super(TWBertTVQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, img_db_type=img_db_type,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.num_labels = num_labels
        self.random_sample_clips = random_sample_clips
        self.qid2data = {d["id"]: d for group in datalist for d in group[1]}

        self.video_fmt = video_fmt

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            '''if self.ensemble_n_clips > 1:
                raise NotImplementedError('Do not support multiple clips for now.')
            else:
                video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt) 
                vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            if self.randaug:
                vid_frm_array = self.randaug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)'''
            vid_frm_array = torch.zeros([16, 3, 224, 224], dtype=torch.uint8) # DUMMY for testing
            infos = self._process_frame_sub(examples[0])
            
            # TODO LOAD FRAME
            image = cv2.imread()
            image = cv2.resize(image, (224,224))
            for i in range(min(len(infos["selected_frames"]), 16)):
                if os.path.exists(infos["selected_frames"][i]):
                    img = cv2.imread(infos["selected_frames"][i])
                    img = cv2.resize(img,(224,224))
                    image = img
                else:
                    img = image
                vid_frm_array[i,:,:,:].copy_(torch.from_numpy(img).permute(2,0,1))
            if len(infos["selected_frames"]) < 16:
                for i in range(16-len(infos["selected_frames"])):    
                    vid_frm_array[i+len(infos["selected_frames"]),:,:,:].copy_(torch.from_numpy(img).permute(2,0,1))
            #print(vid_frm_array, vid_frm_array.shape)


            for example in examples:
                example['sub_txt'] = infos["sub_txt"]
                example["rel_loc"] = "{} to {}".format(int(infos["rel_loc"][0] * 100), int(infos["rel_loc"][1] * 100))
                
            examples = [self._get_single_example(e) for e in examples]        
            
            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
            
    def _process_frame_sub(self, example):
        #max_frame_no = max([int(x.split('.')[0]) for x in os.listdir(self.img_db_dir)])
        #max_time = (max_frame_no - 1) / 3.0
        max_frame_no = 1000000
        max_time = 80
        ts0, ts1 = example['ts']
        ts0 = max(ts0, 0)
        ts1 = min(ts1, max_time)

        segment_size = 35 / self.num_frm
        midpoint = (ts0 + ts1) / 2.0
        midpoint = round(midpoint * 3) / 3
        t_start = midpoint - segment_size * 0.5
        t_end = midpoint + segment_size * 0.5

        times_used0 = [{'start_time': t_start, 'end_time': t_end}]
        for i in range(self.num_frm // 2):
            for delta in [-segment_size, segment_size]:
                t0 = t_start + delta * (i+1)
                t1 = t_end + delta * (i+1)

                t0 = round(t0 * 3) / 3
                t1 = round(t1 * 3) / 3

                if t1 < 0 or t0 > max_time:
                    continue
                if len(times_used0) < self.num_frm:
                    times_used0.append({'start_time': t0, 'end_time': t1})

        times_used0 = sorted(times_used0, key=lambda x: x['start_time'])
        
        selected_frames = []
        times_used = []
        for trow in times_used0:
            t_midframe = (trow['start_time'] + trow['end_time']) / 2.0
            t_mid_3ps_idx = int(round(t_midframe * 3.0)) + 1
            t_mid_3ps_idx = max(t_mid_3ps_idx, 1)
            t_mid_3ps_idx = min(t_mid_3ps_idx, max_frame_no)
            
            for i in range(10):
                fn = os.path.join(self.img_db_dir, example['vid_name'].split(" ")[0]+"_frames", example['vid_name'].split(" ")[-1], f'{t_mid_3ps_idx:05d}.jpg')
                if os.path.exists(fn) or True: # NEED TO BE REPLACED
                    selected_frames.append(fn)
                    times_used.append(trow)
                    break
                else:
                    t_mid_3ps_idx = np.random.randint(int(round(trow['start_time'] * 3.0)), int(round(trow['end_time'] * 3.0)) + 1)
    
        show_subname = example['vid_name'].split(" ")[-1]
        sub_fn = os.path.join()
        def _parse_ts(ts):
            sec = ts.hours * 3600 + ts.minutes * 60 + ts.seconds + ts.milliseconds / 1000.0
            return sec
        for ts in times_used:
            ts['sub'] = []

        bounds = np.array([x['start_time'] for x in times_used] + [times_used[-1]['end_time']])
        
        for sub_item in pysrt.open(sub_fn):
            start_time = _parse_ts(sub_item.start)
            end_time = _parse_ts(sub_item.end)
            mid_time = (start_time + end_time) / 2.0
            pos = np.searchsorted(bounds, mid_time)
            if (pos > 0) and (pos <= len(times_used)):
                times_used[pos-1]['sub'].append(sub_item.text)

        for ts in times_used:
            ts['sub'] = ' '.join(ts['sub'])
            ts['sub'] = unidecode(ftfy.ftfy(ts['sub'])).replace('\n', ' ')

        my_duration = times_used0[-1]['end_time'] - times_used[0]['start_time']
        rel_localized_tstart = (ts0 - times_used[0]['start_time']) / my_duration
        rel_localized_tend = (ts1 - times_used[0]['start_time']) / my_duration
        #print(selected_frames)
        infos = {
            "selected_frames": selected_frames, 
            "sub_txt": " [SEP] ".join([ts['sub'] for ts in times_used if len(ts['sub']) > 0]),
            "rel_loc": (rel_localized_tstart, rel_localized_tend),
        }
        return infos
        

    def _get_single_example(self, data):
        example = dict(
            q_str=data["qa_query"],
            question_id=data["id"],
            qa_choices=data["qa_choices"],
            label=data["qa_label"],
            sub_txt=data["sub_txt"],
            rel_loc=data["rel_loc"],
        )
        return example
    
    
    def evaluate_qa(self, results):
        '''
        results: dict
        {
            question_id: 12222
            logprobs: torch.FloatTensor
            pred: int
        }
        '''
        preds, gts = [], []
        qid2pred_ans = {r["question_id"]: r["pred"] for r in results}
        
        for qid, pred_ans in qid2pred_ans.items():
            gt_ans = self.qid2data[qid]["qa_label"]
            preds.append(pred_ans)
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        metrics["overall_acc"] = float(np.mean(preds == gts))
        
        return metrics
    
    
class TVQACollator(object):
    def __init__(self, tokenizer, max_length=20, n_options=5, use_sub=True, use_rel_loc=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_options = n_options
        self.use_sub = use_sub
        self.use_rel_loc = use_rel_loc

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
    
        # text_str_list = flat_list_of_lists([
        #     [d["q_str"] + " " + "answer:" + " " + d["qa_choices"][i] for i in range(self.n_options)] 
        #     for d in text_examples
        # ])  # (B * n_options, )
        
        text_str_list = []
        for i, d in enumerate(text_examples):
            text_str_list_tmp = []
            for j in range(self.n_options):
                tmp_txt = d["q_str"] + " " + "answer: "
                if self.use_rel_loc:
                    tmp_txt = d["rel_loc"] + " [SEP] " + tmp_txt
                tmp_txt = tmp_txt + d["qa_choices"][j]
                if self.use_sub:
                    tmp_txt = tmp_txt + " [SEP] " + d["sub_txt"]
                text_str_list_tmp.append(tmp_txt)
            text_str_list.append(text_str_list_tmp)

        text_str_list = flat_list_of_lists(text_str_list)
       
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        
        question_ids = [d["question_id"] for d in text_examples]
        return dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
