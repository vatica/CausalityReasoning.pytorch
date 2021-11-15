# -*- coding: utf-8 -*-
# @Time    : 2021年11月15日
# @Author  : Bole Ma
# @File    : main.py   完整模型的训练和测试

import os
import json
import numpy
import random
import logging
import argparse
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import *

""" 读取数据集 """
class CausalDataset(Dataset):
    def __init__(self, args, word_embeddings, tokenizer, device, mode):
        self.data_path = args.data_path
        self.max_obj_len = args.max_obj_len
        self.max_seq_len = args.max_seq_len
        self.word_embeddings = word_embeddings
        self.tokenizer = tokenizer
        self.device = device
        self.mode = mode
        
        # event data
        with open(os.path.join(self.data_path, 'Contextual_dataset.json'), 'r') as f:
            self.raw_dataset = json.load(f)[mode]
        # obj label
        with open(os.path.join(self.data_path, 'objects', 'custom_data_info.json'), 'r') as f:
            self.ind_to_classes = json.load(f)['ind_to_classes']
        # obj data
        with open(os.path.join(self.data_path, 'objects', mode, 'custom_prediction.json'), 'r') as f:
            self.raw_object = json.load(f)

        self.data_set = self.tensorize_example(self.raw_dataset, self.raw_object)
        print('successfully loaded {} examples for {} data'.format(len(self.data_set), mode))

    def tensorize_example(self, raw_data, raw_object):
        tensorized_dataset = list()
        for tmp_video_id in raw_data:
            tmp_video = raw_data[tmp_video_id]
            # if event exists
            if len(tmp_video['image_0']['event']) > 0:
                tensorized_dataset += self.tensorize_frame(tmp_video['image_0'], tmp_video['image_1'], raw_object,
                                                            tmp_video['category'], tmp_video_id.split('_')[1], int(0))
            if len(tmp_video['image_1']['event']) > 0:
                tensorized_dataset += self.tensorize_frame(tmp_video['image_1'], tmp_video['image_2'], raw_object,
                                                            tmp_video['category'], tmp_video_id.split('_')[1], int(1))
            if len(tmp_video['image_2']['event']) > 0:
                tensorized_dataset += self.tensorize_frame(tmp_video['image_2'], tmp_video['image_3'], raw_object,
                                                            tmp_video['category'], tmp_video_id.split('_')[1], int(2))
            if len(tmp_video['image_3']['event']) > 0:
                tensorized_dataset += self.tensorize_frame(tmp_video['image_3'], tmp_video['image_4'], raw_object,
                                                            tmp_video['category'], tmp_video_id.split('_')[1], int(3))
        return tensorized_dataset

    def tensorize_frame(self, image_one, image_two, raw_object, category, video_id, image_id):
        image_one_key = os.path.join(self.data_path, image_one['path'])
        sorted_entities_1 = raw_object[image_one_key]['bbox_labels'][:self.max_obj_len]
        embed_entities_1 = [self.word_embeddings[self.ind_to_classes[entity]] for entity in sorted_entities_1]
        embed_entities_1 = embed_entities_1 + [[0] * 300]  * (self.max_obj_len - len(embed_entities_1))
        embed_entities_1 = torch.Tensor(embed_entities_1).to(self.device)

        bbox_entities_1 = raw_object[image_one_key]['bbox'][:self.max_obj_len]
        bbox_entities_1 = bbox_entities_1 + [[0] * 4]  * (self.max_obj_len - len(bbox_entities_1))
        bbox_entities_1 = torch.Tensor(bbox_entities_1).to(self.device)
        roi_entities_1 = os.path.join(self.data_path, 'objects', self.mode, image_one['path'].split('/')[-1][:-4] + '.npy')

        image_two_key = os.path.join(self.data_path, image_two['path'])
        sorted_entities_2 = raw_object[image_two_key]['bbox_labels'][:self.max_obj_len]
        embed_entities_2 = [self.word_embeddings[self.ind_to_classes[entity]] for entity in sorted_entities_2]
        embed_entities_2 = embed_entities_2 + [[0] * 300]  * (self.max_obj_len - len(embed_entities_2))
        embed_entities_2 = torch.Tensor(embed_entities_2).to(self.device)

        bbox_entities_2 = raw_object[image_two_key]['bbox'][:self.max_obj_len]
        bbox_entities_2 = bbox_entities_2 + [[0] * 4]  * (self.max_obj_len - len(bbox_entities_2))
        bbox_entities_2 = torch.Tensor(bbox_entities_2).to(self.device)
        roi_entities_2 = os.path.join(self.data_path, 'objects', self.mode, image_two['path'].split('/')[-1][:-4] + '.npy')

        if len(sorted_entities_1) == 0 or len(sorted_entities_2) == 0:
            return []

        tensorized_examples_for_one_frame = list()
        for tail_dict in image_one['event'].values():
            if self.mode == 'training':
                positive_list = list()
                negative_list = list()
                for tmp_event_pair in tail_dict:
                    if tmp_event_pair[1] == 1:
                        positive_list.append(tmp_event_pair)
                    if tmp_event_pair[1] == 0:
                        negative_list.append(tmp_event_pair)
                random.shuffle(negative_list)
                negative_list = negative_list[:len(positive_list)]
                positive_list.extend(negative_list)
                candidate_list = positive_list
            elif self.mode == 'validation' or self.mode == 'testing':
                candidate_list = tail_dict

            for tmp_event_pair in candidate_list:
                event_1 = tmp_event_pair[0].split('$$')[0]
                event_2 = tmp_event_pair[0].split('$$')[1]
                label = torch.tensor(tmp_event_pair[1]).to(self.device)
                if len(event_1.split(' ')) > 1 and len(event_2.split(' ')) > 1:
                    bert_tokenized_event = self.tokenizer.encode('[CLS] ' + event_1 + ' . [SEP] ' + event_2 + ' . [SEP]')[:self.max_seq_len]
                    
                    padded_event = bert_tokenized_event + [0] * (self.max_seq_len - len(bert_tokenized_event))
                    padded_event = torch.tensor(padded_event).to(self.device)

                    tensorized_examples_for_one_frame.append({'bert_event': padded_event,
                                                                'embed_entities_1': embed_entities_1,
                                                                'embed_entities_2': embed_entities_2,
                                                                'bbox_entities_1': bbox_entities_1,
                                                                'bbox_entities_2': bbox_entities_2,
                                                                'roi_entities_1': roi_entities_1,
                                                                'roi_entities_2': roi_entities_2,
                                                                'label': label,
                                                                'category': category,
                                                                'video_id': video_id,
                                                                'image_id': image_id,
                                                                'event_key': event_1
                                                                })
        return tensorized_examples_for_one_frame

    def reload_train_data(self):
        assert self.mode == 'training'
        self.data_set = self.tensorize_example(self.raw_dataset, self.raw_object)
        print('successfully reloaded %d examples for training data' % len(self.data_set))

    def __getitem__(self, index):
        data = self.data_set[index]

        padded_event = data['bert_event']
        mask_event = (padded_event != 0).bool()

        embed_entities_1 = data['embed_entities_1']
        embed_entities_2 = data['embed_entities_2']
        bbox_entities_1 = data['bbox_entities_1']
        bbox_entities_2 = data['bbox_entities_1']
        roi_entities_1 = numpy.load(data['roi_entities_1'])[:self.max_obj_len].tolist()
        roi_entities_1 = roi_entities_1 + [[0] * 4096]  * (self.max_obj_len - len(roi_entities_1))
        roi_entities_1 = torch.Tensor(roi_entities_1).to(self.device)
        roi_entities_2 = numpy.load(data['roi_entities_2'])[:self.max_obj_len].tolist()
        roi_entities_2 = roi_entities_2 + [[0] * 4096]  * (self.max_obj_len - len(roi_entities_2))
        roi_entities_2 = torch.Tensor(roi_entities_2).to(self.device)
        mask_obj_1 = ((embed_entities_1 != 0)[:, 0]).bool()
        mask_obj_2 = ((embed_entities_2 != 0)[:, 0]).bool()

        label = data['label']

        if self.mode == 'validation' or self.mode == 'testing':
            return padded_event, mask_event, \
                embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1, \
                embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2, label, \
                data['category'], data['video_id'], data['image_id'], data['event_key']

        return padded_event, mask_event, \
            embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1, \
            embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2, label

    def __len__(self):
        return len(self.data_set)


""" 模型 """
class CausalityReasoning(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.embedding_size = 300
        self.hidden_dim = 512

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.roi_embed = nn.Linear(4096, 1024)
        self.lin_obj = nn.Linear(128 + 1024 + 300, self.hidden_dim)
        self.lin_event = nn.Linear(768, self.hidden_dim)

        self.encoder_layers_obj = torch.nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_obj = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layers_obj, num_layers=2)

        self.encoder_layers_rel = torch.nn.TransformerEncoderLayer(d_model=self.hidden_dim * 2, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer_rel = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layers_rel, num_layers=2)

        self.attention_to_entity = torch.nn.Linear(self.hidden_dim * 3, 1)

        self.vision_last_layer = nn.Sequential(*[
            nn.Dropout(0.3), nn.Linear(self.hidden_dim * 3, 2),
        ])
        self.lang_last_layer = nn.Sequential(*[
            nn.Linear(768, self.hidden_dim), nn.Dropout(0.3), nn.Linear(self.hidden_dim, 2),
        ])

    def cross_attention(self, eventuality_representation, entity_representations, mask_obj):
        number_of_objects = entity_representations.size(1)

        # context mask
        mask_obj_fill = mask_obj.unsqueeze(-1).repeat(1, 1, entity_representations.size(2))
        entity_representations = entity_representations.masked_fill(mask=~mask_obj_fill, value=0)
        # event attn
        event_raw_representation = eventuality_representation # bz x 1 x 768
        event_raw_representation = event_raw_representation.repeat(1, number_of_objects, 1)  # bz x numobj x 768
        event_attention = self.attention_to_entity(
            torch.cat([event_raw_representation, entity_representations], dim=2))  # bz x numobj x d1+d2 → bz x numobj x 1
        # context rep
        context_representation = torch.sum(entity_representations * event_attention, dim=1, keepdim=True) / \
            torch.sum(mask_obj, dim=1, keepdim=True).unsqueeze(-1).float() # bz x 1 x 300

        return eventuality_representation.squeeze(1), context_representation.squeeze(1)   # bz x d

    def forward(self, event, mask, embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1,\
            embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2):

        box_embed_1 = self.bbox_embed(self.encode_box_info(bbox_entities_1))
        box_embed_2 = self.bbox_embed(self.encode_box_info(bbox_entities_2))

        roi_embed_1 = self.roi_embed(roi_entities_1)
        roi_embed_2 = self.roi_embed(roi_entities_2)

        obj_rep_1 = torch.cat((box_embed_1, roi_embed_1, embed_entities_1), -1)
        obj_rep_1 = self.lin_obj(obj_rep_1)
        obj_rep_2 = torch.cat((box_embed_2, roi_embed_2, embed_entities_2), -1)
        obj_rep_2 = self.lin_obj(obj_rep_2)

        obj_rep_1 = self.transformer_obj(src=obj_rep_1.permute(1,0,2), src_key_padding_mask=~mask_obj_1).permute(1,0,2) # S x N x E
        obj_rep_2 = self.transformer_obj(src=obj_rep_2.permute(1,0,2), src_key_padding_mask=~mask_obj_2).permute(1,0,2)

        entities_rel = list()
        mask_obj = list()
        for e_1, e_2, m_1, m_2 in zip(obj_rep_1, obj_rep_2, mask_obj_1, mask_obj_2):
            masked_entities_1 = e_1[m_1]
            masked_entities_2 = e_2[m_2]
            entity_rel = torch.cat([torch.cat((e_i, e_j), dim=-1).unsqueeze(0) \
                for e_i in masked_entities_1 for e_j in masked_entities_2], dim=0)
            entity_sim = torch.Tensor([F.cosine_similarity(e_i, e_j, dim=-1) \
                for e_i in masked_entities_1 for e_j in masked_entities_2]).to(entity_rel.device)
            _, idx = entity_sim.sort(descending=True)
            entity_rel = entity_rel[idx][:15]
            entity_pad = torch.zeros((15 - entity_rel.shape[0], e_1.shape[1] + e_2.shape[1])).to(entity_rel.device)
            entities_rel.append(torch.cat((entity_rel, entity_pad), dim=0).unsqueeze(0))
            mask_obj.append(entity_rel.shape[0] * [1] + (15 - entity_rel.shape[0]) * [0])
        entities_rel = torch.cat(entities_rel)
        mask_obj = torch.Tensor(mask_obj).bool().to(entities_rel.device)

        entities_rel = self.transformer_rel(src=entities_rel.permute(1,0,2), src_key_padding_mask=~mask_obj).permute(1,0,2)

        event_representation = self.bert(event, attention_mask=mask)  # bz x 50 x 768
        event_representation = event_representation[0][:, :1, :]    # bz x 1 x 768
        lang_logits = self.lang_last_layer(event_representation.clone().detach().squeeze(1))

        event_raw_representation = self.lin_event(event_representation)
        event_weighted_representation, event_context_representation = self.cross_attention(
            event_raw_representation, entities_rel, mask_obj)   # bz x d       

        overall_representation = torch.cat((event_weighted_representation, event_context_representation), dim=1)
        vision_logits = self.vision_last_layer(overall_representation)

        """ fusion """
        if not self.training:
            lang_score = lang_logits.softmax(dim=-1)[:, 1:]
            max_lang = max(lang_logits.softmax(dim=-1)[0, 0], lang_logits.softmax(dim=-1)[0, 1])
            vision_score = vision_logits.softmax(dim=-1)[:, 1:]
            max_vision = max(vision_logits.softmax(dim=-1)[0, 0], vision_logits.softmax(dim=-1)[0, 1])
            if max_vision > max_lang:
                beta = torch.exp(torch.sqrt(max_vision - max_lang))
            else:
                beta = torch.exp(-torch.sqrt(max_lang - max_vision))
            prediction = (1 + beta * beta) * lang_score * vision_score / (beta * beta * lang_score + vision_score)
            return prediction

        return lang_logits, vision_logits

    def encode_box_info(self, bbox_entities):
        boxes_info = []
        for boxes in bbox_entities:
            img_size = (600, 600)
            wid = img_size[0]
            hei = img_size[1]
            wh = boxes[:, 2:] - boxes[:, :2] + 1.0
            xy = boxes[:, :2] + 0.5 * wh
            w, h = wh.split([1,1], dim=-1)
            x, y = xy.split([1,1], dim=-1)
            x1, y1, x2, y2 = boxes.split([1,1,1,1], dim=-1)
            assert wid * hei != 0
            info = torch.cat([w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei,
                            w*h/(wid*hei)], dim=-1).view(-1, 9)
            boxes_info.append(info.unsqueeze(0))
        return torch.cat(boxes_info, dim=0)


""" 训练 """
def train(model, train_dataloader, optimizer, criterion, scheduler):
    total_loss = 0
    model.train()
    for i, (event, mask, embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1,\
        embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2, label) in enumerate(train_dataloader):
        final_prediction_lang, final_prediction_vis = model(event, mask, embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1, \
            embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2)
        loss_lang = criterion(final_prediction_lang, label).mean()
        loss_vis = criterion(final_prediction_vis, label).mean()
        loss = loss_lang + loss_vis
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % 100 == 0:
            print('iteration:', str(i+1), ', current loss:', str(total_loss / 100), ', lr:', optimizer.param_groups[-1]["lr"])
            total_loss = 0


""" 测试 """
def test(model, dataset):
    #initialize
    recall_list = [1, 5, 10]
    prediction_dict = dict()
    for video in range(100):
        prediction_dict['video_'+str(video)] = dict()
        for image in range(4):
            prediction_dict['video_'+str(video)]['image_'+str(image)] = dict()
    gt_positive_example = 0
    pred_positive_example = dict()
    for top_k in recall_list:
        pred_positive_example['top'+str(top_k)] = 0

    # evaluate
    model.eval()
    for i, (event, mask, embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1,\
        embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2,\
        label, category, video_id, image_id, event_key) in enumerate(dataset):
        final_prediction = model(event.unsqueeze(0), mask.unsqueeze(0), \
            embed_entities_1.unsqueeze(0), bbox_entities_1.unsqueeze(0), roi_entities_1.unsqueeze(0), mask_obj_1.unsqueeze(0),\
            embed_entities_2.unsqueeze(0), bbox_entities_2.unsqueeze(0), roi_entities_2.unsqueeze(0), mask_obj_2.unsqueeze(0))
        # softmax_prediction = F.softmax(final_prediction, dim=1)

        tmp_one_result = dict()
        tmp_one_result['True_score'] = final_prediction.data[0][0]
        tmp_one_result['label'] = label.item()

        if event_key not in prediction_dict['video_'+str(video_id)]['image_'+str(image_id)].keys():
            prediction_dict['video_'+str(video_id)]['image_'+str(image_id)][event_key] = list()
        prediction_dict['video_' + str(video_id)]['image_' + str(image_id)][event_key].append(tmp_one_result)

        if label == 1:
            gt_positive_example += 1

        if (i+1) % 1000 == 0:
            print('iteration:', i+1)

    for video in range(100):
        for image in range(4):
            current_predict = prediction_dict['video_'+str(video)]['image_'+str(image)]
            for key in current_predict:
                current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)), reverse=True)
                for top_k in recall_list:
                    tmp_top_predict = current_predict[key][:top_k]
                    for tmp_example in tmp_top_predict:
                        if tmp_example['label'] == 1:
                            pred_positive_example['top' + str(top_k)] += 1

    recall_result = dict()
    for top_k in recall_list:
        recall_result['Recall_' + str(top_k)] = pred_positive_example['top' + str(top_k)] / gt_positive_example

    return recall_result


""" 按类别测试 """
def test_by_type(model, dataset, recall_k):
    correct_count = dict()
    all_count = dict()
    correct_count['overall'] = 0
    all_count['overall'] = 0
    model.eval()

    # initialize
    prediction_dict = dict()
    for video in range(100):
        prediction_dict['video_' + str(video)] = dict()
        for image in range(4):
            prediction_dict['video_' + str(video)]['image_' + str(image)] = dict()

    for i, (event, mask, embed_entities_1, bbox_entities_1, roi_entities_1, mask_obj_1,\
        embed_entities_2, bbox_entities_2, roi_entities_2, mask_obj_2,\
        label, category, video_id, image_id, event_key) in enumerate(dataset):
        final_prediction = model(event.unsqueeze(0), mask.unsqueeze(0), \
            embed_entities_1.unsqueeze(0), bbox_entities_1.unsqueeze(0), roi_entities_1.unsqueeze(0), mask_obj_1.unsqueeze(0),\
            embed_entities_2.unsqueeze(0), bbox_entities_2.unsqueeze(0), roi_entities_2.unsqueeze(0), mask_obj_2.unsqueeze(0))

        # softmax_prediction = F.softmax(final_prediction, dim=1)

        if category not in correct_count:
            correct_count[category] = 0
        if category not in all_count:
            all_count[category] = 0

        tmp_one_result = dict()
        tmp_one_result['True_score'] = final_prediction.data[0][0]
        tmp_one_result['label'] = label.item()
        tmp_one_result['category'] = category

        if event_key not in prediction_dict['video_' + str(video_id)][
            'image_' + str(image_id)].keys():
            prediction_dict['video_' + str(video_id)]['image_' + str(image_id)][event_key] = list()
        prediction_dict['video_' + str(video_id)]['image_' + str(image_id)][event_key].append(tmp_one_result)

        if label.item() == 1:
            all_count['overall'] += 1
            all_count[category] += 1

    for video in range(100):
        for image in range(4):
            current_predict = prediction_dict['video_' + str(video)]['image_' + str(image)]
            for key in current_predict:
                current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)),
                                              reverse=True)
                # print(current_predict[key])
                tmp_top_predict = current_predict[key][:recall_k]
                for tmp_example in tmp_top_predict:
                    if tmp_example['label'] == 1:
                        correct_count[tmp_example['category']] += 1
                        correct_count['overall'] += 1

    accuracy_by_type = dict()
    for tmp_category in all_count:
        accuracy_by_type[tmp_category] = correct_count[tmp_category] / all_count[tmp_category]

    return accuracy_by_type


""" 读取glove嵌入向量 """
def load_embedding_dict(path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = numpy.zeros(300)
    embedding_dict = collections.defaultdict(lambda: default_embedding)
    if len(path) > 0:
        vocab_size = None
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word_end = line.find(" ")
                word = line[:word_end]
                embedding = numpy.fromstring(line[word_end + 1:], numpy.float32, sep=" ")
                assert len(embedding) == 300
                embedding_dict[word] = embedding
        if vocab_size is not None:
            assert vocab_size == len(embedding_dict)
        print("Done loading word embeddings.")
    return embedding_dict


""" 学习率warmup """
from bisect import bisect_right
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones=(11500, 20000),
        gamma=0.2,
        warmup_factor=0.1,
        warmup_iters=400,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


""" 固定随机种子 """
def fix_seed(seed=1357246891):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--data_path", default='/data/causal/data/', type=str, required=False,
                        help="path of dataset")
    parser.add_argument("--embed_path", default='/data/causal/glove/glove.6B.300d.txt', type=str, required=False,
                        help="path of glove")
    parser.add_argument("--max_obj_len", default=10, type=int, required=False,
                        help="maximum number of objects per image")
    parser.add_argument("--max_seq_len", default=100, type=int, required=False,
                        help="maximum number of words per event")
    parser.add_argument("--batch_size", default=8, type=int, required=False,
                        help="batchsize")
    parser.add_argument("--epoch", default=20, type=int, required=False,
                        help="training how many epochs")
    parser.add_argument("--val_per_epoch", default=10, type=int, required=False,
                        help="evaluate on validation dataset per epoch")
    parser.add_argument("--lr", default=0.0001, type=float, required=False,
                        help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, required=False,
                        help="learning momentum")
    parser.add_argument("--test_by_type", default=False, required=False,
                        help="Evaluate the model by types (i.e., Sports, Socializing, Household, Personal Care, Eating)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    fix_seed()

    # Use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device:', device)
    print('number of gpu:', torch.cuda.device_count())

    # Choose model
    word_embeddings = load_embedding_dict(args.embed_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    current_model = CausalityReasoning.from_pretrained('bert-base-uncased').to(device)
    current_model.load_state_dict(torch.load('/data/CausalityReasoning.pytorch/checkpoints/pretrained_atomic/pretrained_atomic.pth'))

    # Load data
    train_data = CausalDataset(args, word_embeddings, tokenizer, device, 'training')
    valid_data = CausalDataset(args, word_embeddings, tokenizer, device, 'validation')
    test_data = CausalDataset(args, word_embeddings, tokenizer, device, 'testing')

    # training tools
    optimizer = torch.optim.SGD(current_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = WarmupMultiStepLR(optimizer=optimizer)
    criterion = nn.CrossEntropyLoss()

    # for test
    # current_model.load_state_dict(torch.load('/data/CausalityReasoning.pytorch/checkpoints/model_CR/CR_model.pth'))
    # test_performance = test(current_model, test_data)
    # print('Test accuracy:', test_performance)
    # assert 1==2

    # accuracy_by_type = dict()
    # for top_k in [1, 5, 10]:
    #     accuracy_by_type['Recall_'+str(top_k)] = test_by_type(current_model, test_data, top_k)
    # print('Test accuracy (by type):', accuracy_by_type)
    # assert 1==2

    # train
    for i in range(args.epoch):
        print('Epoch:', i+1)
        print('Training...')
        train_data.reload_train_data()
        train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        train(current_model, train_dataloader, optimizer, criterion, scheduler)

        torch.save(current_model.state_dict(), './checkpoints/building/' + args.model + '_' + str(i+1) + '.pth')
        if (i + 1) % args.val_per_epoch == 0 and (i + 1) != args.epoch:
            print('Validating...')
            dev_performance = test(current_model, valid_data)
            print('Dev accuracy:', dev_performance)
            
    print('End.')


if __name__=='__main__':
    main()