from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        
        # 이미지 및 어노테이션 불러오기
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']

        # IR 및 RGB 이미지 경로 설정
        IR_img_path = os.path.join(self.IR_img_dir, file_name)
        RGB_img_path = os.path.join(self.RGB_img_dir, file_name)

        # COCO 어노테이션 불러오기
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        # IR 및 RGB 이미지 읽기
        IR_img = cv2.imread(IR_img_path)
        RGB_img = cv2.imread(RGB_img_path)

        # 이미지 크기 설정
        IR_height, IR_width = IR_img.shape[0], IR_img.shape[1]
        RGB_height, RGB_width = RGB_img.shape[0], RGB_img.shape[1]

        # 이미지 중심 설정
        IR_c = np.array(
            [IR_img.shape[1] / 2., IR_img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            IR_input_h = (IR_height | self.opt.pad) + 1
            IR_input_w = (IR_width | self.opt.pad) + 1
            IR_s = np.array([IR_input_w, IR_input_h], dtype=np.float32)
        else:
            IR_s = max(IR_img.shape[0], IR_img.shape[1]) * 1.0
            IR_input_h, IR_input_w = self.opt.input_h, self.opt.input_w

        RGB_c = np.array(
            [RGB_img.shape[1] / 2., RGB_img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            RGB_input_h = (RGB_height | self.opt.pad) + 1
            RGB_input_w = (RGB_width | self.opt.pad) + 1
            RGB_s = np.array([RGB_input_w, RGB_input_h], dtype=np.float32)
        else:
            RGB_s = max(RGB_img.shape[0], RGB_img.shape[1]) * 1.0
            RGB_input_h, RGB_input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                IR_s = IR_s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                IR_w_border = self._get_border(128, IR_img.shape[1])
                IR_h_border = self._get_border(128, IR_img.shape[0])
                IR_c[0] = np.random.randint(
                    low=IR_w_border, high=IR_img.shape[1] - IR_w_border)
                IR_c[1] = np.random.randint(
                    low=IR_h_border, high=IR_img.shape[0] - IR_h_border)

                RGB_s = RGB_s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                RGB_w_border = self._get_border(128, RGB_img.shape[1])
                RGB_h_border = self._get_border(128, RGB_img.shape[0])
                RGB_c[0] = np.random.randint(
                    low=RGB_w_border, high=RGB_img.shape[1] - RGB_w_border)
                RGB_c[1] = np.random.randint(
                    low=RGB_h_border, high=RGB_img.shape[0] - RGB_h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                IR_c[0] += IR_s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                IR_c[1] += IR_s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                IR_s = IR_s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

                RGB_c[0] += RGB_s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                RGB_c[1] += RGB_s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                RGB_s = RGB_s * np.clip(np.random.randn()
                                        * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                IR_img = IR_img[:, ::-1, :]
                IR_c[0] = IR_width - IR_c[0] - 1
                RGB_img = RGB_img[:, ::-1, :]
                RGB_c[0] = RGB_width - RGB_c[0] - 1

        IR_trans_input = get_affine_transform(
            IR_c, IR_s, 0, [IR_input_w, IR_input_h])
        IR_inp = cv2.warpAffine(
            IR_img, IR_trans_input, (IR_input_w, IR_input_h), flags=cv2.INTER_LINEAR)
        IR_inp = (IR_inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, IR_inp, self._eig_val, self._eig_vec)
        IR_inp = (IR_inp - self.mean) / self.std
        IR_inp = IR_inp.transpose(2, 0, 1)
        IR_output_h = IR_input_h // self.opt.down_ratio
        IR_output_w = IR_input_w // self.opt.down_ratio
        num_classes = self.num_classes

        IR_trans_output = get_affine_transform(
            IR_c, IR_s, 0, [IR_output_w, IR_output_h])

        RGB_trans_input = get_affine_transform(
            RGB_c, RGB_s, 0, [RGB_input_w, RGB_input_h])
        RGB_inp = cv2.warpAffine(
            RGB_img, RGB_trans_input, (RGB_input_w, RGB_input_h), flags=cv2.INTER_LINEAR)
        RGB_inp = (RGB_inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, RGB_inp, self._eig_val, self._eig_vec)
        RGB_inp = (RGB_inp - self.mean) / self.std
        RGB_inp = RGB_inp.transpose(2, 0, 1)
        RGB_output_h = RGB_input_h // self.opt.down_ratio
        RGB_output_w = RGB_input_w // self.opt.down_ratio
        num_classes = self.num_classes

        RGB_trans_output = get_affine_transform(
            RGB_c, RGB_s, 0, [RGB_output_w, RGB_output_h])

        hm = np.zeros((num_classes, IR_output_h, IR_output_w),
                      dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, IR_output_h, IR_output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros(
            (self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros(
            (self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = IR_width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], IR_trans_output)
            bbox[2:] = affine_transform(bbox[2:], IR_trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, IR_output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, IR_output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * IR_output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(
                        axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'ir_input': IR_inp, 'rgb_input': RGB_inp, 'hm': hm,
               'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh,
                       'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': IR_c, 's': IR_s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
