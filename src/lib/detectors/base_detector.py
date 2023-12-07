from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

    def pre_process(self, RGB_image, IR_image, scale, meta=None):
        RGB_height, RGB_width = RGB_image.shape[0:2]
        IR_height, IR_width = IR_image.shape[0:2]
        RGB_new_height = int(RGB_height * scale)
        RGB_new_width = int(RGB_width * scale)
        IR_new_height = int(IR_height * scale)
        IR_new_width = int(IR_width * scale)

        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array(
                [RGB_new_width / 2., RGB_new_height / 2.], dtype=np.float32)
            s = max(RGB_height, RGB_width) * 1.0
        else:
            inp_height = (RGB_new_height | self.opt.pad) + 1
            inp_width = (RGB_new_width | self.opt.pad) + 1
            c = np.array(
                [RGB_new_width // 2, RGB_new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        RGB_resized_image = cv2.resize(
            RGB_image, (RGB_new_width, RGB_new_height))
        IR_resized_image = cv2.resize(IR_image, (IR_new_width, IR_new_height))
        RGB_inp_image = cv2.warpAffine(
            RGB_resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        IR_inp_image = cv2.warpAffine(
            IR_resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        RGB_inp_image = ((RGB_inp_image / 255. - self.mean) /
                         self.std).astype(np.float32)
        IR_inp_image = ((IR_inp_image / 255. - self.mean) /
                        self.std).astype(np.float32)

        RGB_images = RGB_inp_image.transpose(2, 0, 1).reshape(
            1, 3, inp_height, inp_width)
        IR_images = IR_inp_image.transpose(2, 0, 1).reshape(
            1, 3, inp_height, inp_width)
        # if self.opt.flip_test:
        #     images=np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        RGB_images = torch.from_numpy(RGB_images)
        IR_images = torch.from_numpy(IR_images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return RGB_images, IR_images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                rgb_images = pre_processed_images['RGB_images'][scale][0]
                ir_images = pre_processed_images['IR_images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            rgb_images = rgb_images.to(self.opt.device)
            ir_images = ir_images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(
                rgb_images, ir_images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                self.debug(debugger, rgb_images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
