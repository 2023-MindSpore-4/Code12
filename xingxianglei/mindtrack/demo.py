import argparse
import ast
import os
import os.path as osp
import time
import cv2
# import torch

from loguru import logger

from mindyolo import create_model, non_max_suppression, xyxy2xywh, scale_coords
from mindyolo.utils.config import load_config, Config
from tracking_utils.timer import Timer
from tracker.byte_tracker import BYTETracker
import argparse
import ast
import math
import os
import sys
import time
import cv2
import numpy as np
import yaml
import mindspore as ms
from mindspore import Tensor, context, nn
from datetime import datetime

from visualize import plot_tracking

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"])
    parser.add_argument("--device_target", type=str, default="GPU", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--weight", type=str, default="yolox-x.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")

    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_infer", help="save dir")

    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    return parser

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r
class Predictor(object):
    def __init__(
            self,
            model,
            args,
    ):
        self.model = model
        self.conf_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.conf_free = args.conf_free
        self.nms_time_limit = args.nms_time_limit
        self.img_size = args.img_size
        self.stride = max(max(args.network.stride), 32)
        self.num_class = args.data.nc
        self.test_size=(800, 1440)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img_info["ratio"] = 1

        timer.tic()
        outputs = self.detect(img)
        return outputs, img_info

    def detect(self, img):

        img,r=preproc(img, self.test_size, self.rgb_means, self.std)
        img = Tensor(img, ms.float32).unsqueeze(0)
        out = self.model(img)  # inference and training outputs
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out.asnumpy()

        out = non_max_suppression(
            out,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            conf_free=self.conf_free,
            multi_label=True,
            time_limit=self.nms_time_limit,
        )
        return out


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def parse_args(parser):
    parser_config = argparse.ArgumentParser(description="Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="yolox-x.yaml", help="YAML config file specifying default arguments."
    )

    args_config, remaining = parser_config.parse_known_args()

    # Do we have a config file to parse?
    if args_config.config:
        cfg, _, _ = load_config(args_config.config)
        cfg = Config(cfg)
        parser.set_defaults(**cfg)
        parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return Config(vars(args))


def main():
    parser = get_parser_infer()
    args = parse_args(parser)
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    current_time = time.localtime()
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)
    tracker = BYTETracker(args, frame_rate=args.fps)
    # cap = cv2.VideoCapture('/media/ubuntu/5E78A6A178A67803/mindtrack/ByteTrack/videos/palace.mp4')
    files = get_image_list('images')
    files.sort()
    timer = Timer()
    predictor = Predictor(network, args)
    results = []
    vis_folder = osp.join('output', "track_vis")
    os.makedirs(vis_folder, exist_ok=True)
    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0][:,:5], [img_info['height'], img_info['width']], (800, 1440))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


if __name__ == "__main__":
    main()
