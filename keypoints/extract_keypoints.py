# Script to download YouTube videos
# and run a trained Detectron-model on each frame.
#
# Author: Bernhard Bermeitinger, bernhard.bermeitinger@uni-passau.de
from __future__ import print_function, with_statement
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import argparse
from joblib import Parallel, delayed
import json
import logging
import cv2 as cv
import os
import sys
from time import time
import pickle
import matplotlib.pyplot as plt
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s :: %(name)-20s:  %(message).1000s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

import cv2
import numpy as np
import pafy
import pycocotools.mask as mask_utils

import detectron.core.test_engine as infer_engine
import utils.c2 as c2_utils
import utils.vis as vis_utils
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg, cfg, merge_cfg_from_file
from datasets.json_dataset import JsonDataset

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

c2_utils.import_detectron_ops()
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
dpi=200
kp_thresh=2
thresh=0.9
THRESHOLD = 0.75
def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='jpg', out_when_no_box=False):
    """Visual debugging of detections."""
    #print(max(boxes[:, 4]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        print(boxes)
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh) and not out_when_no_box:
        return

    dataset_keypoints, _ = get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    if boxes is None:
        sorted_inds = [] # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        print('Sorted inds:',sorted_inds)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            print('Diese box wird ignoriert, da der score zu schlechte ist:', score)
            continue

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            print('kps', kps)
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')
def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    print(cls_boxes)
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map
def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="folder with your images",
        dest="videos",
        required=True
    )

    parser.add_argument(
        "--output",
        help="Output file",
        dest="output",
        default="output.json"
    )

    parser.add_argument(
        "--overwrite",
        help="Overwrite the output file",
        dest="overwrite",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--download-path",
        help="Download path where videos are stored",
        dest="download_path",
        default="videos"
    )

    parser.add_argument(
        "--redownload",
        help="Forces the download of the video files.",
        dest="redownload",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--model-cfg",
        help="File for the model configuration",
        dest="model_config",
        required=True
    )

    parser.add_argument(
        "--write-videos",
        help="Folder to write inferred videos",
        dest="write_videos",
        default=None
    )
    parser.add_argument(
        "--save-res",
        help="Save Result-Visalization",
        dest="saveres",
        default=None
    )
    return parser.parse_args()


def _valid_args(args):
    if os.path.isfile(args.videos):
        log.error("The given videos file is not a file: %s", args.videos)
        return False

    if os.path.isfile(args.output):
        log.warning("The given output file already exists: %s", args.output)
        if args.overwrite:
            log.warning("The output file will be overwritten.")
        else:
            log.warning("The output file exists, you must give --overwrite to overwrite the file.")

    if args.write_videos is not None:
        if os.path.isdir(args.write_videos):
            log.warning("The path were videos are written to already exists: %s", args.write_videos)

    if os.path.isdir(args.download_path):
        log.info("The download path already exists: %s", args.download_path)

    if not os.path.isfile(args.model_config):
        log.error("The given model config file is not a file: %s", args.model_config)
        return False

    return True


def _print_download_progress(total, *stats):
    already_downloaded, percentage, rate, _ = stats

    if already_downloaded == total or percentage >= 1.0:
        log.info("Download finished: {} MB with rate {:.0f} KB/s".format(total / 1024 / 1024, rate))


class DownloadException(Exception):
    def __init__(self, message):
        super(DownloadException, self).__init__(message)


def _download_video(link, target_folder, redownload):
    try:
        log.debug("Will attempt to download '%s' to '%s' (redownload: %s)", link, target_folder, redownload)

        youtube = pafy.new(link)
        video = youtube.getbestvideo(preftype='mp4')  # we ignore audio at the moment
        video_id = youtube.videoid.encode('utf8')
        file_name = "{video_id}.{extension}".format(
            video_id=video_id, extension=video.extension.encode('utf8'))

        target_file = os.path.join(target_folder, file_name)
        if os.path.isfile(target_file):
            log.warning("The video for the ID %s is already downloaded.", video_id)
            log.debug("The link is '%s' and the video_id '%s', downloaded at '%s'", link, video_id, target_file)
            if not redownload:
                log.info("Will not redownload file.")
                return video_id, target_file
            else:
                os.remove(target_file)
                log.debug("File deleted: %s", target_file)

        log.info("Start downloading video: '%s' to '%s' ...", link, target_file)
        try:
            downloaded_file = video.download(filepath=target_file, progress="KB", callback=_print_download_progress)
        except pafy.util.HTTPError as e:
            raise DownloadException("HTTPError occurred")

        if downloaded_file != target_file:
            log.error("The downloaded file is not at the location where it should be.")
            raise DownloadException("The downloaded file is not at the location where it should be.")

        return video_id, target_file
    except:
        print('Fehler bei:',link)
        return None, None


def _parallel_download(link, download_path, redownload):
    try:
        return _download_video(link, download_path, redownload)
    except DownloadException as e:
        log.warning("Could not download video at link: %s, because %s", link, e)
        return None, None


def _download_videos(video_text_file, download_path, redownload):
    log.debug("Reading text file: %s", video_text_file)
    if os.path.isdir(download_path):
        log.warning("Download path already exists: %s", download_path)
    else:
        log.info("Will create download path at: %s", download_path)
        os.makedirs(download_path, mode=0o700)
    links = []
    with open(video_text_file, 'r') as fi:
        for line in fi:
            line = line.strip()
            if not line.startswith("http"):
                log.warning("The line '%s' is not a valid link.", line)
                continue
            links.append(line)
    downloaded_videos = Parallel(n_jobs=12)(delayed(_parallel_download)(link, download_path, redownload) for link in links)
    try:
        error_videos = [v for v in downloaded_videos if any(i is None for i in v)]
    except:
        error_videos=[]
    print(error_videos)
    log.warning("The download of %s videos failed.", len(error_videos))
    log.info("Downloaded %s videos", len(downloaded_videos))

    return [v for v in downloaded_videos if v not in error_videos]


class Detectron(object):
    def __init__(self, model_config):
        merge_cfg_from_file(model_config)
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg()
        print('TEST.WEIGHTS:',cfg.TEST.WEIGHTS)
        self.__model = infer_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS)
        self.__dataset = JsonDataset(cfg.TRAIN.DATASETS[0])

    def infer_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        with c2_utils.NamedCudaScope(0):
            boxes, segments, keypoints = infer_engine.im_detect_all(self.__model, image, None)

        return boxes, segments, keypoints

    def draw_on_image(self, image, boxes, segments, keypoints, threshold=THRESHOLD):
        if isinstance(image, str):
            image = cv2.imread(image)

        result = vis_utils.vis_one_image_opencv(
            im=image,
            boxes=boxes,
            keypoints=keypoints,
            segms=segments,
            thresh=threshold,
            dataset=self.__dataset,
            show_class=False,
            show_box=True
        )

        return result

    @staticmethod
    def count_frames(video_file):
        video = cv2.VideoCapture(video_file)
        frame_count = 0
        while True:
            ret, _ = video.read()
            if not ret:
                break
            frame_count += 1
        
        video.release()
        
        return frame_count

    def infer_video(self, video_file):
        frame_count = 0
        
        video = cv2.VideoCapture(video_file)
        video_boxes = []
        video_segments = []
        video_dimensions = None
        total = 0
        diff=0
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                log.info("Video ended")
                break
            if video_dimensions is None:
                video_dimensions = frame.shape[1], frame.shape[0]

            if i % 10 == 0:
                t0 = time()
                log.debug("Inferring frame %s / %s", i + 1, frame_count)
                frame_boxes, frame_segments = self.infer_image(frame)
                t1 = time()
                diff += t1 - t0
                log.debug("Frame %s / %s inferred, took %s s", i + 1, frame_count, diff)
            video_boxes.append(frame_boxes)
            video_segments.append(frame_segments)

        video.release()
        log.info("Inference took %s min", float(60) / 60)

        return video_boxes, video_segments, video_dimensions

    def infer_img(self, frame):
	
        dimensions = frame.shape[1], frame.shape[0]
        frame_boxes, frame_segments, frame_keypoints = self.infer_image(frame)

        return frame_boxes, frame_segments, frame_keypoints, dimensions

    def get_class_name(self, index):
        if index >= len(self.__dataset.classes):
            log.error("class index higher than classes length: %s", index)
            log.debug("classes: %s", self.__dataset.classes)
        return self.__dataset.classes[index]


FRAME_RATE = 12.0


def _write_video(model, boxes, segments, dimensions, video_output_path, video_input_file):
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path, 0o755)

    video_output_file = "{}_INFERENCE.mp4".format(os.path.splitext(os.path.basename(video_input_file))[0])

    log.info("Will write inferred video to '%s'", video_output_file)

    video_in = cv2.VideoCapture(video_input_file)
    video_out = cv2.VideoWriter(
        os.path.join('out',video_output_path, video_output_file),
        cv2.VideoWriter.fourcc(*'mp4v'),
        FRAME_RATE,
        dimensions
    )

    for frame_boxes, frame_segments in zip(boxes, segments):
        ret, frame = video_in.read()
        if not ret:
            break

        img = model.draw_on_image(frame, frame_boxes, frame_segments)
        img = cv2.resize(img, dimensions, cv2.INTER_NEAREST)
        video_out.write(img)

    video_in.release()
    video_out.release()

def _write_img(model, boxes, segments, keypoints, dimensions, output_file, frame):

    log.info("Will write inferred video to '%s'", output_file)

    img = model.draw_on_image(frame, boxes, segments, keypoints)
    img = cv2.resize(img, dimensions, cv2.INTER_NEAREST)
    cv2.imwrite(output_file, img)

def _main(video_text_file, download_path, output, redownload, model_config, write_videos, save_res):
    outputname=video_text_file.split('/')[-1]+'_'+model_config.split('/')[-1].split('.')[0]
    print(model_config)
    detectron = Detectron(model_config)
    if not os.path.exists(outputname+'_out/'):
        os.makedirs(outputname+'_out/', 0o755)
    if not os.path.exists(outputname + '_out/'):
        os.makedirs(outputname + '_out/cropped/', 0o755)
    imgs_information = {}

    #if os.path.exists(output):
    #    with open(output, 'r') as fi:
    #        videos_information = json.load(fi)
    imgs_kp={}
    dfkps = pd.DataFrame(columns=['Bild', 'Achse', 'Person', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    for img in os.listdir(video_text_file):
        dataset_keypoints, _ = get_keypoints()
        kp_lines = kp_connections(dataset_keypoints)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
        log.info(img)
        frame = cv2.imread(video_text_file+"/"+img, 1)
        print('Bilder:',img)
        frame_boxes, frame_segments, frame_keypoints, frame_dimensions = detectron.infer_img(frame)
        frame_area = frame_dimensions[0] * frame_dimensions[1]
        frame_information = []
        log.debug("frame_boxes: %s", frame_boxes)
        log.debug("frame_segments: %s", frame_segments)
        log.debug("frame_keypoints: %s", frame_keypoints)

        if isinstance(frame_boxes, list):
            frame_boxes, frame_segments, frame_keypoints, _ = vis_utils.convert_from_cls_format(frame_boxes, frame_segments, frame_keypoints)

        if frame_boxes is not None and frame_boxes.shape[0] != 0:
            print('lenframeboxes', len(frame_boxes))
            sorted_inds = range(len(frame_boxes))

            # for i in sorted_inds:
            #     try:
            #         class_name = detectron.get_class_name(i)
            #     except IndexError as e:
            #         log.error("Cannot get_class_name: %s", e)
            #         log.debug("sorted_inds: %s", sorted_inds)
            #         log.debug("frame_boxes: %s", frame_boxes)
            #         log.debug("frame_segments: %s", frame_segments)
            #         log.debug("score: %s", score)
            #
            #     score = float(frame_boxes[i, -1])
            #
            #     if score < THRESHOLD or class_name == '__background__':
            #         continue
            #     log.debug("Frame %s: found class '%s' with score '%s'", img, class_name, score)
            #
            #     frame_information.append({
            #         'label': class_name,
            #         'total_area': str(frame_keypoints),
            #         'percentage': 0,
            #         'score': score,
            #         'bbox': frame_boxes[i, :4].astype(np.int).tolist()
            #     })
            #     #print('schreibe in Pickle Bild:', img,frame_segments)
            #imgs_kp[str(img)]= {'kp':frame_keypoints, 'score':score,'bbox':frame_boxes,'segm':frame_segments}
            keypoints=frame_keypoints
            for i in range(len(frame_keypoints)):

                if (keypoints is not None and len(keypoints) > i) and (frame_boxes[i, -1]>thresh):
                    print('Boundingbox',str(img),frame_boxes[i, 1], ':', frame_boxes[i, 3], ',', frame_boxes[i, 0], ':',frame_boxes[i, 2],frame.shape[0],frame.shape[1])
                    framecropped = frame[int(frame_boxes[i, 1]):int(frame_boxes[i, 3]),
                            int(frame_boxes[i, 0]):int(frame_boxes[i, 2])]
                    #cv2.imwrite('messigray.png', framecropped)
                    framex=int(frame_boxes[i, 0])
                    framey=int(frame_boxes[i, 1])
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(float(framecropped.shape[1]) / dpi, float(framecropped.shape[0]) / dpi)
                    print('framecropped:',framecropped.shape,float(framecropped.shape[1]) / dpi, float(framecropped.shape[0]) / dpi)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.axis('off')
                    fig.add_axes(ax)
                    #fig.savefig('test_' + str(img) + '.jpg', dpi=dpi)
                    im2 = cv.cvtColor(framecropped, cv.COLOR_BGR2RGB)
                    ax.imshow(im2)
                    kps = keypoints[i]
                    #print('Kps x', kps[0])
                    #print('Kps y', kps[1])
                    # print('kps', kps)
                    ind = len(dfkps)
                    dfkps.set_value(ind, 'Achse', 'x')
                    dfkps.set_value(ind, 'Bild', img.split('/')[0])
                    dfkps.set_value(ind, 'Person', i)
                    dfkps.set_value(ind + 1, 'Achse', 'y')
                    dfkps.set_value(ind + 1, 'Bild', img.split('/')[0])
                    dfkps.set_value(ind + 1, 'Person', i)
                    #fig.savefig('test_' + str(img) + '.jpg', dpi=dpi)
                    for z in range(len(kps[1])):
                        if 2 < kps[2][z]:
                            dfkps.set_value(ind, z, kps[0][z])
                            dfkps.set_value(ind + 1, z, kps[1][z])
                    if save_res == 'True':
                        #print(dfkps)
                        plt.autoscale(False)
                        for l in range(len(kp_lines)):
                            i1 = kp_lines[l][0]
                            i2 = kp_lines[l][1]
                            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                                x = [kps[0, i1]-framex, kps[0, i2]-framex]
                                y = [kps[1, i1]-framey, kps[1, i2]-framey]
                                print('keypoint',l,':',x,y)
                                line = plt.plot(x, y)
                                plt.setp(line, color=colors[l], linewidth=3.0, alpha=0.7)
                            if kps[2, i1] > kp_thresh:
                                plt.plot(
                                    kps[0, i1]-framex, kps[1, i1]-framey, '.', color=colors[l],
                                    markersize=3.0, alpha=0.7)

                            if kps[2, i2] > kp_thresh:
                                plt.plot(
                                    kps[0, i2]-framex, kps[1, i2]-framey, '.', color=colors[l],
                                    markersize=3.0, alpha=0.7)
                            #fig.savefig('test_'+str(img)+'_'+str(l)+'.jpg', dpi=dpi)

                            # add mid shoulder / mid hip for better visualization
                        mid_shoulder = (
                                               kps[:2, dataset_keypoints.index('right_shoulder')] +
                                               kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0

                        sc_mid_shoulder = np.minimum(
                            kps[2, dataset_keypoints.index('right_shoulder')],
                            kps[2, dataset_keypoints.index('left_shoulder')])
                        mid_hip = (
                                          kps[:2, dataset_keypoints.index('right_hip')] +
                                          kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
                        sc_mid_hip = np.minimum(
                            kps[2, dataset_keypoints.index('right_hip')],
                            kps[2, dataset_keypoints.index('left_hip')])
                        if (sc_mid_shoulder > kp_thresh and
                                kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                            x = [mid_shoulder[0]-framex, kps[0, dataset_keypoints.index('nose')]-framex]
                            y = [mid_shoulder[1]-framey, kps[1, dataset_keypoints.index('nose')]-framey]
                            line = plt.plot(x, y)
                            print(x,y)
                            plt.setp(
                                line, color=colors[len(kp_lines)], linewidth=3.0, alpha=0.7)
                        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                            x = [mid_shoulder[0]-framex, mid_hip[0]-framex]
                            y = [mid_shoulder[1]-framey, mid_hip[1]-framey]
                            print(x,y)
                            line = plt.plot(x, y)
                            plt.setp(
                                line, color=colors[len(kp_lines) + 1], linewidth=3.0,
                                alpha=0.7)

                        output_name = os.path.basename(img) + str(i) + '_kp.jpg'
                        size = fig.get_size_inches() * fig.dpi
                        print(size)
                        fig.savefig(os.path.join(outputname+'_out', '{}'.format(output_name)), dpi=dpi)
                        plt.close('all')
                    #dfkps.to_csv(outputname + '_dfkps.csv', sep='\t')
                    dfkps.to_pickle(outputname + '_dfkps.p')


        else:
            log.debug("Found nothing in picture %s", img)

        imgs_information[img] = frame_information

        log.info("Write intermediate file")
        with open(output, 'w') as fo:
            json.dump(imgs_information, fo, indent=2)
        pickle.dump(imgs_kp, open(outputname + "_kps.p", "wb"))
        #_write_img(detectron, frame_boxes, frame_segments, frame_keypoints, frame_dimensions, os.getcwd()+'/'+outputname+'_out/'+img.split('.')[0]+'_kp.jpg', frame)



if __name__ == '__main__':
    args = _parse_args()
    log.info("Started with arguments: %s", args)

    if not _valid_args(args):
        log.error("At least one argument is invalid")
        sys.exit(1)

    _main(video_text_file=args.videos,
          download_path=args.download_path,
          redownload=args.redownload,
          output=args.output,
          model_config=args.model_config,
          write_videos=args.write_videos,
          save_res=args.saveres)
