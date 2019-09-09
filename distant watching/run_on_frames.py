# Script to download YouTube videos
# and run a trained Detectron-model on each frame.
#
# Author: Bernhard Bermeitinger, bernhard.bermeitinger@uni-passau.de

from __future__ import print_function, with_statement

import argparse
from joblib import Parallel, delayed
import json
import logging
import os
import sys
from detectron.utils.colormap import colormap
from time import time
import pycocotools.mask as mask_util
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s :: %(name)-20s:  %(message).1000s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_WHITE = (255, 255, 255)
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
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

THRESHOLD = 0.75


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_folder",
        help="Folder in which folders of video frames are stored in.",
        dest="video_folder",
        required=True
    )

    parser.add_argument(
        "--output",
        help="Output file",
        dest="output",
        default="output.json"
    )
    parser.add_argument(
        "--outputdir",
        help="Output directory",
        dest="outputdir",
        default="frames/Bandera/frames"
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

    return parser.parse_args()


def _valid_args(args):
    if not os.path.isdir(args.video_folder):
        log.error("The given videos file is not a file: %s", args.video_folder)
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
        extraiertes_video=_download_video(link, download_path, redownload)
        print('Groesse der heruntergeladenen Datei:',len(extraiertes_video))

        return extraiertes_video
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
        print(cfg.TEST.WEIGHTS)
        self.__model = infer_engine.initialize_model_from_cfg(cfg.TEST.WEIGHTS)
        self.__dataset = JsonDataset(cfg.TRAIN.DATASETS[0])
        #print(self.__dataset.classes)

    def vis_one_image_opencv(
            self, im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
            show_box=True, dataset=None, show_class=True, class_str=None):
        """Constructs a numpy array with the detections visualized."""
        def vis_class(img, pos, class_str, font_scale=0.35):
            """Visualizes the class."""
            img = img.astype(np.uint8)
            x0, y0 = int(pos[0]), int(pos[1])
            # Compute text size.
            txt = class_str
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
            # Place text background.
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
            # Show text.
            txt_tl = x0, y0 - int(0.3 * txt_h)
            cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
            return img
        def vis_bbox(img, bbox, thick=1):
            """Visualizes a bounding box."""
            img = img.astype(np.uint8)
            (x0, y0, w, h) = bbox
            x1, y1 = int(x0 + w), int(y0 + h)
            x0, y0 = int(x0), int(y0)
            cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
            return img
        def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
            """Visualizes a single binary mask."""

            img = img.astype(np.float32)
            idx = np.nonzero(mask)

            img[idx[0], idx[1], :] *= 1.0 - alpha
            img[idx[0], idx[1], :] += alpha * col

            if show_border:
                #test=cv2.findContours(
                #    mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                _, contours, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                _WHITE = (255, 255, 255)
                _GRAY = (218, 227, 218)
                _GREEN = (18, 127, 15)
                cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

            return img.astype(np.uint8)

        if segms is not None and len(segms) > 0:
            masks = mask_util.decode(segms)
            color_list = colormap()
            mask_color_id = 0

            # show box (off by default)
            if show_box:
                im = vis_bbox(
                    im, (boxes[0], boxes[1], boxes[2] - boxes[0], boxes[3] - boxes[1]))

            # show class (off by default)
            if show_class:
                im = vis_class(im, (boxes[0], boxes[1] - 2), class_str)

            # show mask
            if segms is not None:
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1
                im = vis_mask(im, masks, color_mask)

            # show keypoints
            if keypoints is not None and len(keypoints) > i:
                im = vis_keypoints(im, keypoints[i], kp_thresh)

        return im
    def infer_image(self, image):
        #print(len(image))
        if isinstance(image, str):
            image = cv2.imread(image)
        with c2_utils.NamedCudaScope(0):
            boxes, segments, _ = infer_engine.im_detect_all(self.__model, image, None)

        return boxes, segments

    def draw_on_image(self, image, boxes, segments, class_names, threshold=THRESHOLD):
        if isinstance(image, str):
            image = cv2.imread(image)


        result = self.vis_one_image_opencv(
            self,
            im=image,
            boxes=boxes,
            segms=segments,
            thresh=threshold,
            dataset=self.__dataset,
            show_class=True,
            show_box=True,
            class_str=class_names
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

    def infer_video(self, video_folder,frame,frame_count,diff):


        t0 = time()
        print("Inferring frame",  frame_count)
        frame_boxes, frame_segments = self.infer_image(frame)
        t1 = time()
        diff += t1 - t0
        print("Frame ",  frame_count,'of video',video_folder,'inferred, took ', diff)
        return frame_boxes, frame_segments


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
        os.path.join(video_output_path, video_output_file),
        cv2.VideoWriter.fourcc(*'mp4v'),
        FRAME_RATE,
        dimensions
    )

    for frame_boxes, frame_segments in zip(boxes, segments):
        ret, frame = video_in.read()
        if not ret:
            break

        img = model.vis_one_image_opencv(
            im=image,
            boxes=boxes,
            segms=segments,
            thresh=model.threshold,
            dataset=model.__dataset,
            show_class=True,
            show_box=True,
            classname=class_names
        )
        img = cv2.resize(img, dimensions, cv2.INTER_NEAREST)
        video_out.write(img)

    video_in.release()
    video_out.release()


def _main(video_folder, download_path, output, redownload, model_config, write_videos, outputdir):
    if os.path.exists(output):
        with open(output, 'r') as fi:
            videos_information = json.load(fi)
    else:
        videos_information = {}
    if os.path.isdir(video_folder):
        detectron = Detectron(model_config)

        for file in os.listdir(video_folder):
            print('Bearbeite Video:',file)
            if os.path.isdir(os.path.join(video_folder,file)) and (file!='a5arrD39XjY.mp4') and (file not in videos_information):
                video_dimensions=None
                video_boxes=[]
                video_segments=[]
                video_information=[]
                frame_count=0
                diff = 0
                for filename in sorted(os.listdir(os.path.join(video_folder,file))):
                    if filename.endswith('jpg'):
                        found=False
                        class_names = []
                        #print(os.path.join(video_folder, file,filename))
                        frame=cv2.imread(os.path.join(video_folder, file,filename))
                        if video_dimensions is None:
                            video_dimensions = frame.shape[1], frame.shape[0]

                        frame_boxes, frame_segments = detectron.infer_image(frame)
                        frame_information = []
                        print('Nach infer:', type(frame_boxes),type(frame_segments))
                        if isinstance(frame_boxes, list):
                            frame_boxes, frame_segments, _, classes = vis_utils.convert_from_cls_format(frame_boxes,frame_segments, None)
                        print('Nach convert:', type(frame_boxes),type(frame_segments), classes)


                        if frame_boxes is not None and frame_boxes.shape[0] != 0:
                            video_area = video_dimensions[0] * video_dimensions[1]
                            box_areas = (frame_boxes[:, 2] - frame_boxes[:, 0]) * (frame_boxes[:, 3] - frame_boxes[:, 1])

                            sorted_inds = np.argsort(-box_areas)
                            print(box_areas, sorted_inds)

                            for i in sorted_inds:
                                try:
                                    class_name = detectron.get_class_name(classes[i])
                                    if class_name != '__background__':
                                        class_names.append(class_name)
                                except IndexError as e:
                                    log.error("Cannot get_class_name: %s", e)
                                    log.debug("sorted_inds: %s", sorted_inds)
                                    log.debug("box_areas: %s", box_areas)
                                    log.debug("frame_boxes: %s", frame_boxes)
                                    log.debug("frame_segments: %s", frame_segments)

                                score = float(frame_boxes[i, -1])
                                if not(score < THRESHOLD or class_name == '__background__'):
                                    found=True
                                    log.debug("Frame %s: found class '%s' with score '%s'", frame_count, class_name, score)

                                    segment_area = int(mask_utils.area(frame_segments[i]))
                                    frame_information.append({
                                        'label': class_name,
                                        'total_area': segment_area,
                                        'percentage': float(segment_area) / float(video_area),
                                        'score': score,
                                        'bbox': frame_boxes[i, :4].astype(np.int).tolist()
                                    })

                                    frame = detectron.vis_one_image_opencv(im=frame, boxes=frame_boxes[i], segms=frame_segments[i], class_str=class_name)

                        else:
                            log.debug("Found nothing in frame %s", frame_count)
                        if found:
                            img = cv2.resize(frame, video_dimensions, cv2.INTER_NEAREST)
                            if not os.path.exists(os.path.join(outputdir, file)):
                                os.makedirs(os.path.join(outputdir, file), 0o755)
                            cv2.imwrite(os.path.join(outputdir, file, filename), img)
                            with open(os.path.join(outputdir, file, (filename.split('.')[0]+'.json')), 'w') as fo:
                                json.dump(frame_information, fo, indent=2)

                        video_information.append(frame_information)
                        frame_count+=1

                print('video',file,'bearbeitet. Ergebisse:',video_dimensions,video_information)
                log.info("Write intermediate file")
                videos_information[file] = video_information
                with open(output, 'w') as fo:
                    json.dump(videos_information, fo, indent=2)
    #videos = _download_videos(video_text_file=video_text_file, download_path=download_path, redownload=redownload)

    for idx, (video_id, video_file) in enumerate(videos, start=1):
        log.info("Video %s/%s: Start inference for video_id '%s' on file '%s'", idx, len(videos), video_id, video_file)
        #if video_id in videos_information:
        #    log.warning("The video with the id '%s' is already analyzed, skipping", video_id)
        #    continue

        #
        # print(video_boxes, video_segments, video_dimensions)
        # video_area = video_dimensions[0] * video_dimensions[1]
        #
        # video_information = []
        #
        # for frame_count, (frame_boxes, frame_segments) in enumerate(zip(video_boxes, video_segments)):
        #     frame_information = []
        #
        #     if isinstance(frame_boxes, list):
        #         frame_boxes, frame_segments, _, _ = vis_utils.convert_from_cls_format(frame_boxes, frame_segments, None)
        #
        #     if frame_boxes is not None and frame_boxes.shape[0] != 0:
        #         box_areas = (frame_boxes[:, 2] - frame_boxes[:, 0]) * (frame_boxes[:, 3] - frame_boxes[:, 1])
        #
        #         sorted_inds = np.argsort(-box_areas)
        #
        #         for i in sorted_inds:
        #             try:
        #                 class_name = detectron.get_class_name(i)
        #             except IndexError as e:
        #                 log.error("Cannot get_class_name: %s", e)
        #                 log.debug("sorted_inds: %s", sorted_inds)
        #                 log.debug("box_areas: %s", box_areas)
        #                 log.debug("frame_boxes: %s", frame_boxes)
        #                 log.debug("frame_segments: %s", frame_segments)
        #
        #             score = float(frame_boxes[i, -1])
        #             if score < THRESHOLD or class_name == '__background__':
        #                 continue
        #             log.debug("Frame %s: found class '%s' with score '%s'", frame_count, class_name, score)
        #             segment_area = int(mask_utils.area(frame_segments[i]))
        #             frame_information.append({
        #                 'label': class_name,
        #                 'total_area': segment_area,
        #                 'percentage': float(segment_area) / float(video_area),
        #                 'score': score,
        #                 'bbox': frame_boxes[i, :4].astype(np.int).tolist()
        #             })
        #     else:
        #         log.debug("Found nothing in frame %s", frame_count)
        #
        #     video_information.append(frame_information)
        #
        # videos_information[video_id] = video_information
        #
        # log.info("Write intermediate file")
        # with open(output, 'w') as fo:
        #     json.dump(videos_information, fo, indent=2)
        #
        # if write_videos is not None:
        #     _write_video(detectron, video_boxes, video_segments, video_dimensions, write_videos, video_file)


if __name__ == '__main__':
    args = _parse_args()
    log.info("Started with arguments: %s", args)

    if not _valid_args(args):
        log.error("At least one argument is invalid")
        sys.exit(1)

    _main(video_folder=args.video_folder,
          download_path=args.download_path,
          output=args.output,
          redownload=args.redownload,
          model_config=args.model_config,
          write_videos=args.overwrite,
          outputdir=args.outputdir)
