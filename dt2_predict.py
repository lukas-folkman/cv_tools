import copy
import os
import itertools
import numpy as np
from types import SimpleNamespace
import cv2
import torch
import torch.utils.data as torchdata
from torchvision.ops import nms
import traceback
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetMapper
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.common import MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

if os.path.exists('utils.py'):
    import utils
else:
    import cv_tools as utils


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--output_fn', required=True)
    parser.add_argument('--conf_thr', type=float, default=0)
    args = parser.parse_args()

    frames_fns = sorted([os.path.join(args.img_dir, fn) for fn in os.listdir(args.img_dir) if utils.is_image(fn)])
    dataset = dt2_predict(model=args.model, img_dir=args.img_dir, frames_fns=frames_fns, conf_thr=args.conf_thr)
    utils.fill_in_areas(dataset['annotations'], fill_in_ids=True, fill_in_iscrowd=True)
    utils.save_json(dataset, args.output_fn)


def dt2_predict(cfg, weights_fn, dataset, output_dir, output_fn=None, model_cat_names=None, predict_cat_names=None,
                threshold=None, NMS_threshold=None, detections_per_image=None, min_max_img_size=None, imgs_per_batch=1,
                track=None, track_buffer=None, new_track_thr=None, track_match_thr=None, track_high_thr=None, track_low_thr=None,
                save_pred_frames=False, vis_threshold=None, evaluate=False, device=None, warm_up=False,
                compress=True, eval_log_info=None, one_based_video_frames=False, quick_debug=False, strict=True):
    if save_pred_frames:
        # fail early import
        import bbox_visualizer
    assert track in [None, utils.BOT_SORT]
    if imgs_per_batch is None:
        imgs_per_batch = 1
    assert imgs_per_batch is not None and imgs_per_batch > 0
    assert not (model_cat_names is None and predict_cat_names is not None)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(dataset, str) and os.path.isdir(dataset):
        dataset = sorted(utils.read_images_from_dir(dir_name=dataset, basename_only=False))

    video_input = None
    if utils.is_iterable(dataset):
        if isinstance(dataset, tuple) and len(dataset) == 2:
            dataset_name = 'predictions'
            json_file, img_dir = dataset
            dataset = utils.read_json(json_file, only_imgs=True)
            predict_cat_names = utils.assure_consistent_cat_names(dataset, predict_cat_names=predict_cat_names)
            if predict_cat_names is None:
                predict_cat_names = model_cat_names
            register_coco_instances(
                name=dataset_name, json_file=json_file, image_root=img_dir,
                metadata=dict(thing_classes=predict_cat_names if predict_cat_names is not None else model_cat_names)
            )
        else:
            dataset_name = None
            if all([utils.is_image(x) for x in dataset]):
                video_input = False
            elif all([utils.is_video(x) for x in dataset]):
                if not all([utils.is_video(x, extensions='mp4') for x in dataset]):
                    raise ValueError(f'Only mp4 videos are supported: {dataset}')
                video_input = True
            else:
                raise ValueError(f'Incorrect inputs: {dataset}')
    else:
        assert isinstance(dataset, str), dataset
        dataset_name = dataset

    if isinstance(cfg, str):
        cfg = read_dt2_config(cfg, strict=strict)
    else:
        cfg = cfg.clone()
        cfg.defrost()

    cfg.MODEL.WEIGHTS = weights_fn
    if dataset_name is not None:
        cfg.DATASETS.TEST = (dataset_name,)
    device = utils.get_device(device, model='dt2')
    cfg.MODEL.DEVICE = device
    if threshold is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    if NMS_threshold is not None:
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_threshold
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = NMS_threshold
    if detections_per_image is not None:
        cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image
    if min_max_img_size is not None:
        cfg.INPUT.MIN_SIZE_TEST = min_max_img_size[0] if min_max_img_size[0] > 0 else 0
        cfg.INPUT.MAX_SIZE_TEST = min_max_img_size[1] if min_max_img_size[1] > 0 else int(1e4)

    if model_cat_names is not None and model_cat_names != predict_cat_names:
        assert set(model_cat_names).intersection(predict_cat_names) != 0
        test_class_subset_mask = [c in predict_cat_names for c in model_cat_names]
        # 1-based mapping from model_cat_names to predict_cat_names
        remap_cat_names = {i + 1: predict_cat_names.index(c) + 1 for i, c in enumerate(model_cat_names) if c in predict_cat_names}
        set_subset_class_ROI_head(cfg, test_class_subset_mask)
    else:
        remap_cat_names = None
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_fn)
    model.eval()

    if warm_up:
        utils.model_warm_up(model, data_mapper=lambda image: [numpy_to_dt2_input(image)])

    predictions = []
    predictions_without_nms = []
    # dataset: DT2 dataset from a DT2 database
    if dataset_name is not None:
        data_loader = build_batchable_detection_test_loader(
            cfg=cfg, dataset_name=dataset_name, batch_size=imgs_per_batch)

        with torch.no_grad():
            for i, inputs in enumerate(data_loader):
                outputs = model(inputs)
                for inp, outp in zip(inputs, outputs):
                    predictions.append({
                        "image_id": inp["image_id"],
                        "instances": instances_to_coco_json(outp["instances"].to("cpu"), img_id=inp["image_id"])
                    })
                    if save_pred_frames:
                        visualize_predictions(
                            img=cv2.imread(inp["file_name"]), det_per_img=predictions[-1]['instances'],
                            cat_names=MetadataCatalog.get(dataset_name).get('thing_classes', None),
                            vis_threshold=vis_threshold,
                            output_fn=os.path.join(output_dir, os.path.basename(inp["file_name"])),
                            one_based=False, save=True
                        )
                if quick_debug:
                    break

    # dataset: list of images or list of videos
    else:
        if output_fn is None:
            output_fn = os.path.join(output_dir, 'predictions.json')
        assert not evaluate
        if track:
            assert new_track_thr is None or new_track_thr <= 1
            from botsort.tracker.mc_bot_sort import BoTSORT
            from sort import sort
            track_cfg = SimpleNamespace(
                track_high_thresh=track_high_thr if track_high_thr is not None else 0.5,
                track_low_thresh=track_low_thr if track_low_thr is not None else 0.1,
                new_track_thresh=new_track_thr if new_track_thr is not None else 0.6,
                track_buffer=track_buffer if track_buffer is not None else 30,
                match_thresh=track_match_thr if track_match_thr is not None else 0.8,
                cmc_method='sparseOptFlow',
                name='DT2',
                ablation=False,
                with_reid=False,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                mot20=False
            )
            tracker = BoTSORT(args=track_cfg)

        predictor = DefaultPredictor(cfg)
        assert predictor.input_format == utils.BGR_FORMAT

        if not video_input:
            # os.makedirs(os.path.join(output_dir, 'without_nms'), exist_ok=True)
            for inp_fn in dataset:
                # use PIL, to be consistent with evaluation
                img = utils.read_image(inp_fn, format=utils.BGR_FORMAT)
                outp = predictor(img)

                # det_per_img_without_nms = instances_to_coco_json(outp["instances"].to("cpu"), img_id=inp_fn)
                # predictions_without_nms.append({"image_id": inp_fn, "instances": det_per_img_without_nms})
                # if save_pred_frames:
                #     visualize_predictions(
                #         img=img, det_per_img=det_per_img_without_nms,
                #         cat_names=model_cat_names, vis_threshold=vis_threshold,
                #         output_fn=os.path.join(output_dir, 'without_nms', os.path.basename(inp_fn)),
                #         one_based=False, save=True
                #     )

                if track:
                    # class agnostic NMS
                    idx = nms(outp["instances"].pred_boxes.tensor, outp["instances"].scores,
                              iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                else:
                    idx = np.arange(len(outp["instances"]), dtype=int)

                det_per_img = instances_to_coco_json(outp["instances"][idx].to("cpu"), img_id=inp_fn)
                if track:
                    try:
                        update_tracker_with_detection(
                            tracker=tracker, det_per_img=det_per_img, img=img, iou_func=sort.iou_batch)
                        det_per_img = [d for d in det_per_img if 'track_id' in d]
                    except:
                        print(f'Tracking failed for {inp_fn}')
                        traceback.print_exc()
                predictions.append({"image_id": inp_fn, "instances": det_per_img})
                if save_pred_frames:
                    visualize_predictions(
                        img=img, det_per_img=det_per_img,
                        cat_names=model_cat_names, vis_threshold=vis_threshold,
                        output_fn=os.path.join(output_dir, os.path.basename(inp_fn)), one_based=False, save=True
                    )
        else:
            # video_input
            for inp_fn in dataset:
                vid_predictions = []
                assert os.path.isfile(inp_fn)
                input_video = cv2.VideoCapture(inp_fn)
                width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = input_video.get(cv2.CAP_PROP_FPS)
                # num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
                video_out_fn = os.path.join(output_dir, os.path.basename(inp_fn))
                if save_pred_frames:
                    CODEC = 'mp4v'
                    assert video_out_fn.endswith('.mp4')
                    if os.path.exists(video_out_fn):
                        os.remove(video_out_fn)
                    output_video = cv2.VideoWriter(
                        filename=video_out_fn,
                        fourcc=cv2.VideoWriter_fourcc(*CODEC),
                        fps=float(frames_per_second),
                        frameSize=(width, height),
                        isColor=True,
                    )

                for f, frame in enumerate(utils.frame_from_video(input_video)):
                    frame_fn = f"{inp_fn[:-4]}.frame_{f + (1 if one_based_video_frames else 0):06d}.jpg"
                    outp = predictor(frame)
                    if track:
                        # class agnostic NMS
                        idx = nms(outp["instances"].pred_boxes.tensor, outp["instances"].scores,
                                  iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                    else:
                        idx = np.arange(len(outp["instances"]), dtype=int)
                    det_per_img = instances_to_coco_json(outp["instances"][idx].to("cpu"), img_id=frame_fn)
                    if track:
                        try:
                            update_tracker_with_detection(
                                tracker=tracker, det_per_img=det_per_img, img=frame, iou_func=sort.iou_batch)
                            det_per_img = [d for d in det_per_img if 'track_id' in d]
                        except:
                            print(f'Tracking failed for frame {f}')
                            traceback.print_exc()
                    vid_predictions.append({"image_id": frame_fn, "instances": det_per_img})
                    if save_pred_frames:
                        visualize_predictions(
                            img=frame, det_per_img=det_per_img,
                            cat_names=model_cat_names, vis_threshold=vis_threshold,
                            output_video=output_video, one_based=False, save=True
                        )
                predictions.extend(copy.deepcopy(vid_predictions))
                save_predictions_as_json(
                    predictions=vid_predictions,
                    output_fn=os.path.join(os.path.dirname(output_fn), f"{os.path.basename(inp_fn[:-4] if inp_fn[-4] == '.' else inp_fn)}.json"),
                    model_cat_names=model_cat_names, remap_cat_names=remap_cat_names, compress=compress)
                input_video.release()
                if save_pred_frames:
                    output_video.release()

    if dataset_name is not None or not video_input:
        save_predictions_as_json(predictions=predictions, output_fn=output_fn,
                                 model_cat_names=model_cat_names, remap_cat_names=remap_cat_names, compress=compress)

        if len(predictions_without_nms) != 0:
            save_predictions_as_json(predictions=predictions_without_nms, output_fn=output_fn.replace('.json', '.without_nms.json'),
                                     model_cat_names=model_cat_names, remap_cat_names=remap_cat_names,
                                     compress=compress)

    if evaluate and dataset_name is not None:
        if eval_log_info:
            print(eval_log_info)
        r = utils.evaluate(gt_coco=MetadataCatalog.get(dataset_name).json_file, dt_coco=output_fn,
                           iouType='segm' if cfg.MODEL.MASK_ON else 'bbox', maxDets=detections_per_image,
                           areaRng=None, areaRngLbl=None, PR_curve=True, allow_zero_area_boxes=True,
                           fix_zero_ann_ids=True, verbose=True)
        print("Categories:", " ".join([str(c) for c in predict_cat_names]))
        print("AP50 for each category:", r.precision.mean(axis=0))

    print("Finished predictions")
    return output_fn, predictions


def update_tracker(tracker, det_per_img, image,
                   aspect_ratio_thresh=10, min_box_area=1, box_format='xywh'):
    '''

    :param tracker: BoTSORT tracker object
    :param det_per_img: (n,6) array of detections per image [[x, y, x, y, conf, cls], ...]
                        det_per_img WILL BE MODIFIED IN PLACE
    :param image: (h, w, c) array of pixels in the BGR (cv2) format
    :return: track_ids
    '''

    if len(det_per_img):
        if box_format.lower() == 'xywh':
            det_per_img[:, :4] = utils.from_XYWH_to_XYXY(det_per_img[:, :4])
        else:
            assert box_format.lower() == 'xyxy'
    else:
        det_per_img = []

    output_stracks = tracker.update(det_per_img, image)

    # REMOVE THIS IN THE FUTURE
    for t in output_stracks:
        vertical = t.tlwh[2] / t.tlwh[3] > aspect_ratio_thresh
        too_small = t.tlwh[2] * t.tlwh[3] <= min_box_area
        assert not too_small and not vertical, det_per_img

    tracks = np.asarray(
        [t.tlwh.tolist() + [t.score, t.cls, t.track_id, t.idx] for t in output_stracks],
        dtype=np.float32
    )
    assert len(tracks) == 0 or len(tracks) == len(set(tracks[:, -1])), "Not a unique matching to detections"

    return tracks


def update_tracker_with_detection(tracker, det_per_img, img, iou_func):
    if len(det_per_img) != 0:
        tracks = update_tracker(
            tracker=tracker,
            det_per_img=np.asarray([
                np.asarray(det['bbox'] + [det['score'], -1], dtype=np.float32) for det in det_per_img
            ]),
            image=img
        )
        # print(tracks)
        if len(tracks) != 0:
            # tracks: [[x, y, w, h, score, class, track_id, idx], ...]
            for idx, t in zip(tracks[:, -1].astype(int), tracks[:, :-1]):
                assert det_per_img[idx]['score'] == t[4], (det_per_img[idx], t)
                iou = iou_func(np.asarray([utils.from_XYWH_to_XYXY(det_per_img[idx]['bbox'])]),
                               np.asarray([utils.from_XYWH_to_XYXY(t[:4])]))
                assert np.all(iou > 0.5), (det_per_img[idx], t, iou)
                det_per_img[idx]['track_id'] = int(t[6])  # update id
                det_per_img[idx]['bbox'] = t[:4].tolist() # update box


def visualize_predictions(
    img, det_per_img, cat_names=None, vis_threshold=None, output_fn=None, output_video=None, one_based=False, save=False
):
    if vis_threshold:
        det_per_img = [d for d in det_per_img if d['score'] > vis_threshold]
    if len(det_per_img) != 0:
        vis_img = utils.plot_bboxes(
            img, det_per_img, cat_dict={cid: cat for cid, cat in enumerate(cat_names, 1 if one_based else 0)})
    else:
        vis_img = img
    if save:
        assert sum([output_fn is None, output_video is None]) == 1
        if output_fn is not None:
            assert isinstance(output_fn, str)
            utils.write_image(vis_img, output_fn)
        if output_video is not None:
            output_video.write(vis_img)
    return vis_img


def save_predictions_as_json(predictions, output_fn, model_cat_names=None, remap_cat_names=None, compress=False):
    predictions = list(itertools.chain(*[p["instances"] for p in predictions]))
    # detectron2 is 0-based but COCO-format is 1-based
    for pred in predictions:
        cls = pred['category_id'] + 1
        assert model_cat_names is None or cls in range(1, len(model_cat_names) + 1)
        # remap to predict_cat_names
        if remap_cat_names is not None:
            cls = remap_cat_names[cls]
        pred['category_id'] = cls
    utils.save_json(dict(annotations=predictions), output_fn, only_preds=True, compress=compress)


def numpy_to_dt2_input(image):
    return dict(image=torch.from_numpy(image.transpose(2, 0, 1)))


def read_dt2_config(fn, strict=True):
    cfg = detectron2.config.get_cfg()
    cfg.TEST['EVAL_CLASS_SUBSET'] = None
    cfg.MODEL.RESNETS['DROPOUT'] = None
    cfg.MODEL.ROI_BOX_HEAD['DROPOUT'] = None
    cfg.INPUT['COPY_PASTE'] = detectron2.config.config.CfgNode(
        {'ENABLED': False, 'TYPE': None, 'N_OBJECTS': None, 'MAX_IOA': None, 'JSON_FN': None, 'IMG_DIR': None})
    cfg.merge_from_file(fn)
    assert not strict or cfg.INPUT.FORMAT == utils.BGR_FORMAT, f'C.INPUT.FORMAT must be "BGR", not {cfg.INPUT.FORMAT}'
    return cfg


def set_subset_class_ROI_head(cfg, test_class_subset_mask):
    cfg.MODEL.ROI_HEADS.NAME = 'ClassSubsetROIHeads'
    cfg.MODEL.ROI_HEADS['CLASS_SUBSET_MASK'] = list(test_class_subset_mask)
    cfg.MODEL.ROI_HEADS['CLASS_SUBSET_MASK'].append(True)  # background class


def build_batchable_detection_test_loader(cfg, dataset_name, batch_size=None, num_workers=None):

    if batch_size is None or batch_size == 1:
        # the default Detectron2 approach
        data_loader = build_detection_test_loader(
            cfg=cfg, dataset_name=dataset_name)
    else:
        # replica of the above but allowing batch_size > 1
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]

        dataset = get_detection_dataset_dicts(
            names=dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
            ] if cfg.MODEL.LOAD_PROPOSALS else None
        )
        dataset = MapDataset(dataset=dataset, map_func=DatasetMapper(cfg, False))

        data_loader = torchdata.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None if isinstance(dataset, torchdata.IterableDataset) else InferenceSampler(len(dataset)),
            num_workers=cfg.DATALOADER.NUM_WORKERS if num_workers is None else num_workers,
            collate_fn=trivial_batch_collator,
        )

    return data_loader


if __name__ == '__main__':
    main()