import os
import shutil
import sys
from ultralytics import YOLO
import itertools
import copy
import cv2
import torch
import numpy as np

if os.path.exists('utils.py'):
    import utils
else:
    import cv_tools as utils


def yolo_predict(model, dataset, output_dir=None, output_fn=None, video_input=None, model_cat_names=None, predict_cat_names=None,
                 threshold=None, NMS_threshold=None, detections_per_image=None, img_size=None,
                 track=None, track_buffer=None, new_track_thr=None, track_match_thr=None, track_high_thr=None, track_low_thr=None,
                 save_pred_frames=False, vis_threshold=None, evaluate=False, warmup=False,
                 stream=True, device=None, compress=True, iouType='bbox', eval_log_info=None, quick_debug=False):

    assert track in [None, True, False, utils.SORT, utils.BOT_SORT, utils.BYTE_TRACK, utils.DUMMY_TRACK]
    if track is True:
        track = utils.BOT_SORT
    assert stream, 'Without streaming tracking does not work with long videos'
    assert not evaluate or isinstance(dataset, tuple)
    assert not (model_cat_names is None and predict_cat_names is not None)

    if vis_threshold == 0:
        vis_threshold = None
    if save_pred_frames and vis_threshold is not None:
        from ultralytics.engine.results import Results

    assert video_input in [None, False, True]
    if isinstance(dataset, str) and os.path.isdir(dataset):
        vids = utils.read_videos_from_dir(dir_name=dataset, basename_only=False)
        if video_input:
            dataset = sorted(vids)
        else:
            imgs = utils.read_images_from_dir(dir_name=dataset, basename_only=False)
            if len(imgs) == 0 and len(vids) != 0:
                print(f'WARNING: Did not find any images in {dataset}, did you forget to specify "video_input=True"?')
            dataset = sorted(imgs)

    if isinstance(dataset, tuple):
        # This is implemented for pure predict and SORT tracking
        # It assumes COCO dataset, so it can be easily replaced by other detectors
        print('Dataset from COCO!')
        assert track not in [utils.BOT_SORT, utils.BYTE_TRACK]
        input_json_fn, img_dir = dataset
        from_coco = True

        dataset = utils.read_json(input_json_fn, only_imgs=True)
        predict_cat_names = utils.assure_consistent_cat_names(dataset, predict_cat_names=predict_cat_names)
        if predict_cat_names is None:
            predict_cat_names = model_cat_names
        dataset, img_ids = zip(*[
            (os.path.join(img_dir, img['file_name']), img['id']) for img in dataset['images']
        ])
        if quick_debug:
            dataset = dataset[:2]
            img_ids = img_ids[:2]
        assert track is None or list(dataset) == sorted(dataset)
    else:
        # This is implemented to experiment with YOLO and for BOT_SORT AND BYTE_TRACK (they require video input)
        from_coco = False
        img_ids = None
        if not utils.is_iterable(dataset):
            dataset = [dataset]

    if predict_cat_names is None:
        predict_cat_names = model_cat_names

    if isinstance(model, str):
        model = YOLO(model)
    os.makedirs(output_dir, exist_ok=True)

    _yolo_cat_names = [name for name in model.names]
    if model_cat_names is not None:
        assert list(range(len(model_cat_names))) == _yolo_cat_names or list(model_cat_names) == _yolo_cat_names, \
            (list(range(len(model_cat_names))), list(model_cat_names), _yolo_cat_names)

    if track is not None:
        if track == utils.SORT:
            from sort import sort
            assert new_track_thr is None or (new_track_thr >= 1 and new_track_thr == int(new_track_thr))
            track_cfg = {}
            if track_buffer is not None:
                track_cfg['max_age'] = track_buffer
            if new_track_thr is not None:
                track_cfg['min_hits'] = new_track_thr
            if track_match_thr is not None:
                track_cfg['iou_threshold'] = track_match_thr
            tracker = sort.Sort(**track_cfg)
        elif track != utils.DUMMY_TRACK:
            track_cfg = dict(
                tracker_type=track,
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
                gmc_method='sparseOptFlow',
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False
            )

            if track_buffer is not None:
                track_cfg['track_buffer'] = track_buffer
            if new_track_thr is not None:
                track_cfg['new_track_thresh'] = new_track_thr
            if track_match_thr is not None:
                track_cfg['match_thresh'] = track_match_thr
            if track_high_thr is not None:
                track_cfg['track_high_thresh'] = track_high_thr
            if track_low_thr is not None:
                track_cfg['track_low_thresh'] = track_low_thr

            track_cfg_fn = os.path.join(output_dir, 'track_config.yaml')
            utils.save_yaml(track_cfg, track_cfg_fn)

    if model_cat_names is not None and model_cat_names != predict_cat_names:
        assert set(model_cat_names).intersection(predict_cat_names) != 0
        test_classes_idx = np.arange(len(model_cat_names), dtype=int)[[c in predict_cat_names for c in model_cat_names]]
        # 1-based mapping from model_cat_names to predict_cat_names
        remap_cat_names = {i + 1: predict_cat_names.index(c) + 1 for i, c in enumerate(model_cat_names) if c in predict_cat_names}
    else:
        test_classes_idx = None
        remap_cat_names = None

    kwargs = dict(
        project=output_dir, name='predictions',
        classes=test_classes_idx, stream=stream, device=utils.get_device(device, model='yolo8'),
        save_crop=False, save_txt=False, save_conf=False
    )
    if img_size is not None:
        kwargs['imgsz'] = img_size
    if threshold is not None:
        kwargs['conf'] = threshold
    if NMS_threshold is not None:
        kwargs['iou'] = NMS_threshold
    if detections_per_image is not None:
        kwargs['max_det'] = detections_per_image
    if warmup:
        utils.model_warm_up(model)
    yolo_pred_dir = os.path.join(output_dir, 'predictions')
    shutil.rmtree(yolo_pred_dir, ignore_errors=True)

    predictions = []
    just_videos = []
    images = None if from_coco else []
    short_video_ids = all([isinstance(source, str) and utils.is_video(source) for source in dataset]) and \
                      len(dataset) == len(['.'.join(os.path.basename(source).split('.')[:-1]) for source in dataset])
    for i, source in enumerate(dataset):
        if isinstance(source, str) and utils.is_video(source):
            assert video_input is None or video_input is True, f'Found {source} but video_input is {video_input}'
            is_video = True
            print(source)
        else:
            is_video = False
        just_videos.append(is_video)
        kwargs['source'] = source
        if track in [None, utils.SORT, utils.DUMMY_TRACK]:
            kwargs['save'] = save_pred_frames and (is_video or vis_threshold is None)
            outputs = model.predict(**kwargs)
        else:
            kwargs['save'] = True
            kwargs['verbose'] = False
            outputs = model.track(tracker=track_cfg_fn, **kwargs)

        if is_video:
            assert not from_coco
            vid_predictions = []
        for j, outp in enumerate(outputs):
            if track == utils.SORT:
                boxes_xyxy = outp.boxes.xyxy.cpu().numpy()
                track_ids = tracker.update(boxes_xyxy)
                if len(track_ids) != len(boxes_xyxy):
                    print(f'WARNING: {len(track_ids)} track_ids and {len(boxes_xyxy)} boxes')
                assert len(track_ids) <= len(boxes_xyxy)
            elif track != utils.DUMMY_TRACK:
                track_ids = outp.boxes.id.clone().cpu().numpy() if outp.boxes.id is not None else None
            else:
                track_ids = None
            sys.stdout.flush()

            img_id = img_ids[i] if from_coco else \
                f'{(".".join(os.path.basename(source).split(".")[:-1]) if short_video_ids else source) if is_video else ""}{"_" if is_video else ""}{(j + 1)}'
            filename = outp.path

            if is_video:
                n_digits_fn = 6 if stream else max(6, int(np.ceil(np.log10(len(outputs)))))
                filename = f'{".".join(filename.split(".")[:-1])}.frame_{j:0{n_digits_fn}d}.jpg'
                vid_predictions.append({
                "image_id": img_id,
                "instances": utils.instances_to_coco_json(
                    outp, filename, track=track if track != utils.DUMMY_TRACK else None, track_ids=track_ids,
                    one_based_cats=True, model_cat_names=model_cat_names, remap_cat_names=remap_cat_names)
            })
            else:
                predictions.append({
                    "image_id": img_id,
                    "instances": utils.instances_to_coco_json(
                        outp, img_id, track=track if track != utils.DUMMY_TRACK else None, track_ids=track_ids,
                        one_based_cats=True, model_cat_names=model_cat_names, remap_cat_names=remap_cat_names)
                })

            if images is not None:
                images.append(dict(
                    id=img_id,
                    file_name=filename,
                ))

            if save_pred_frames and not is_video:
                if track == utils.SORT:
                    img = cv2.imread(filename)
                    for ann in predictions[-1]['instances']:
                        bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]]
                        label = f'ID:{ann["track_id"]} {predict_cat_names[ann["category_id"] - 1]}'
                        color = [0, 0, 0]
                        color[-ann["category_id"]] = 255
                        utils.draw_box(img=img, bbox=bbox, label=label, color=color)
                    cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), img)
                elif vis_threshold is not None:
                    assert outp.masks is None and outp.probs is None
                    vis_mask = outp.boxes.conf > vis_threshold
                    if vis_mask.any():
                        subset_outp = Results(orig_img=cv2.imread(filename), path=outp.path, names=outp.names,
                                              boxes=outp.boxes.data[torch.as_tensor(vis_mask)], masks=None, probs=None)
                        utils.draw_boxes_natively(model='yolo8', result=subset_outp,
                                                  filename=os.path.join(output_dir, os.path.basename(filename)))
                    else:
                        shutil.copyfile(src=filename,
                                        dst=os.path.join(output_dir, os.path.basename(filename)))
                else:
                    # reusing YOLO plotting API (now instead I copy the files as it can be faster)
                    shutil.copyfile(src=os.path.join(yolo_pred_dir, os.path.basename(filename)),
                                    dst=os.path.join(output_dir, os.path.basename(filename)))

        if is_video:
            predictions.extend(copy.deepcopy(vid_predictions))
            utils.save_json(
                dict(annotations=list(itertools.chain(*[p["instances"] for p in vid_predictions]))),
                fn=os.path.join(output_dir, f"{os.path.basename(source[:-4] if source[-4] == '.' else source)}.json"),
                only_preds=True, compress=compress
            )

        if save_pred_frames and is_video:
            video_fn = os.path.join(f'{yolo_pred_dir}{str(i + 1) if i != 0 else ""}', os.path.basename(source))
            if not os.path.exists(video_fn) and video_fn.endswith('.mp4') and os.path.exists(f'{video_fn[:-4]}.avi'):
                video_fn = f'{video_fn[:-4]}.avi'

            if os.path.exists(video_fn):
                shutil.copyfile(src=video_fn,
                                dst=os.path.join(output_dir, os.path.basename(video_fn)))
            else:
                print(f'WARNING: cannot move {video_fn}.')

    if output_fn is None:
        output_fn = os.path.join(output_dir, 'predictions.json')
    for i in range(len(dataset)):
        shutil.rmtree(f'{yolo_pred_dir}{str(i + 1) if i != 0 else ""}', ignore_errors=True)
    predictions = list(itertools.chain(*[p["instances"] for p in predictions]))
    if len(just_videos) == 0 or not np.all(just_videos):
        utils.save_json(
            dict(annotations=predictions) if from_coco else dict(images=images, annotations=predictions),
            output_fn, assert_correct=from_coco, only_preds=True, compress=compress
        )

    if evaluate:
        assert from_coco
        if eval_log_info:
            print(eval_log_info)
        r = utils.evaluate(gt_coco=input_json_fn, dt_coco=output_fn, iouType=iouType, maxDets=detections_per_image,
                           areaRng=None, areaRngLbl=None, PR_curve=True, allow_zero_area_boxes=True,
                           fix_zero_ann_ids=True, verbose=True)
        print("Categories:", " ".join([str(c) for c in predict_cat_names]))
        print("AP50 for each category:", r.precision.mean(axis=0))
    print("Finished predictions")
    return output_fn, predictions
