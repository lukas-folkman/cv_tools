import os
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, choices=['dt2', 'yolo8'],
                        help='Choose if to train YOLOv8 or Detectron2.')
    parser.add_argument('--weights_fn', required=True,
                        help='Model weights (checkpoint) file.')
    parser.add_argument('--config_fn',
                        help='Model configuration file. Required for DT2, optional for YOLO. If empty, it is expected '
                        'that "config.yaml" is located in the same directory as the "weights_fn".')

    parser.add_argument('--input_fn', nargs='+',
                        help='When a JSON dataset/annotations file in the COCO format supplied, specify also "input_dir".'
                             ' If a single image or video file supplied, do NOT specify "input_dir".')
    parser.add_argument('--input_dir',
                        help='Directory with images referenced in "input_fn".')
    parser.add_argument('--video_input', action='store_true', help='The inputs to be processed are videos.')
    parser.add_argument('--output_dir', required=True,
                        help='Specify the output directory.')
    parser.add_argument('--output_fn',
                        help='Specify output JSON file. If not set, "output_dir/predictions.json" will be used.')
    parser.add_argument('--model_cat_names', required=True, nargs='+',
                        help='List of categories for the model (in the correct order). Categories and their order must '
                             'match the categories used for training the model. A JSON file (e.g. training data file) '
                             'can be supplied instead.')

    parser.add_argument('--track', action='store_true', help='Track fish with "tracker" (default: "botsort").')
    parser.add_argument('--tracker', choices=['botsort', 'bytetrack'], default='botsort')
    parser.add_argument('--track_match_thr', type=float, default=0.7,
                        help='Only YOLO. Intersection over union threshold for tracking, experimental feature.')
    parser.add_argument('--new_track_thr', type=float, default=0.5,
                        help='Threshold to start tracking an object.')
    parser.add_argument('--track_high_thr', type=float, default=0.5,
                        help='Threshold for the association in step 1.')
    parser.add_argument('--track_low_thr', type=float, default=0.05,
                        help='Threshold for the association in step 2.')
    parser.add_argument('--track_buffer', type=int, default=30,
                        help='For how many frames an object ID should be remembered for.')

    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Minimum prediction score for the detection to be considered a detection (included).')
    parser.add_argument('--NMS_threshold', type=float, default=0.7,
                        help='Threshold for non-maximum suppression (NMS) postprocessing of raw predictions.')
    parser.add_argument('--vis_threshold', type=float, default=0.1,
                        help='Test time prediction visualization threshold (visualization only).')
    parser.add_argument('--detections_per_image', type=int, default=100,
                        help='Maximum number of detections per image.')

    parser.add_argument('--img_size', type=int, nargs='+',
                        help='If YOLO, then this is the long edge, defaults to 640. '
                             'If DT2, then this is a tuple (short_edge, max_long_edge), defaults to (800, 1333)')
    parser.add_argument('--do_not_save_pred_frames', action='store_true',
                        help='Do NOT save predicted frames.')
    parser.add_argument('--do_not_evaluate', action='store_true',
                        help='Do NOT evaluate accuracy of predictions.')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4'],
                        help='Device to be used for computation. Defaults to "cuda:0" if cuda is available.')
    args = parser.parse_args()

    if os.path.exists('utils.py'):
        import utils
    else:
        import cv_tools as utils
    from datetime import datetime

    try:

        assert 0 <= args.track_match_thr <= 1, '"track_match_thr" must be greater or equal to 0 and less or equal to 1'
        assert 0 <= args.new_track_thr <= 1, '"new_track_thr" must be greater or equal to 0 and less or equal to 1'
        assert 0 <= args.track_high_thr <= 1, '"track_high_thr" must be greater or equal to 0 and less or equal to 1'
        assert 0 <= args.track_low_thr <= 1, '"track_low_thr" must be greater or equal to 0 and less or equal to 1'
        assert 0 <= args.threshold <= 1, '"threshold" must be greater or equal to 0 and less or equal to 1'
        assert 0 <= args.NMS_threshold <= 1, '"NMS_threshold" must be greater or equal to 0 and less or equal to 1'
        assert 0 <= args.vis_threshold <= 1, '"vis_threshold" must be greater or equal to 0 and less or equal to 1'
        assert not (args.track and args.model == 'dt2') or args.tracker == utils.BOT_SORT, '"bytetrack" not for "dt2"'
        assert args.tracker in [None, utils.BOT_SORT, utils.BYTE_TRACK], 'incorrect "tracker"'

        if args.model_cat_names is not None \
                and len(args.model_cat_names) == 1 \
                and utils.file_or_gzip_exists(args.model_cat_names[0]):
            args.model_cat_names = utils.load_model_cat_names_from_file(args.model_cat_names[0])
            print(f'INFO: Loaded model\'s categories: {args.model_cat_names}')

        if args.model == 'dt2':
            assert args.img_size is None or len(args.img_size) == 2, \
                'When using "dt2", "img_size" is a tuple (short_edge, max_long_edge)'

            if args.config_fn is None:
                model_dir = os.path.dirname(args.weights_fn)
                model_name = f'{".".join(os.path.basename(args.weights_fn).split(".")[:-1])}'
                if os.path.exists(os.path.join(model_dir, f'{model_name}.yaml')):
                    args.config_fn = os.path.join(model_dir, f'{model_name}.yaml')
                elif os.path.exists(os.path.join(model_dir, 'config.yaml')):
                    args.config_fn = os.path.join(model_dir, 'config.yaml')
                else:
                    assert False, 'Specify "config_fn" for a dt2 model'
                print(f'INFO: Using "{args.config_fn}" as config.')

        if args.model == 'yolo8':
            assert args.img_size is None or len(args.img_size) == 1, \
                'When using "yolo8", "img_size" is the long edge'
            if args.img_size is not None:
                args.img_size = args.img_size[0]

        _err_msg = 'The following inputs are possible: video inputs ("input_fn") or image inputs ("input_fn") or' \
                   ' one directory with images/videos ("input_dir") or one COCO-format dataset with ".json" extension' \
                   ' ("input_fn") together with one directory with images ("input_dir")'
        if args.input_fn is None and args.input_dir is not None:
            dataset = args.input_dir
        elif args.input_fn is not None and args.input_dir is not None \
                and len(args.input_fn) == 1 and args.input_fn[0].lower().endswith('.json'):
            assert not args.track, 'Tracking takes video inputs ("input_fn") or one directory with images ("input_dir")'
            dataset = (args.input_fn[0], args.input_dir)
        elif args.input_fn is not None and args.input_dir is None:
            if (all([utils.is_video(x) for x in args.input_fn]) or
                 all([utils.is_image(x) for x in args.input_fn])):
                dataset = args.input_fn
            elif len(args.input_fn) == 1 and os.path.isdir(args.input_fn[0]):
                dataset = args.input_fn[0]
            else:
                assert False,_err_msg
        else:
            assert False, _err_msg

        print(f'INFO: this is the input dataset: {dataset}')

    except AssertionError as e:
        parser.error(str(e))

    if args.model == 'dt2':
        from dt2_predict import dt2_predict as predict_func
        kwargs = dict(
            cfg=args.config_fn, weights_fn=args.weights_fn, min_max_img_size=args.img_size,
        )
    elif args.model == 'yolo8':
        from yolo_predict import yolo_predict as predict_func
        kwargs = dict(model=args.weights_fn, img_size=args.img_size)

    shared_kwargs = dict(
        dataset=dataset, output_dir=args.output_dir, output_fn=args.output_fn,
        video_input=True if args.video_input else None, model_cat_names=args.model_cat_names,
        threshold=args.threshold, NMS_threshold=args.NMS_threshold,
        detections_per_image=args.detections_per_image, vis_threshold=args.vis_threshold,
        save_pred_frames=not args.do_not_save_pred_frames, evaluate=not args.do_not_evaluate, device=args.device,
        track=args.tracker if args.track else False, track_buffer=args.track_buffer, new_track_thr=args.new_track_thr,
        track_match_thr=args.track_match_thr, track_low_thr=args.track_low_thr, track_high_thr=args.track_high_thr
    )

    print('INFO: detect_and_track.py', end=' ')
    for arg in vars(args):
        print(f'--{arg} {getattr(args, arg)}', end=' ')

    print(f'\nSTARTED: {datetime.now().strftime("%d %B %Y, %H:%M:%S")}')
    predict_func(**shared_kwargs, **kwargs)
    print(f'FINISHED: {datetime.now().strftime("%d %B %Y, %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
