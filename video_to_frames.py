import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', nargs='+', help='filename: video (cannot be used if "input_dir" specified)')
    parser.add_argument('--input_dir', help='directory to be scanned for videos (cannot be used if "input_fn" specified')
    parser.add_argument('--output_dir')
    parser.add_argument('--ext', default='mp4', help='required if "input_dir" specified, not case-sensitive')
    parser.add_argument('--every_n_frame', type=int)
    parser.add_argument('--every_n_sec', type=int)
    parser.add_argument('--start_at', type=int, help='convenience if you extracted every 20th frame and you want to increase to every 10th frame use "--every_n_frame 20 --start_at 10"')
    parser.add_argument('--stop_at', type=int)
    parser.add_argument('--just_info', action='store_true', help='only report on number of frames')
    parser.add_argument('--subfolders', action='store_true', help='create output subfolder for every video')
    parser.add_argument('--save_json', action='store_true', help='create JSON annotations files (only if "subfolder" turned on)')
    parser.add_argument('--cat_names', nargs='+', help='if "cat_names" listed, JSON file will include them (only if "save_json" turned on)')
    parser.add_argument('--disable_fast_forward', action='store_true', help='if the video file is not correctly formatted, it has to be split frame-by-frame without fast forward')
    args = parser.parse_args()

    import numpy as np
    import os

    try:
        assert np.sum([args.input_fn is not None, args.input_dir is not None]) == 1, 'Provide one of --input_fn or --input_dir'
        assert args.input_fn is None or all([os.path.exists(fn) for fn in args.input_fn]), 'Some files --input_fn do not exist'
        assert args.input_dir is None or os.path.isdir(args.input_dir), 'Directory --input_dir does not exist'
        assert args.input_dir is None or args.ext is not None, 'Extension --ext need to be specified'
        assert not args.save_json or args.subfolders, 'Cannot --save_json if not --subfolders'
        assert args.cat_names is None or args.save_json, 'Only list --cat_names when --save_json'
        assert args.just_info or args.output_dir, 'Specify --output_dir'
    except AssertionError as e:
        parser.error(str(e))

    import utils

    if args.ext.startswith('.'):
        args.ext = args.ext[1:]

    if args.input_dir is not None:
        args.input_fn = utils.read_files_from_dir(
            args.input_dir, filter_func=lambda f: f.lower().endswith(f'.{args.ext.lower()}'), basename_only=False)

    if not args.just_info:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    for fn in args.input_fn:

        if args.subfolders and not args.just_info:
            out_dir = os.path.join(args.output_dir, utils.basename_without_ext(fn))
            os.makedirs(out_dir, exist_ok=True)

        if args.just_info:
            metadata = utils.get_video_metadata(fn, manual_counting=args.disable_fast_forward)
            print(f'{os.path.basename(fn)}: {metadata["total_frames"]} frames, {metadata["fps"]} fps, {metadata["duration"]} duration, {metadata["resolution"]} resolution')
        else:
            print(f'Splitting {os.path.basename(fn)} into frames, location: {out_dir}{os.path.sep}')
            if fn.lower().endswith('.asf') and not args.disable_fast_forward:
                print('WARNING: Many of our ASF videos were not correctly formatted. '
                      'If this program hangs, turn on "disable_fast_forward".')

            frames, _ = utils.extract_frames(
                filename=fn, output_dir=out_dir, every_n_sec=args.every_n_sec, every_n_frame=args.every_n_frame,
                start_at=args.start_at, stop_at=args.stop_at, fast_forward=False if args.disable_fast_forward else 'auto')

            if args.save_json:
                utils.prepare_prediction_dataset_on_disk(
                    input_fn=None, img_dir=out_dir, output_dir=out_dir, cat_names=args.cat_names,
                    out_json_fn='annotations.json')


if __name__ == '__main__':
    main()
