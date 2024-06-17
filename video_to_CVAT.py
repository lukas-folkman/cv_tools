import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='filename of the video')
    parser.add_argument('--just_info', action='store_true', help='only report the FPS and the total number of frames')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--annotate', help='Model "pt" file to make predictions for bboxes')
    parser.add_argument('--output_dir', default='.', help='directory to store created CVAT task')
    parser.add_argument('--conf_thr', type=float, default=0.5,
    help='Annotate with predictions that have confidence >= "conf_thr" (default: 0.5, i.e. 50% confident)')
    parser.add_argument('--every_n_frame', type=int, help='extract only every N-th frame')
    parser.add_argument('--every_n_sec', type=int, help='extract only every N-th second')
    parser.add_argument('--after_n_extracted', type=int, help='after N frames extracted jump ahead by "jump_after_n_extracted"')
    parser.add_argument('--jump_after_n_extracted', type=int, help='after "after_n_extracted" frames extracted jump ahead by N')
    parser.add_argument('--start_at', type=int, help='skip frames at the beginning and start at frame N (suggestions: '
    'if you extracted every 20th frame and want to increase to every 10th frame use "--every_n_frame 20 --start_at 10"')
    parser.add_argument('--stop_at', type=int, help='skip frames at the end and stop at frame N')
    parser.add_argument('--disable_fast_forward', action='store_true', help='if the video file is not correctly '
    'formatted, it has to be split frame-by-frame without fast forward, turn this on if the program does not work well')
    parser.add_argument('--zip', action='store_true', help='Zip the archive.')
    parser.add_argument('--add_delete_label', action='store_true', help='Add delete label in CVAT.')
    parser.add_argument('--use_cache', action='store_true', help='Experimental feature. Do NOT use it.')
    args = parser.parse_args()

    try:
        import os
        assert os.path.exists(args.video), f'Video does not exist {args.video}'
        assert args.just_info or args.labels is not None, 'Specify --labels label1 label2 ...'
        assert len(args.labels) == len(set(args.labels)), 'Labels must be unique'
    except AssertionError as e:
        parser.error(str(e))

    print(args.video)

    import utils

    metadata = utils.get_video_metadata(args.video, manual_counting=args.disable_fast_forward)
    print(f'{metadata["total_frames"]} frames, {metadata["fps"]} fps, {metadata["duration"]} duration, {metadata["resolution"]} resolution')

    if not args.just_info:
        import shutil
        task_name = utils.basename_without_ext(args.video)
        args.output_dir = os.path.join(args.output_dir, task_name)
        img_dir = os.path.join(args.output_dir, 'data')
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)

        print(f'Splitting {os.path.basename(args.video)} into frames, location: {img_dir}')
        if args.video.lower().endswith('.asf') and not args.disable_fast_forward:
            print('WARNING: Many of our ASF videos were not correctly formatted. '
                  'If this program hangs, turn on "disable_fast_forward".')

        frames_fns, _ = utils.extract_frames(
            filename=args.video, output_dir=img_dir,
            every_n_sec=args.every_n_sec, every_n_frame=args.every_n_frame,
            start_at=args.start_at, stop_at=args.stop_at, fast_forward=False if args.disable_fast_forward else 'auto',
            after_n_extracted=args.after_n_extracted, jump_after_n_extracted=args.jump_after_n_extracted
        )

        if args.annotate is not None:
            print('\nAUTO-ANNOTATION')
            from yolo_predict import yolo_annotate
            dataset = yolo_annotate(model=args.annotate, img_dir=img_dir, frames_fns=frames_fns, labels=args.labels, conf_thr=args.conf_thr)

        print('Creating a CVAT task')

        utils.package_for_cvat(
            task_name=task_name, frames_fns=frames_fns,
            labels=args.labels + (['delete'] if args.add_delete_label else []),
            use_cache=args.use_cache, output_dir=args.output_dir, dataset=dataset if args.annotate else None)

        if args.zip:
            print('Zipping the CVAT task')
            archived_fn = shutil.make_archive(
                base_name=os.path.join(args.output_dir, task_name), format='zip', root_dir=args.output_dir)
        else:
            archived_fn = None

        print(f'{f"Upload {archived_fn}" if args.zip else "Create and upload a ZIP archive"} '
              'into CVAT using "[+] Create from backup" option in the "Tasks" view')


if __name__ == '__main__':
    main()
