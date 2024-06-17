import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_files', nargs='+', help='Filenames of COCO json annotations')
    parser.add_argument('--video_dir', required=True, help='directory where videos are stored')
    parser.add_argument('--output_dir', required=True, help='directory to save the extracted frames')
    parser.add_argument('--do_not_traverse', action='store_true',
                help='Do not traverse the subdirectories of the video_dir')
    parser.add_argument('--copy_to_tmp', action='store_true',
                help='Copy the video into a temp dir (e.g. if it is in the cloud)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing jpg files')
    parser.add_argument('--ext', default='mp4', help='Search for this kind of video files')
    parser.add_argument('--verbose', type=int, default=1, help='0: quite, 1: info, 2: detailed')
    args = parser.parse_args()

    try:
        import os
        for fn in args.annot_files:
            assert os.path.exists(fn), f'File {fn} does not exist'
            assert fn.endswith('.json'), f'File {fn} is not a json file'
        assert os.path.exists(args.video_dir), f'{args.video_dir} does not exist'
        assert os.path.exists(args.output_dir), f'{args.output_dir} does not exist'
        assert os.path.isdir(args.output_dir), f'{args.output_dir} is not a directory'

        if os.path.isfile(args.video_dir):
            from utils import is_video
            assert is_video(args.video_dir), f'{args.video_dir} is not a supported video file'
            assert len(args.annot_files) == 1, f'More than one annotation file for a single video'
    except AssertionError as e:
        parser.error(str(e))

    import sys
    import utils

    for annot_fn in args.annot_files:
        try:
            if args.verbose:
                print(f'READING {annot_fn}')
            assert annot_fn.endswith('.json'), f'File {annot_fn} is not a json file'
            dataset_name = os.path.basename(annot_fn)[:-5]

            if os.path.isfile(args.video_dir):
                video_fn = args.video_dir
            elif args.do_not_traverse:
                video_fn = os.path.join(args.video_dir, f'{dataset_name}.{args.ext}')
            else:
                from glob import glob
                matches = glob(f'{args.video_dir}{os.path.sep}**{os.path.sep}{dataset_name}.{args.ext}', recursive=True)
                assert len(matches) in [0, 1], f'Found more than 1 video matching {dataset_name}.{args.ext}'
                video_fn = matches[0] if len(matches) != 0 else None

            assert video_fn is not None and os.path.exists(video_fn), f'No video for {dataset_name}'

            dataset = utils.read_json(annot_fn, verbose=args.verbose > 0)
            frames_ids, frame_fns = utils.extract_frames_for_dataset(
                images=dataset['images'], video_fn=video_fn, output_dir=args.output_dir, copy_to_tmp=args.copy_to_tmp,
                overwrite=args.overwrite, verbose=args.verbose)

        except AssertionError as e:
            print(f'ERROR: {e}\n', file=sys.stderr)


if __name__ == '__main__':
    main()
