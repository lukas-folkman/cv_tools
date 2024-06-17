import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('zipped_annot_files', nargs='+', help='filenames of zipped COCO json annotations exported from CVAT')
    parser.add_argument('--output_dir', required=True, help='directory to unpacked and processed json files')
    parser.add_argument('--do_not_fix_frame_names', action='store_true',
                help='keep the "file_name" field intact, default: rename so that it matches the name of the video')
    parser.add_argument('--copy_to_tmp', action='store_true',
                        help='Copy the zip file into a temp dir (e.g. if it is in the cloud)')
    parser.add_argument('--remove_img_without_annot', action='store_true', help='Remove images without annotations')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing json files')
    parser.add_argument('--verbose', type=int, default=1, help='0: quite, 1: info, 2: detailed')
    args = parser.parse_args()

    try:
        import os
        for fn in args.zipped_annot_files:
            assert os.path.exists(fn), f'File {fn} does not exist'
            assert fn.endswith('.zip'), f'File {fn} is not a zip archive'
        assert os.path.exists(args.output_dir), f'{args.output_dir} does not exist'
        assert os.path.isdir(args.output_dir), f'{args.output_dir} is not a directory'
    except AssertionError as e:
        parser.error(str(e))

    import sys
    import utils
    import shutil
    from tempfile import TemporaryDirectory

    for zip_fn in args.zipped_annot_files:
        try:
            assert zip_fn.endswith('.zip'), f'File {fn} is not a zip archive'
            dataset_name = os.path.basename(zip_fn)[:-4]
            fn_prefix = os.path.join(args.output_dir, dataset_name)

            if args.overwrite or not os.path.exists(f'{fn_prefix}.json'):
                if args.verbose:
                    print('UNPACKING', zip_fn)
                with TemporaryDirectory(dir=args.output_dir) as tmp_dir:
                    if args.copy_to_tmp:
                        temp_zip_fn = os.path.join(tmp_dir, os.path.basename(zip_fn))
                        shutil.copy(zip_fn, temp_zip_fn)
                        zip_fn = temp_zip_fn
                    shutil.unpack_archive(zip_fn, extract_dir=tmp_dir, format='zip')
                    unpacked_fn = os.path.join(tmp_dir, 'annotations', 'instances_default.json')
                    assert os.path.exists(unpacked_fn), \
                        f'Archive did not contain annotations{os.path.sep}instances_default.json'
                    shutil.move(unpacked_fn, f'{fn_prefix}.json')
                dataset = utils.read_json(f'{fn_prefix}.json', verbose=args.verbose > 0)

                if args.remove_img_without_annot:
                    utils.remove_empty(dataset, info=dataset_name, verbose=args.verbose > 0)

                if not args.do_not_fix_frame_names:
                    offset = 0
                    for i, img in enumerate(dataset['images']):
                        # format: frame_0000001
                        frame_str = utils.get_frame_string_id(img['file_name'])
                        frame_num = frame_str.split('_')[1]
                        n_digits_fn = max(6, len(frame_num))
                        frame_num = int(frame_num)
                        if frame_num == 0:
                            assert i == 0, 'Found frame 0 but it was not the first one'
                            offset = 1
                        frame_str = f'frame_{frame_num + offset:0{n_digits_fn}d}'
                        if args.verbose == 2:
                            print('before:', img['file_name'])
                        img['file_name'] = f'{dataset_name}.{frame_str}.jpg'
                        if args.verbose == 2:
                            print('after:', img['file_name'])

                utils.save_json(dataset, f'{fn_prefix}.json')
                if args.verbose:
                    print(f'CREATED {fn_prefix}.json\n')
            else:
                if args.verbose:
                    print(f'SKIPPING {fn_prefix}.json already exists')
        except AssertionError as e:
            print(f'ERROR: {e}\n', file=sys.stderr)


if __name__ == '__main__':
    main()
