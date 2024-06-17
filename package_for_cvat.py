import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', required=True,
                        help='COCO JSON annotation file')
    parser.add_argument('--img_dir', required=True,
                        help='Directory with images')
    parser.add_argument('--task_name', required=True,
                        help='Task name')
    parser.add_argument('--output_dir', required=True,
                        help='directory to store created CVAT task')
    parser.add_argument('--start_at', type=int,
                        help='Start at frame (inclusive, 1-based)')
    parser.add_argument('--stop_at', type=int,
                        help='Stop at frame (inclusive, 1-based)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle CVAT task')
    parser.add_argument('--zip', action='store_true',
                        help='ZIP CVAT package')
    args = parser.parse_args()

    import os
    import shutil
    import utils

    dataset = utils.read_json(args.input_fn)

    if args.shuffle:
        import numpy as np
        np.random.RandomState(0).shuffle(dataset['images'])

    if args.start_at is not None or args.stop_at is not None:
        dataset['images'] = dataset['images'][max(0, args.start_at - 1):min(len(dataset['images']), args.stop_at)]
        dataset['annotations'] = utils.filter_annotations_with_images(dataset)

    dataset['images'] = sorted(dataset['images'], key=lambda x: x['file_name'])

    frames_fns = [os.path.join(args.img_dir, img['file_name']) for img in dataset['images']]
    labels = utils.get_category_names(dataset)

    args.output_dir = os.path.join(args.output_dir, args.task_name)
    cvat_data_dir = os.path.join(args.output_dir, 'data')
    if os.path.exists(cvat_data_dir):
        shutil.rmtree(cvat_data_dir)
    os.makedirs(cvat_data_dir, exist_ok=True)

    utils.package_for_cvat(task_name=args.task_name, frames_fns=frames_fns, labels=labels, use_cache=False,
                           output_dir=args.output_dir, dataset=dataset)

    for fn in frames_fns:
        shutil.copy(fn, cvat_data_dir)

    if args.zip:
        print('Zipping the CVAT task')
        archived_fn = shutil.make_archive(
            base_name=os.path.join(args.output_dir, args.task_name), format='zip', root_dir=args.output_dir)
    else:
        archived_fn = None

    print(f'{f"Upload {archived_fn}" if args.zip else "Create and upload a ZIP archive"} '
          'into CVAT using "[+] Create from backup" option in the "Tasks" view')


if __name__ == '__main__':
    main()
