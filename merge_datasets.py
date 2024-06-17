import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--img_dirs', nargs='+')
    parser.add_argument('--out_fn', required=True)
    parser.add_argument('--split_test_groups', nargs='+')
    parser.add_argument('--group_splitter', default='.frame_')
    args = parser.parse_args()

    if len(args.datasets) < 2:
        parser.error(f'Specify multiple datasets')

    if args.img_dirs is not None and len(args.img_dirs) != 1 and len(args.img_dirs) != len(args.datasets):
        parser.error(f'Specify correct number og "img_dirs"')

    import utils

    merged = utils.read_all_datasets(args.datasets, data_img_dirs=args.img_dirs)
    if args.img_dirs is not None:
        dataset, image_root = merged
        print(f'New image_root: {image_root}')
    else:
        dataset = merged

    utils.save_json(dataset=dataset, fn=args.out_fn)
    utils.read_json(args.out_fn)

    if args.split_test_groups:
        print('\nCreating train/test split...')
        print('Test set groups:', args.split_test_groups)
        train, test = utils.split_groups_into_train_test(
            dataset=dataset, test_groups=args.split_test_groups, group_splitter=args.group_splitter)
        for d, suffix in [
            (train, 'train'),
            (test, 'test'),
        ]:
            print(f'\n{suffix}')
            out_fn = f"{args.out_fn[:-5] if args.out_fn.lower().endswith('json') else out_fn}.{suffix}.json"
            utils.save_json(dataset=d, fn=out_fn)
            utils.read_json(out_fn)
            print(set(utils.get_filename_groups(d['images'], split_char=args.group_splitter)))


if __name__ == '__main__':
    main()
