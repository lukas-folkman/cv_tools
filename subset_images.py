import argparse
import utils
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--img_list', required=True)
    parser.add_argument('--output_fn', required=True)
    parser.add_argument('--check_order', action='store_true')
    args = parser.parse_args()

    dataset = utils.read_json(args.annot_fn)
    img_list = [img['file_name'] for img in utils.read_json(args.img_list)['images']]

    dataset['images'] = [img for img in dataset['images'] if img['file_name'] in img_list]
    if args.check_order:
        print('Checking the order')
        assert [img['file_name'] for img in dataset['images']] == img_list
    dataset['annotations'] = utils.filter_annotations_with_images(dataset)
    utils.save_json(dataset, args.output_fn)


if __name__ == '__main__':
    main()
