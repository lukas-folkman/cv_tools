import argparse
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='directory with (jpg) images', required=True)
    parser.add_argument('--voc_dir', help='directory with VOC annotation (xml) files', required=True)
    parser.add_argument('--out_fn', help='output COCO annotation file in json format', required=True)
    parser.add_argument('--fix_voc_filenames', action='store_true',
                        help='try to fix filenames in the xml voc annotations')
    parser.add_argument('--fix_cat_names', action='store_true',
                        help='try to fix category names in Jellytoring dataset in the xml voc annotations')
    parser.add_argument('--sort_cat_names', action='store_true',
                        help='this makes test/train splits have the same cat ids')
    parser.add_argument('--info', help='humanly description of the dataset')
    args = parser.parse_args()

    if args.fix_voc_filenames:
        answer = input(f'You selected "--fix_voc_filenames" which will PERMANENTLY modify '
                       'the filename field of every xml annotation in your dataset, proceed? [Yes/No]: ')
        if answer.lower() not in ["y", "yes"]:
            return -1

    if args.fix_cat_names:
        answer = input(f'You selected "--fix_cat_names" which will PERMANENTLY modify '
                       'the object name field of every xml annotation in your dataset, proceed? [Yes/No]: ')
        if answer.lower() not in ["y", "yes"]:
            return -1

    dataset = utils.voc_to_coco(img_dir=args.img_dir, voc_dir=args.voc_dir, coco_fn=args.out_fn, info=args.info,
                                fix_voc_filenames=args.fix_voc_filenames, fix_cat_names=args.fix_cat_names,
                                sort_cat_names=args.sort_cat_names)


if __name__ == '__main__':
    main()
