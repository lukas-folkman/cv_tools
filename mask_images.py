import argparse
import shutil
import sys

import numpy as np
import os
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--skip_backup', action='store_true')
    parser.add_argument('--color', nargs='+')
    parser.add_argument('--delete_label', default='delete')
    parser.add_argument('--case_sensitive', action='store_true')
    args = parser.parse_args()

    assert args.color is None or args.color == ['average'] or args.color == ['blur'] or len(args.color) == 3
    if args.color == ['average'] or args.color == ['blur']:
        args.color = args.color[0]

    dataset = utils.read_json(args.annot_fn)
    cat_dict = utils.get_category_names(dataset, lower_case=not args.case_sensitive, as_dict=True)
    cat_id = cat_dict.get(args.delete_label.lower() if not args.case_sensitive else args.delete_label)
    if cat_id is None:
        print(f'No {args.delete_label} category')
        sys.exit()
    if not any([ann['category_id'] == cat_id for ann in dataset['annotations']]):
        print(f'No {args.delete_label} annotations')
        sys.exit()

    kwargs = dict(annot_fn=args.annot_fn, img_dir=args.img_dir, delete_label=args.delete_label,
                  color=args.color, case_sensitive=args.case_sensitive)

    if args.output_dir is None:
        done = False
        while not done:
            answer = input(f'Remove PERMANENTLY annotations with label {args.delete_label} and mask them PERMANENTLY in images? [Yes/No]: ')
            if answer.lower() in ["y", "yes"]:
                if not args.skip_backup:
                    shutil.copyfile(
                        src=args.annot_fn,
                        dst=f'{args.annot_fn[:-5] if args.annot_fn.endswith(".json") else args.annot_fn}.unmasked.json'
                    )
                    anns_per_img = utils.get_annotations_dict(dataset)
                    for img in dataset['images']:
                        if any([ann['category_id'] == cat_id for ann in anns_per_img[img['id']]]):
                            shutil.copyfile(
                                src=os.path.join(args.img_dir, img['file_name']),
                                dst=os.path.join(args.img_dir, f'{img["file_name"][:-4] if img["file_name"].endswith(".jpg") else img["file_name"]}.unmasked.jpg')
                            )
                utils.mask_images(**kwargs)
                done = True
            elif answer.lower() in ["n", "no"]:
                done = True
            else:
                print('Wrong input.')
                done = False
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        utils.mask_images(output_dir=args.output_dir, **kwargs)


if __name__ == '__main__':
    main()
