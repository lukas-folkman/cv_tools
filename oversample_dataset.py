import argparse
import copy
import utils
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fns', nargs='+', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--copy_number', type=int, default=1)
    parser.add_argument('--delete_label', required=True)
    parser.add_argument('--out_fn', required=True)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    print('Reading the datasets...')
    dataset = utils.read_all_datasets(args.annot_fns)
    dataset_orig = copy.deepcopy(dataset) if args.merge else None
    cat_dict = utils.get_category_names(dataset, lower_case=True, as_dict=True)
    args.delete_label = args.delete_label.lower()
    assert args.delete_label in cat_dict, f'"{args.delete_label}" not a category: {list(cat_dict.keys())}'
    delete_id = cat_dict[args.delete_label]
    bboxes_to_delete = utils.bboxes_for_category(dataset, cat_id=delete_id, as_dict=True)
    annotations = utils.get_annotations_dict(dataset)
    new_img_dir = os.path.join(args.img_dir, f'{args.delete_label}_masked')
    os.makedirs(new_img_dir, exist_ok=True)

    print(f'Masking the "{args.delete_label}" objects...')
    new_imgs = []
    for img in dataset['images']:
        if any([ann['category_id'] != delete_id for ann in annotations[img['id']]]):
            assert os.path.basename(img['file_name']) == img['file_name'],\
                f"Directory-nested filenames are not supported {img['file_name']}"
            new_imgs.append(img)
            filename = os.path.join(args.img_dir, img['file_name'])
            new_basename = f"{utils.basename_without_ext(img['file_name'])}.{args.delete_label}_masked{args.copy_number}.{img['file_name'].split('.')[-1]}"
            new_filename = os.path.join(new_img_dir, new_basename)
            shutil.copyfile(src=filename, dst=new_filename)
            if img['id'] in bboxes_to_delete:
                utils.mask_image(filename=new_filename, boxes_xywh=bboxes_to_delete[img['id']], color='average')
            img['file_name'] = os.path.join(os.path.basename(new_img_dir), new_basename)

    print(f'Removing the "{args.delete_label}" category...')
    utils.remove_category(dataset=dataset, category_id=delete_id, verbose=False)
    dataset['images'] = new_imgs
    dataset['annotations'] = utils.filter_annotations_with_images(dataset)

    if args.merge:
        print(f'Merging all datasets...')
        utils.merge_datasets(dataset_orig, dataset)

    print(f'Saving the dataset...')
    utils.save_json(dataset=dataset, fn=args.out_fn)
    dataset = utils.read_json(fn=args.out_fn)


if __name__ == '__main__':
    main()
