import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--img_dir')
    parser.add_argument('--only_json', action='store_true')
    parser.add_argument('--remove_imgs_without_this_label', nargs='+')
    args = parser.parse_args()

    if not args.only_json and args.img_dir is None:
        parser.error(f'Specify "img_dir"')

    remove_img_without_annot(annot_fn=args.annot_fn, img_dir=args.img_dir, only_json=args.only_json,
                             remove_imgs_without_this_label=args.remove_imgs_without_this_label)


def remove_img_without_annot(annot_fn, img_dir, only_json, remove_imgs_without_this_label):
    import utils
    dataset = utils.read_json(annot_fn)
    done = False
    while not done:
        answer = input(f'Remove PERMANENTLY images without annotation? [Yes/No]: ')
        if answer.lower() in ["y", "yes"]:
            if remove_imgs_without_this_label is not None:
                dataset['images'] = utils.filter_images_with_specific_label(dataset, label=remove_imgs_without_this_label)
                dataset['annotations'] = utils.filter_annotations_with_images(dataset)
            else:
                utils.remove_empty(dataset)
            utils.save_json(dataset, annot_fn, assert_correct=False)
            if not only_json:
                annot_imgs = [img['file_name'] for img in dataset['images']]
                dir_imgs = utils.read_files_from_dir(dir_name=img_dir, filter_func=utils.is_image, basename_only=True)
                rm_imgs = [fn for fn in dir_imgs if fn not in annot_imgs]
                for fn in rm_imgs:
                    os.remove(os.path.join(img_dir, fn))
                dir_imgs = utils.read_files_from_dir(dir_name=img_dir, filter_func=utils.is_image, basename_only=True)
                missing_imgs = [fn for fn in annot_imgs if fn not in dir_imgs]
                if len(missing_imgs) != 0:
                    print(f'WARNING: missing some images {missing_imgs}')
            done = True
        elif answer.lower() in ["n", "no"]:
            done = True
        else:
            print('Wrong input.')
            done = False


if __name__ == '__main__':
    main()
