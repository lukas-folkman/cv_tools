import argparse
import utils
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--old_json_img_prefix', default='')
    parser.add_argument('--new_json_img_prefix')
    args = parser.parse_args()

    remove_annot_without_img(annot_fn=args.annot_fn, img_dir=args.img_dir,
                             old_json_img_prefix=args.old_json_img_prefix, new_json_img_prefix=args.new_json_img_prefix)


def remove_annot_without_img(annot_fn, img_dir, old_json_img_prefix=None, new_json_img_prefix=None):

    if old_json_img_prefix is None:
        old_json_img_prefix = ''

    dataset = utils.read_json(annot_fn, assert_correct=False)
    imgs = utils.read_files_from_dir(dir_name=img_dir, filter_func=utils.is_image, basename_only=True)

    done = False
    while not done:
        answer = input(f'Remove PERMANENTLY annotations without images? [Yes/No]: ')
        if answer.lower() in ["y", "yes"]:
            dataset = utils.subset_dataset_to_imgs(dataset, *[os.path.join(old_json_img_prefix, img) for img in imgs])
            if new_json_img_prefix is not None:
                for img in dataset['images']:
                    img['file_name'] = os.path.join(new_json_img_prefix, os.path.basename(img['file_name']))
            if len(dataset['images']) == 0:
                raise ValueError('No images left, considering setting "old_json_img_prefix"')
            utils.save_json(dataset=dataset, fn=annot_fn)
            done = True
        elif answer.lower() in ["n", "no"]:
            done = True
        else:
            print('Wrong input.')
            done = False


if __name__ == '__main__':
    main()
