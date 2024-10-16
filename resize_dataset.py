import argparse
import utils
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--short_size', type=int)
    parser.add_argument('--long_size', type=int)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--new_img_dir')
    parser.add_argument('--method', choices=['short_edge', 'long_edge'], default='short_edge')
    parser.add_argument('--do_not_enlarge', action='store_true')
    parser.add_argument('--do_not_write_images', action='store_true')
    parser.add_argument('--out_fn', required=True)
    args = parser.parse_args()

    if args.method == 'long_edge':
        assert args.long_size is not None
        assert args.short_size is None
    else:
        assert args.short_size is not None

    assert args.do_not_write_images or args.new_img_dir is not None

    resize_shape_op = utils.get_resize_long_edge_shape if args.method == 'long_edge' \
        else utils.get_resize_short_edge_shape

    dataset = utils.read_json(args.annot_fn)
    if not args.do_not_write_images:
        os.makedirs(args.new_img_dir, exist_ok=True)
    dataset = utils.resize_dataset(dataset=dataset, img_dir=args.img_dir, new_img_dir=args.new_img_dir,
                                   short_size=args.short_size, long_size=args.long_size,
                                   resize_shape_op=resize_shape_op, do_not_enlarge=args.do_not_enlarge,
                                   do_not_write_images=args.do_not_write_images)
    utils.save_json(dataset=dataset, fn=args.out_fn)


if __name__ == '__main__':
    main()
