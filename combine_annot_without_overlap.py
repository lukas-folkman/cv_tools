import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', required=True,
                        help='Primary annotations file (all annotations will be used)')
    parser.add_argument('--secondary', required=True,
                        help='Secondary annotations file (only non-overlapping with "primary" will be used)')
    parser.add_argument('--output_fn', required=True,
                        help='Output filename')
    parser.add_argument('--iou_thr', default=0.5,
        help='Intersection-over-union threshold used to determine overlap of "primary" and "secondary" annotations')
    parser.add_argument('--conf_thr', type=float, default=0.1,
                        help='Only predictions above "conf_thr" will be kept in "secondary"')
    parser.add_argument('--reduce', action='store_true',
                        help='Remove duplicate predictions from secondary')
    args = parser.parse_args()

    import utils
    primary = utils.read_json(args.primary)
    secondary = utils.read_json(args.secondary)
    for d, source in [(primary, 'manual'), (secondary, 'auto')]:
        for ann in d['annotations']:
            ann['source'] = source

    merged = utils.combine_annot_without_overlap(
        primary, secondary, conf_thr=args.conf_thr, iou_thr=args.iou_thr, reduce=args.reduce)
    utils.save_json(merged, args.output_fn)


if __name__ == '__main__':
    main()
