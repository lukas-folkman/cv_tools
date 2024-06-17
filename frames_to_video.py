import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_fn', required=True)
    parser.add_argument('--ext', default='jpg')
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--fourcc')
    args = parser.parse_args()

    if args.ext.startswith('.'):
        args.ext = args.ext[1:]

    filenames = [os.path.join(args.input_dir, fn) for fn in sorted(os.listdir(args.input_dir))
                 if fn.endswith(f'.{args.ext}')]

    if args.width is None or args.height is None:
        import cv2
        shape = cv2.imread(filenames[0]).shape
        if args.height is None:
            args.height = shape[0]
        if args.width is None:
            args.width = shape[1]
        print(f'width: {args.width}')
        print(f'height: {args.height}')

    import utils
    utils.frames_to_video(
        filenames=filenames, output_fn=args.output_fn,
        fps=args.fps, width=args.width, height=args.height, fourcc=args.fourcc)


if __name__ == '__main__':
    main()
