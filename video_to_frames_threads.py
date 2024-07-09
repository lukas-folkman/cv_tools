import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', nargs='+', help='filename: video (cannot be used if "input_dir" specified)')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    import queue
    import threading
    import cv2
    q = queue.Queue()

    n_digits_fn = 6
    out_prefix = args.input_fn
    f_number = 0
    zero_based = False

    def read_frames():
        cap = cv2.VideoCapture(args.input_fn)
        ret, frame = cap.read()
        q.put(frame)
        while ret:
            ret, frame = cap.read()
            q.put(frame)

    def save_frames():
        global f_number
        while True:
            if not q.empty():
                frame = q.get()
                cv2.imshow("frame1", frame)
                fn = f"{out_prefix}.frame_{f_number + (0 if zero_based else 1):0{n_digits_fn}d}.jpg"
                cv2.imwrite(fn, frame)
                f_number += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    read_thread = threading.Thread(target=read_frames)
    save_thread = threading.Thread(target=save_frames)
    read_thread.start()
    save_thread.start()


if __name__ == '__main__':
    main()
