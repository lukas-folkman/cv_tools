import argparse
import utils
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--old', nargs='+', required=True)
    parser.add_argument('--new', nargs='+', required=True)
    args = parser.parse_args()
    assert len(args.old) == len(args.new)

    dataset = utils.read_json(args.annot_fn)
    rename_dict = {old: new for old, new in zip(args.old, args.new)}

    done = False
    while not done:
        answer = input(f'Rename PERMANENTLY {rename_dict}? [Yes/No]: ')
        if answer.lower() in ["y", "yes"]:
            utils.rename_category(dataset=dataset, rename_dict=rename_dict)
            utils.save_json(dataset=dataset, fn=args.annot_fn)
            done = True
        elif answer.lower() in ["n", "no"]:
            done = True
        else:
            print('Wrong input.')
            done = False


if __name__ == '__main__':
    main()
