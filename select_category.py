import argparse
import utils
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_fn', required=True)
    parser.add_argument('--category_id', type=int, required=True)
    args = parser.parse_args()

    dataset = utils.read_json(args.annot_fn, assert_correct=False)

    done = False
    while not done:
        answer = input(f'Subset PERMANENTLY annotations to category ID {args.category_id}? [Yes/No]: ')
        if answer.lower() in ["y", "yes"]:
            utils.select_category(dataset=dataset, category_id=args.category_id)
            utils.save_json(dataset=dataset, fn=args.annot_fn)
            done = True
        elif answer.lower() in ["n", "no"]:
            done = True
        else:
            print('Wrong input.')
            done = False


if __name__ == '__main__':
    main()
