import os
import argparse
import pandas as pd
import numpy as np
import vent_utils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fn', required=True,
                        help='')
    parser.add_argument('--output_dir', required=True,
                        help='Specify the output directory.')
    parser.add_argument('--model_cat_names', required=True, nargs='+',
                        help='List of categories for the model (in the correct order). Categories and their order must '
                             'match the categories used for training the model. A JSON file (e.g. training data file) '
                             'can be supplied instead.')
    parser.add_argument('--fps', type=int, required=True,
                        help='Frames per second.')

    parser.add_argument('--drop_DJ_sequence', type=int,
                        help='Drop fish if DJ sequence is as long or longer than this number')
    parser.add_argument('--drop_DJ_fraction', type=float, default=0.5,
                        help='Drop fish if DJ fraction is larger than this')
    parser.add_argument('--n_impute_randomly', type=int, default=1,
                        help='Number of missed detections to impute with random choice of neighbouring detections')
    parser.add_argument('--fix_early_open_within_closed', type=int, default=1,
                        help='Before treating singletons, turn closed-open-closed into closed-closed-closed. '
                        'This parameter how many "open" detections nested within "closed" to convert.')
    parser.add_argument('--fix_early_conf_thr', type=float,
                        help='Only apply "fix_early_open_within_closed" for confidence scores less or equal this value.')
    parser.add_argument('--singleton_size', type=int, default=1,
                        help='Drop singletons of this length.')
    parser.add_argument('--singleton_keep_conf_thr', type=float,
                        help='Keep singletons with confidence score higher than this value.')
    parser.add_argument('--impute_singletons', action='store_true',
                        help='Instead of dropping, convert "open" to "closed" and vice versa if it is a singleton.')

    parser.add_argument('--size_filter', type=int,
                        help='Only fish with heads larger than this area')
    parser.add_argument('--conf_filter', type=float,
                        help='Only fish with confidence score larger than this area')
    parser.add_argument('--ram_to_buccal_ratio', type=int,
                        help='Only ram ventilators that were observed for N * open-closed cycle duration.')
    args = parser.parse_args()

    if os.path.exists('utils.py'):
        import utils
    else:
        import cv_tools as utils
    from datetime import datetime

    try:
        assert args.drop_DJ_fraction is None or 0 <= args.drop_DJ_fraction <= 1, '"drop_DJ_fraction" must be greater or equal to 0 and less or equal to 1'
        assert args.fix_early_conf_thr is None or 0 <= args.fix_early_conf_thr <= 1, '"fix_early_conf_thr" must be greater or equal to 0 and less or equal to 1'
        assert args.singleton_keep_conf_thr is None or 0 <= args.singleton_keep_conf_thr <= 1, '"singleton_keep_conf_thr" must be greater or equal to 0 and less or equal to 1'
        assert args.conf_filter is None or 0 <= args.conf_filter <= 1, '"conf_filter" must be greater or equal to 0 and less or equal to 1'

        if args.model_cat_names is not None \
                and len(args.model_cat_names) == 1 \
                and utils.file_or_gzip_exists(args.model_cat_names[0]):
            args.model_cat_names = utils.load_model_cat_names_from_file(args.model_cat_names[0])
            print(f'INFO: Loaded model\'s categories: {args.model_cat_names}')

        _err_msg = 'The input must be one JSON or CSV file or one directory containing JSON/CSV files.'

        if os.path.isdir(args.input_fn):
            # csv_inputs = utils.read_files_from_dir(
            #     args.input_fn,
            #     filter_func=lambda x: x.lower().endswith('.csv'),
            #     basename_only=False)
            # csv_names = [fn[:-4] for fn in csv_inputs]
            # gz_csv_inputs = utils.read_files_from_dir(
            #     args.input_fn,
            #     filter_func=lambda x: x.lower().endswith('.csv.gz'),
            #     basename_only=False)
            # gz_csv_names = [fn[:-7] for fn in gz_csv_inputs]
            json_inputs = utils.read_files_from_dir(
                args.input_fn,
                filter_func=lambda x: x.lower().endswith('.json'),
                basename_only=False)
            json_names = [fn[:-5] for fn in json_inputs]
            gz_json_inputs = utils.read_files_from_dir(
                args.input_fn,
                filter_func=lambda x: x.lower().endswith('.json.gz'),
                basename_only=False)
            gz_json_names = [fn[:-8] for fn in gz_json_inputs]

            # gz_csv_inputs = [fn for i, fn in enumerate(gz_csv_inputs) if gz_csv_names[i] not in csv_names]
            # json_inputs = [fn for i, fn in enumerate(json_inputs) if json_names[i] not in csv_names + gz_csv_names]
            gz_json_inputs = [fn for i, fn in enumerate(gz_json_inputs) if gz_json_names[i] not in json_names] # + csv_names + gz_csv_names
            inputs = json_inputs + gz_json_inputs

        elif os.path.exists(args.input_fn):
            if args.input_fn.lower().endswith('.json') or args.input_fn.lower().endswith('.json.gz') \
                    or args.input_fn.lower().endswith('.csv') or args.input_fn.lower().endswith('.csv.gz'):
                inputs = [args.input_fn]
            else:
                assert False, _err_msg

        else:
            assert os.path.exists(args.input_fn), f'The input "{args.input_fn}" does not exist.'
            assert False, _err_msg

        print(f'INFO: these are the inputs: {inputs}')

    except AssertionError as e:
        parser.error(str(e))

    print(f'\nSTARTED: {datetime.now().strftime("%d %B %Y, %H:%M:%S")}')
    EXTS = ['.csv', '.csv.gz', '.json', '.json.gz']
    EXT_LENS = np.asarray([len(ext) for ext in EXTS])
    for fn in inputs:
        print(fn)
        video_name = os.path.basename(fn)[:-EXT_LENS[[fn.lower().endswith(ext) for ext in EXTS]][0]]
        print(video_name)
        if fn.lower().endswith('.csv') or fn.lower().endswith('.csv.gz'):
            tracks_df = pd.read_csv(fn)
        else:
            tracks_df = utils.predictions_to_df(pred_fn=fn, categories=args.model_cat_names)
        vent_df_no_nulls, fish_qual_df, _ = vent_utils.simply_process_tracks(
            tracks_df=tracks_df,
            drop_DJ_sequence=args.drop_DJ_sequence,
            drop_DJ_fraction=args.drop_DJ_fraction,
            n_impute_randomly=args.n_impute_randomly,
            fix_early_open_within_closed=args.fix_early_open_within_closed,
            fix_early_conf_thr=args.fix_early_conf_thr,
            singleton_size=args.singleton_size,
            singleton_keep_conf_thr=args.singleton_keep_conf_thr,
            impute_singletons=args.impute_singletons,
            other_cols='score',
            status_col='label',
            fps_mod=None,
            random_state=42
        )
        vent_df_no_nulls.to_csv(os.path.join(args.output_dir, f'{video_name}.vent_seq.csv.gz'))
        fish_qual_df.to_csv(os.path.join(args.output_dir, f'{video_name}.fish_qual.csv.gz'))

        vent_rates, vent_lengths, buccal_df, ram_df, _, _ = vent_utils.get_true_buccal_and_ram_estimates(
            vent_df_no_nulls=vent_df_no_nulls,
            fish_qual_df=fish_qual_df,
            fps=args.fps,
            keep_singles=False,
            keep_long_flanks=False,
            size_filter=args.size_filter,
            conf_filter=args.conf_filter,
            filter_true_long_rams=True,
            ram_to_buccal_ratio=args.ram_to_buccal_ratio
        )
        vent_rates.to_csv(os.path.join(args.output_dir, f'{video_name}.vent_rates.csv.gz'))
        vent_lengths.to_csv(os.path.join(args.output_dir, f'{video_name}.open_closed_cycle_duration.csv.gz'))
        buccal_df.to_csv(os.path.join(args.output_dir, f'{video_name}.vent_seq.buccal.csv.gz'))
        ram_df.to_csv(os.path.join(args.output_dir, f'{video_name}.vent_seq.ram.csv.gz'))

    print(f'FINISHED: {datetime.now().strftime("%d %B %Y, %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
