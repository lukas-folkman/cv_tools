import scipy
import pandas as pd
import numpy as np
import time
import warnings
# warnings.simplefilter('error')

MEDIAN_OPEN = 12
MEDIAN_CLOSED = 6
NULL = 'NULL'
DJ = 'DJ'
OPEN = 'open'
CLOSED = 'closed'


def replace_DJs_with_nulls(s, drop_DJ_fraction, drop_DJ_sequence=None, status_col='label', inplace=True, verbose=False):
    assert inplace, 'Only "inplace" fish sequence replacements are implemented'
    n_DJs = (s[status_col].values == DJ).sum()
    if n_DJs != 0:
        drop_fish_DJ = n_DJs / (~s[status_col].isnull()).sum() >= drop_DJ_fraction
        if not drop_fish_DJ and drop_DJ_sequence is not None:
            DJ_seq = []
            for i, val in s[status_col].items():
                if val == DJ:
                    DJ_seq.append(i)
                else:
                    if len(DJ_seq) >= drop_DJ_sequence:
                        drop_fish_DJ = True
                        break
                    DJ_seq = []
            drop_fish_DJ = drop_fish_DJ or len(DJ_seq) >= drop_DJ_sequence

        if drop_fish_DJ:
            if verbose:
                print(f'Dropping: {n_DJs / (~s[status_col].isnull()).sum()}: {s[status_col].values}')
            s[status_col] = np.nan
            assert pd.isnull(s[status_col]).all()
        else:
            s.loc[s[status_col].values == DJ, status_col] = np.nan


def simply_process_tracks(
        tracks_df,
        drop_DJ_sequence=None,  # if sequence is as long or longer than this number
        drop_DJ_fraction=0.5,  # if the fraction is larger than this
        n_impute_randomly=1,
        fix_early_open_within_closed=1,
        fix_early_conf_thr=None,
        singleton_size=1,
        singleton_keep_conf_thr=None,
        impute_singletons=False,
        other_cols='score',
        status_col='label',
        fps_mod=None,
        random_state=42
):

    assert drop_DJ_sequence is None or drop_DJ_sequence >= 1
    assert 0 <= drop_DJ_fraction <= 1
    for arg in [n_impute_randomly, fix_early_open_within_closed, singleton_size]:
        if arg:
            assert arg == int(arg) and arg >= 0

    if tracks_df['fish_id'].isnull().any():
        print('Removing fish with null ID')
        tracks_df = tracks_df.loc[~tracks_df['fish_id'].isnull()]
    assert not tracks_df['video_id'].isnull().any()
    assert not tracks_df['fish_id'].isnull().any()

    if drop_DJ_sequence is not None:
        print(f'Dropping fish if it contains a DJ-DJ-DJ sequence of {drop_DJ_sequence}')
    if drop_DJ_fraction != 0:
        print(f'Dropping fish if DJ fraction is higher than {drop_DJ_fraction}')
    if n_impute_randomly:
        print(f'Imputing nulls of length {n_impute_randomly} with random choice')
    if fix_early_open_within_closed:
        print(f'Changing closed-open-closed to closed-closed-closed for open of {fix_early_open_within_closed}')
    if singleton_size:
        print(f'Dropping fish with singletons of length of {singleton_size}')

    if fps_mod is not None and fps_mod > 1:
        print(f'Selecting every {fps_mod}{"nd" if fps_mod == 2 else "rd" if fps_mod == 3 else "th"} frame')
        tracks_df = tracks_df.loc[tracks_df['frame_id'] % fps_mod == 0]
        for video_id in tracks_df['video_id'].unique():
            video_mask = tracks_df['video_id'] == video_id
            idx = np.arange(0, tracks_df.loc[video_mask, 'frame_id'].max() + 1, fps_mod, dtype=int)
            assert tracks_df.loc[video_mask, 'frame_id'].values.isin(idx).all()
            tracks_df.loc[video_mask, 'frame_id'] = tracks_df.loc[video_mask, 'frame_id'].map({i: j for i, j in zip(idx, np.arange(len(idx)))})

    processing_times = []
    vent_df_no_nulls, fish_sizes_df, fish_score_df, fish_ram_df, DJs_df = [], [], [], [], []
    for video_id in tracks_df['video_id'].unique():
        for fish_id in tracks_df.loc[tracks_df['video_id'] == video_id, 'fish_id'].unique():
            # select a fish individual
            fish_seq = tracks_df.loc[(tracks_df['video_id'] == video_id) & (tracks_df['fish_id'] == fish_id)]
            # store fish statistics
            fish_sizes_df.append([
                video_id, fish_id, fish_seq['area'].median(), fish_seq['area'].mean(), fish_seq['area'].min(),
                fish_seq['area'].max(), fish_seq['area'].quantile(.25), fish_seq['area'].quantile(.75)
            ])
            if (~pd.isnull(fish_seq['score'])).any():
                fish_score_df.append([
                    video_id, fish_id, fish_seq['score'].median(), fish_seq['score'].mean(), fish_seq['score'].min(),
                    fish_seq['score'].max(), fish_seq['score'].quantile(.25), fish_seq['score'].quantile(.75)
                ])
            else:
                fish_score_df.append([video_id, fish_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            # FINALLY, start working towards estimating open/closed durations
            # create a correctly chronologically sorted RangeIndex and return pd.Series on "status_col" (label)
            t0 = time.perf_counter()
            fish_seq = get_vent_sequence(fish_seq, status_col=status_col)
            # replace DJs with nulls
            replace_DJs_with_nulls(
                fish_seq, drop_DJ_fraction=drop_DJ_fraction, drop_DJ_sequence=drop_DJ_sequence,
                status_col=status_col, inplace=True)
            # check if the fish is not to be dropped due to high DJ occurrence
            if not fish_seq[status_col].isnull().all():
                # Removing flanking nulls and impute
                cleanup_and_impute_vent_sequence(
                    fish_seq, n_impute_randomly=n_impute_randomly,
                    fix_early_open_within_closed=fix_early_open_within_closed, fix_conf_thr=fix_early_conf_thr,
                    score_col='score', remove_missing_flanks=True, return_splits=False, status_col=status_col,
                    inplace=True, random_state=random_state)
                # merge consecutive open, closed, and NULL events (create a sequence "summary")
                fish_seq_df = vent_sequence_summary(
                    fish_seq, video_id=video_id, fish_id=fish_id, other_cols=other_cols)
                # keep a list of candidate ram ventilators (no "closed" state observed)
                potential_ram = not (fish_seq_df[status_col] == CLOSED).any()
                # take the longest sequence with no nulls (optionally drop sequences with open-closed-open)
                fish_seq_df = get_longest_non_null_sequence(
                    fish_seq_df, singleton_size=singleton_size, conf_thr=singleton_keep_conf_thr,
                    impute=impute_singletons, score_col='mean_score', status_col=status_col)
                if fish_seq_df is not None:
                    vent_df_no_nulls.append(fish_seq_df)
                    fish_ram_df.append([video_id, fish_id, potential_ram])
                DJs_df.append([video_id, fish_id, False])
            else:
                DJs_df.append([video_id, fish_id, True])
            processing_times.append(time.perf_counter() - t0)

    # concatenate across all fish and all videos
    vent_df_no_nulls = pd.concat(vent_df_no_nulls, axis=0)

    # concatenate fish_qual_df
    fish_sizes_df = pd.DataFrame.from_records(
        fish_sizes_df, columns=['video_id', 'fish_id', 'median', 'mean', 'min', 'max', 'q25', 'q75']
    ).set_index(['video_id', 'fish_id'])
    fish_sizes_df.columns = fish_sizes_df.columns + '_size'
    fish_score_df = pd.DataFrame.from_records(
        fish_score_df, columns=['video_id', 'fish_id', 'median', 'mean', 'min', 'max', 'q25', 'q75']
    ).set_index(['video_id', 'fish_id'])
    fish_score_df.columns = fish_score_df.columns + '_score'
    fish_ram_df = pd.DataFrame.from_records(
        fish_ram_df, columns=['video_id', 'fish_id', 'NO_closed_mouths']
    ).set_index(['video_id', 'fish_id'])
    DJs_df = pd.DataFrame.from_records(
        DJs_df, columns=['video_id', 'fish_id', 'DJ']
    ).set_index(['video_id', 'fish_id'])
    fish_qual_df = pd.concat((DJs_df, fish_ram_df, fish_sizes_df, fish_score_df), axis=1)

    return vent_df_no_nulls, fish_qual_df, processing_times


def split_ram_and_buccal(vent_df):
    is_ram = vent_df.groupby(['video_id', 'fish_id']).apply(lambda x: (x['label'] == OPEN).all())
    ram_df = vent_df.reset_index().set_index(['video_id', 'fish_id']).loc[is_ram.index[is_ram]]
    assert (ram_df.groupby(['video_id', 'fish_id'])['label'].count() == 1).all()
    ram_df = ram_df.reset_index().set_index(['video_id', 'fish_id', 'change_id'])
    buccal_df = vent_df.loc[~vent_df.index.isin(ram_df.index)]
    return ram_df, buccal_df


def fish_with_singletons(df, singleton_size=1, conf_thr=None, score_col='mean_score', status_col='label',
                         return_singletons_idx=False):
    df = remove_flanks(df)
    if score_col not in df.columns:
        df[score_col] = np.nan
    singletons = df.index[
        df[status_col].isin([OPEN, CLOSED]) &
        (df['size'] <= singleton_size) &
        (df[score_col].isnull() | (df[score_col] <= conf_thr))
        ]
    return singletons if return_singletons_idx else len(singletons) != 0


def fix_singletons(current, singleton_size=1, conf_thr=None, impute=False, score_col='mean_score', status_col='label'):
    assert isinstance(current, list)
    if singleton_size is None:
        singleton_size = 0
    if conf_thr is None:
        conf_thr = 1
    current = pd.concat(current, axis=1).T.rename_axis(['video_id', 'fish_id', 'change_id'])
    singletons = fish_with_singletons(
        current, singleton_size=singleton_size, conf_thr=conf_thr, score_col=score_col, status_col=status_col,
        return_singletons_idx=True
    )

    if len(singletons) != 0:
        if impute:
            current.loc[singletons, status_col] = current.loc[singletons, status_col].apply(
                lambda x: CLOSED if x == OPEN else OPEN if x == CLOSED else 'ERROR')
            assert current.loc[singletons, status_col].isin([OPEN, CLOSED]).all()
        else:
            # drop
            current = []
    else:
        # no action
        pass

    return current


def get_longest_non_null_sequence(vent_df, singleton_size=None, conf_thr=None, impute=False, score_col='mean_score', status_col='label'):
    assert_vent_df_correct(vent_df, status_col=status_col)
    assert_one_fish_vent_df(vent_df)
    assert conf_thr is None or (vent_df.loc[~vent_df[score_col].isnull(), score_col] <= 1).all()
    assert (vent_df['size'] >= 1).all()

    current, longest = [], []
    for _, row in vent_df.iterrows():
        if row[status_col] != NULL:
            current.append(row)
        else:
            current = fix_singletons(current, singleton_size=singleton_size, conf_thr=conf_thr,
                                     impute=impute, score_col=score_col, status_col=status_col)
            if len(current) > len(longest) or \
                    (score_col in current and len(current) != 0 and len(current) == len(longest) and current[score_col].mean() > longest[score_col].mean()):
                longest = current
            current = []

    current = fix_singletons(current, singleton_size=singleton_size, conf_thr=conf_thr,
                             impute=impute, score_col=score_col, status_col=status_col)

    if len(current) > len(longest) or \
            (score_col in current and len(current) != 0 and len(current) == len(longest) and current[score_col].mean() > longest[score_col].mean()):
        longest = current

    if len(longest) == 0:
        vent_df = None
    else:
        vent_df = longest.rename_axis(vent_df.index.names).reset_index().drop('change_id', axis=1)
        s_changes = (vent_df[status_col] != vent_df[status_col].shift(1)).cumsum().rename('change_id')
        s_sizes = pd.concat((vent_df, s_changes), axis=1).groupby(['video_id', 'fish_id', 'change_id', status_col])['size'].sum()
        if 'min_score' in vent_df:
            min_scores = pd.concat((vent_df, s_changes), axis=1).groupby(['video_id', 'fish_id', 'change_id', status_col])['min_score'].min()
            vent_df = pd.concat((s_sizes.to_frame(), min_scores.to_frame()), axis=1).reset_index()
        else:
            vent_df = s_sizes.to_frame().reset_index()
        vent_df = vent_df.set_index(['video_id', 'fish_id', 'change_id'])

    return vent_df


def read_vent_df(fn):
    vent_df = pd.read_csv(fn, index_col=['video_id', 'fish_id', 'change_id'])
    assert_vent_df_correct(vent_df)
    return vent_df


def get_vent_sequence(s, df=None, video_id=None, fish_id=None, frame_col='frame_id', status_col='label'):
    if s is None:
        s = df.loc[(df['video_id'] == video_id) & (df['fish_id'] == fish_id)]
    s = s.sort_values(frame_col).set_index(frame_col)[[status_col, 'score']]
    s = s.reindex(pd.RangeIndex(s.index[0], s.index[-1] + 1))
    return s


def cleanup_and_impute_vent_sequence(s, n_impute_randomly=0, fix_early_open_within_closed=0, fix_conf_thr=None, score_col='score',
                                     remove_missing_flanks=True, return_splits=False, status_col='label', inplace=True, random_state=42):
    assert inplace, 'Only "inplace" fish sequence replacements are implemented'
    assert (s.index == pd.RangeIndex(s.index[0], s.index[-1] + 1)).all()
    if remove_missing_flanks:
        while len(s) != 0 and s[status_col].iloc[[0, -1]].isnull().any():
            if pd.isnull(s[status_col].iloc[0]):
                s.drop(s.index[0], inplace=True)
            if len(s) != 0 and pd.isnull(s[status_col].iloc[-1]):
                s.drop(s.index[-1], inplace=True)

    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if n_impute_randomly:
        null_seq, continuous, last_before_null = [], [], None
        if (~s[status_col].isnull()).any():
            for i, val in s[status_col].items():
                if pd.isnull(val):
                    if len(null_seq) == 0:
                        last_before_null = s.loc[i - 1, status_col]
                    null_seq.append(i)
                else:
                    if len(null_seq) > 0:
                        if len(null_seq) <= n_impute_randomly:
                            imputed = random_state.choice([val, last_before_null], size=1).tolist() * len(null_seq)
                            s.loc[null_seq, status_col] = imputed
                            continuous.extend(imputed)
                        else:
                            continuous = []
                    null_seq = []
                    continuous.append(val)

            if 0 < len(null_seq) <= n_impute_randomly:
                assert not remove_missing_flanks
                imputed = [last_before_null] * len(null_seq)
                s.loc[null_seq, status_col] = imputed
                continuous.extend(imputed)

    if fix_early_open_within_closed:
        if (s[status_col] == CLOSED).any() and (s[status_col] == OPEN).any():
            open_seq = []
            for i, val in s[status_col].items():
                if val == OPEN:
                    if len(open_seq) != 0 or (i > s.index[0] and s.loc[i - 1, status_col] == CLOSED):
                        open_seq.append(i)
                else:
                    if val == CLOSED and 0 < len(open_seq) <= fix_early_open_within_closed:
                        assert s.loc[open_seq[0] - 1, status_col] == CLOSED
                        assert s.loc[open_seq[-1] + 1, status_col] == CLOSED
                        assert (s.loc[open_seq, status_col] == OPEN).all()
                        # print(s.loc[open_seq, 'score'].iloc[0])
                        if fix_conf_thr is None or s.loc[open_seq, score_col].mean() <= fix_conf_thr:
                            s.loc[open_seq, status_col] = CLOSED
                    open_seq = []

    if return_splits:
        split_s, null_seq, continuous = [], [], []
        if (~s[status_col].isnull()).any():
            for i, val in s[status_col].items():
                if pd.isnull(val):
                    null_seq.append(i)
                else:
                    if len(null_seq) > 0:
                        split_s.append(
                            pd.Series(continuous, index=pd.RangeIndex(null_seq[0] - len(continuous), null_seq[0]))
                        )
                        continuous = []
                    null_seq = []
                    continuous.append(val)
            split_s.append(
                pd.Series(continuous, index=pd.RangeIndex(s.index[-1] - len(continuous) + 1, s.index[-1] + 1))
            )

    return s, split_s if return_splits else None


def remove_flanks(vent_df, flanks=None, keep_long_flanks=False, status_col='label', strict=False):
    assert_vent_df_correct(vent_df, status_col=None, strict=strict)
    if flanks is None:
        flanks = ['min', 'max']
    elif isinstance(flanks, str):
        flanks = [flanks]
    assert flanks in [['min', 'max'], ['min'], ['max']]

    if keep_long_flanks:
        keep_thrs = {}
        df = remove_flanks(vent_df)
        for video_id in df.index.get_level_values('video_id').unique():
            keep_thrs[video_id] = {}
            vdf = df.loc[df.index.get_level_values('video_id') == video_id]
            for status in [OPEN, CLOSED]:
                keep_thrs[video_id][status] = vdf.loc[vdf[status_col] == status, 'size'].quantile(.75)

    for flank in flanks:
        idx_to_drop = get_flank_ids(vent_df, flank=flank)
        if keep_long_flanks:
            drop_df = vent_df.loc[idx_to_drop]
            idx_to_drop = []
            for video_id in drop_df.index.get_level_values('video_id').unique():
                for status in [OPEN, CLOSED]:
                    thr = keep_thrs.get(video_id)
                    if thr is not None:
                        thr = thr.get(status)
                    if thr is None:
                        thr = 1e6
                    idx_to_drop.extend(
                        drop_df.loc[
                            (drop_df.index.get_level_values('video_id') == video_id) &
                            (drop_df[status_col] == status) &
                            (drop_df['size'] < thr)
                        ].index.tolist()
                    )

        vent_df = vent_df.drop(idx_to_drop)
    return vent_df


def get_flank_ids(vent_df, flank):
    assert flank in ['min', 'max']
    grouped = vent_df.reset_index().groupby(['video_id', 'fish_id'])['change_id']
    reduced = grouped.min() if flank == 'min' else grouped.max() if flank == 'max' else None
    return pd.MultiIndex.from_frame(reduced.to_frame().reset_index())


def vent_sequence_summary(s, video_id, fish_id, status_col='label', other_cols='score'):
    if isinstance(other_cols, str):
        other_cols = [other_cols]
    assert set(s.loc[~s[status_col].isnull(), status_col]).issubset([OPEN, CLOSED]), set(s[status_col])
    statuses = s[status_col].copy().fillna(NULL)
    assert isinstance(statuses.index, pd.RangeIndex) and statuses.index.step == 1
    s_changes = (statuses != statuses.shift(1)).cumsum().rename('change_id')
    s_sizes = pd.concat((statuses, s_changes), axis=1).groupby([status_col, 'change_id']).size()
    if other_cols and (~pd.isnull(s[other_cols])).any().any():
        grouped_other_cols = pd.concat(
            (statuses, s_changes, s[other_cols]), axis=1
        ).groupby([status_col, 'change_id'])[other_cols]

        vent_df = pd.concat((
            s_sizes.rename('size').to_frame(),
            grouped_other_cols.mean().rename(lambda x: f'mean_{x}', axis=1),
            grouped_other_cols.quantile(q=0.25).rename(lambda x: f'q25_{x}', axis=1),
            grouped_other_cols.min().rename(lambda x: f'min_{x}', axis=1),
        ), axis=1).reset_index()
    else:
        vent_df = s_sizes.rename('size').to_frame().reset_index()
    vent_df['fish_id'] = fish_id
    vent_df['video_id'] = video_id
    vent_df = vent_df.set_index(['video_id', 'fish_id', 'change_id']).sort_index()

    return vent_df


def assert_one_fish_vent_df(vent_df):
    assert 'video_id' not in vent_df.index.names or len(vent_df.index.get_level_values('video_id').unique()) == 1
    assert 'fish_id' not in vent_df.index.names or len(vent_df.index.get_level_values('fish_id').unique()) == 1


def assert_vent_df_correct(vent_df, video_id=None, fish_id=None, status_col='label', strict=True):
    assert not strict or vent_df.index.is_unique
    if status_col is not None:
        assert set(vent_df.columns).issuperset([status_col, 'size'])
    if video_id is None and fish_id is None:
        assert set(vent_df.index.names) in [{'video_id', 'fish_id', 'change_id'}, {'video_id', 'change_id'},
                                       {'fish_id', 'change_id'}, {'change_id'}]
    elif video_id is not None:
        assert set(vent_df.index.names) in [{'video_id', 'fish_id', 'change_id'}, {'video_id', 'change_id'}]
    elif fish_id is not None:
        assert set(vent_df.index.names) in [{'video_id', 'fish_id', 'change_id'}, {'fish_id', 'change_id'}]


def select_fish_vent_profile(vent_df, video_id=None, fish_id=None, status_col='label'):
    assert_vent_df_correct(vent_df=vent_df, video_id=video_id, fish_id=fish_id, status_col=status_col)
    if video_id is not None:
        vent_df = vent_df.loc[vent_df.index.get_level_values('video_id') == video_id]
    if fish_id is not None:
        vent_df = vent_df.loc[vent_df.index.get_level_values('fish_id') == fish_id]
    if 'change_id' in vent_df.index.names:
        assert vent_df.index.get_level_values('change_id').is_unique
    if 'change_id' in vent_df.columns:
        assert len(vent_df['change_id']) == len(vent_df['change_id'].unique())
    return vent_df


def subset_vent_df_based_on_size(vent_df, fish_sizes_df, only_larger_than):
    vent_df = vent_df.reset_index().set_index(['video_id', 'fish_id'])
    vent_df = vent_df.loc[fish_sizes_df.index[fish_sizes_df['median'] > only_larger_than].intersection(vent_df.index)]
    return vent_df.reset_index().set_index(['video_id', 'fish_id', 'change_id'])


def convert_to_vent_rates(vent_lengths, fps):
    return ((60 * fps) / vent_lengths).rename('vent_rate')


def get_average_vent_length(vent_df, estimator='mean', paired=True, per_status=False, per_fish=True,
                            remove_flanking=False, keep_long_flanks=False, keep_singles=False,
                            impute_missing_pairs_dict=None, status_col='label'):
    assert estimator in ['median', 'mean']
    assert not (keep_singles and remove_flanking)
    assert_vent_df_correct(vent_df, video_id=None, fish_id='', status_col=status_col)

    video_id_col = ['video_id'] if 'video_id' in vent_df.index.names else []

    if remove_flanking:
        vent_df = remove_flanks(vent_df, keep_long_flanks=keep_long_flanks, status_col=status_col)

    if paired:
        assert not per_status
        assert vent_df[status_col].isin({OPEN, CLOSED}).all(), \
            'Use "vent_df_no_nulls" instead of "vent_df" when "paired=True"'
        vent_df = merge_adjoint_open_closed_episodes(
            vent_df,
            keep_singles=keep_singles,
            impute_missing_pairs_dict=impute_missing_pairs_dict,
            status_col=status_col
        )
        grouped = vent_df.groupby(video_id_col + ['fish_id'])['size']
        reduced = grouped.median() if estimator == 'median' else grouped.mean() if estimator == 'mean' else None
        if not per_fish:
            grouped = reduced.groupby(video_id_col) if video_id_col else reduced
            reduced = grouped.median() if estimator == 'median' else grouped.mean() if estimator == 'mean' else None

    else:
        assert keep_singles, f'ERROR: keep_singles={keep_singles} is not implemented for paired={paired}'
        if (~vent_df[status_col].isin({OPEN, CLOSED})).any():
            print('WARNING: Ignoring NULL and DJ labels')
        vent_df = vent_df.loc[vent_df[status_col].isin({OPEN, CLOSED})]

        grouped = vent_df.groupby(video_id_col + ['fish_id', status_col])['size']
        reduced = grouped.median() if estimator == 'median' else grouped.mean() if estimator == 'mean' else None
        if not per_fish:
            grouped = reduced.groupby(video_id_col + [status_col])
            reduced = grouped.median() if estimator == 'median' else grouped.mean() if estimator == 'mean' else None

        if not per_status:
            group_by = video_id_col + (['fish_id'] if per_fish else [])
            grouped = reduced.groupby(group_by) if group_by else reduced
            reduced = grouped.sum()

    return reduced.rename('duration')


def merge_adjoint_open_closed_episodes(vent_df, keep_singles=True, impute_missing_pairs_dict=None, status_col='label'):
    paired_df = []
    for video_id in vent_df.index.get_level_values('video_id').unique() if 'video_id' in vent_df.index.names else [None]:
        video_df = vent_df.loc[vent_df.index.get_level_values('video_id') == video_id] if video_id else vent_df
        for fish_id in video_df.index.get_level_values('fish_id').unique():
            fish_df = select_fish_vent_profile(vent_df, video_id=video_id, fish_id=fish_id, status_col=status_col)

            if len(fish_df) > 2 and len(fish_df) % 2 == 1 \
                    and fish_df[status_col].iloc[0] == fish_df[status_col].iloc[-1]:
                first_idx = fish_df.iloc[0].name
                last_idx = fish_df.iloc[-1].name
                drop_idx = last_idx if fish_df.loc[first_idx, 'size'] >= fish_df.loc[last_idx, 'size'] else first_idx
                fish_df = fish_df.drop(drop_idx)

            assert len(fish_df) != 0 and (len(fish_df) % 2 == 0 or len(fish_df) == 1)

            if len(fish_df) == 1:
                if keep_singles:
                    if impute_missing_pairs_dict is not None:
                        missing_status = OPEN if fish_df['label'].iloc[0] == CLOSED else CLOSED
                        missing_status_size = impute_missing_pairs_dict[video_id][missing_status]
                    else:
                        missing_status_size = 0
                    paired = [fish_df['size'].iloc[0] + missing_status_size]
                else:
                    paired = None
            else:
                assert len(fish_df) != 0 and len(fish_df) % 2 == 0
                assert fish_df[status_col].isin({OPEN, CLOSED}).all(), fish_df
                previous, pair, paired = None, [], []
                for idx, row in fish_df.iterrows():
                    assert previous is None or previous != row[status_col]
                    previous = row[status_col]
                    pair.append(row)
                    assert len(pair) in [1, 2]
                    if len(pair) == 2:
                        paired.append(pair[0]['size'] + pair[1]['size'])
                        pair = []
                assert len(pair) == 0

            if paired is not None:
                paired_df.append(pd.Series(paired, index=pd.MultiIndex.from_arrays(
                    ([video_id] * len(paired), [fish_id] * len(paired)), names=['video_id', 'fish_id'])))

    paired_df = pd.concat(paired_df).rename('size').to_frame()
    return paired_df


def test_vent_rates(vent_rates, videos1, videos2, p_thr=0.05, test_func='mann-whitney', paired=False, verbose=True, check_normality=True):
    if isinstance(videos1, str):
        videos1 = [videos1]
    if isinstance(videos2, str):
        videos2 = [videos2]
    if isinstance(test_func, str):
        if test_func.lower() in ['t', 't-test', 't_test', 't test']:
            test_func = scipy.stats.ttest_ind if not paired else scipy.stats.ttest_rel
        elif test_func.lower() in ['u', 'u-test', 'u_test', 'u test', 'mann-whitney u', 'mann-whitney', 'wilcoxon']:
            test_func = scipy.stats.mannwhitneyu if not paired else scipy.stats.wilcoxon
        elif test_func.lower() in ['kruskal']:
            assert not paired
            test_func = scipy.stats.kruskal
        else:
            raise ValueError(f'Unknown test_func "{test_func}"')
    test_kwargs = dict(alternative='two-sided') if test_func != scipy.stats.kruskal else dict()

    videos1_vent_rates = vent_rates[vent_rates.index.get_level_values('video_id').isin(videos1)].values
    videos2_vent_rates = vent_rates[vent_rates.index.get_level_values('video_id').isin(videos2)].values
    if len(videos1_vent_rates) < 3 or len(videos2_vent_rates) < 3:
        return videos1_vent_rates, videos2_vent_rates, np.nan, np.nan, None

    if check_normality:
        for v, vr in [(videos1, videos1_vent_rates), (videos2, videos2_vent_rates)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                is_normal = scipy.stats.shapiro(vr)[1] >= p_thr
            if (is_normal and test_func in [scipy.stats.mannwhitneyu, scipy.stats.wilcoxon, scipy.stats.kruskal]) \
                    or (not is_normal and test_func in [scipy.stats.ttest_ind, scipy.stats.ttest_rel]):
                print(f'WARNING: {v} is{"" if is_normal else " NOT"} normally distributed')

    # Find out the direction of the difference
    if test_func == scipy.stats.kruskal:
        if np.median(videos1_vent_rates) > np.median(videos2_vent_rates):
            direction = 'HIGHER'
        elif np.median(videos1_vent_rates) < np.median(videos2_vent_rates):
            direction = 'LOWER'
        else:
            direction = None
    else:
        _, p_greater = test_func(videos1_vent_rates, videos2_vent_rates, alternative='greater')
        _, p_less = test_func(videos1_vent_rates, videos2_vent_rates, alternative='less')
        if p_greater < p_thr <= p_less:
            direction = 'HIGHER'
            if test_func == scipy.stats.mannwhitneyu:
                assert np.median(videos1_vent_rates) >= np.median(videos2_vent_rates)
            if test_func == scipy.stats.ttest_ind:
                assert videos1_vent_rates.mean() > videos2_vent_rates.mean()
        elif p_less < p_thr <= p_greater:
            direction = 'LOWER'
            if test_func == scipy.stats.mannwhitneyu:
                assert np.median(videos1_vent_rates) <= np.median(videos2_vent_rates)
            if test_func == scipy.stats.ttest_ind:
                assert videos1_vent_rates.mean() < videos2_vent_rates.mean()
        else:
            assert p_greater >= p_thr and p_less >= p_thr
            direction = None

    # Perform the two-sided test
    stat, p = test_func(videos1_vent_rates, videos2_vent_rates, **test_kwargs)

    if verbose:
        print(f'{videos1} {f"has {direction}" if p < p_thr else "does NOT have a different"} '
              f'ventilation rate than {videos2}, stat={stat:.1f}, p={p:.1e}{" *" if p < p_thr else ""}')
    return videos1_vent_rates, videos2_vent_rates, stat, p, direction if p < p_thr else None


def get_true_buccal_and_ram_estimates(vent_df_no_nulls, fish_qual_df, fps, keep_singles, keep_long_flanks,
                                      size_filter, conf_filter, filter_true_long_rams, ram_to_buccal_ratio):

    if isinstance(vent_df_no_nulls, str):
        vent_df_no_nulls = pd.read_csv(vent_df_no_nulls, index_col=['video_id', 'fish_id', 'change_id'])
    if isinstance(fish_qual_df, str):
        fish_qual_df = pd.read_csv(fish_qual_df, index_col=['video_id', 'fish_id'])

    ram_df, buccal_df = split_ram_and_buccal(vent_df_no_nulls)

    if size_filter or conf_filter:
        buccal_df = buccal_df.reset_index().set_index(['video_id', 'fish_id'])
        if size_filter:
            buccal_df = buccal_df.loc[
                fish_qual_df.index[fish_qual_df['median_size'] > size_filter ** 2].intersection(buccal_df.index)
            ]
        if conf_filter:
            buccal_df = buccal_df.loc[
                fish_qual_df.index[fish_qual_df['q25_score'] > conf_filter].intersection(buccal_df.index)
            ]
        buccal_df = buccal_df.reset_index().set_index(['video_id', 'fish_id', 'change_id'])

    # just to check that these are truly the only two filters
    assert_df = remove_flanks(buccal_df.copy())
    if not keep_singles:
        n = assert_df.groupby(['video_id', 'fish_id'])['size'].count()
        assert_df = assert_df.reset_index().set_index(['video_id', 'fish_id'])
        assert_df = assert_df.loc[n.index[n > 1]]
        assert_df = assert_df.reset_index().set_index(['video_id', 'fish_id', 'change_id'])
        assert_index = assert_df.groupby(['video_id', 'fish_id'])['size'].count().index

    vent_lengths = get_average_vent_length(
        buccal_df, remove_flanking=True, keep_long_flanks=keep_long_flanks, keep_singles=keep_singles
    )
    assert assert_index.equals(vent_lengths.index)
    vent_rates = convert_to_vent_rates(vent_lengths, fps=fps)

    if filter_true_long_rams:
        assert ram_df.reset_index().set_index(['video_id', 'fish_id']).index.is_unique
        ram_df = ram_df.reset_index().set_index(['video_id', 'fish_id']).loc[
            fish_qual_df.index[fish_qual_df['NO_closed_mouths'] == True]
        ]
        if ram_to_buccal_ratio is not None:
            ram_df = ram_df.loc[ram_df['size'] - (ram_to_buccal_ratio * vent_lengths.groupby('video_id').mean()) > 0]

    return vent_rates, vent_lengths, buccal_df, ram_df, vent_df_no_nulls, fish_qual_df