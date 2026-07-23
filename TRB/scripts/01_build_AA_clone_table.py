import os
import re
import glob
import argparse
import pandas as pd


GENETIC_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W',

    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R',

    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R',

    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}


def translate_nt(seq):
    seq = str(seq).strip().upper().replace('U', 'T')
    seq = re.sub(r'\s+', '', seq)

    if not seq:
        return '', 'empty_sequence'

    if re.search(r'[^ACGT]', seq):
        return '', 'non_acgt_base'

    if len(seq) % 3 != 0:
        return '', 'length_not_multiple_of_3'

    aa = []

    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        aa1 = GENETIC_CODE.get(codon)

        if aa1 is None:
            return '', 'unknown_codon'

        if aa1 == '*':
            return '', 'stop_codon'

        aa.append(aa1)

    return ''.join(aa), 'ok'


def sample_id_from_filename(path):
    base = os.path.basename(path)
    suffix = '_TRB_CDR3_NT_frequency_error_correct.csv'

    if base.endswith(suffix):
        return base[:-len(suffix)]

    return re.sub(r'\.csv$', '', base)


def read_nt_table(path):
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        comment='#',
        engine='python'
    )

    if df.shape[1] < 2:
        raise ValueError('Input file has fewer than 2 columns: %s' % path)

    cols = ['cdr3_nt', 'read_count']

    if df.shape[1] >= 3:
        cols.append('input_frequency')

    if df.shape[1] >= 4:
        cols.append('input_cell_frequency')

    extra_n = df.shape[1] - len(cols)
    cols.extend(['extra_%d' % (i + 1) for i in range(extra_n)])

    df.columns = cols

    keep_cols = ['cdr3_nt', 'read_count']

    for c in ['input_frequency', 'input_cell_frequency']:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].copy()

    df['cdr3_nt'] = (
        df['cdr3_nt']
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace('U', 'T', regex=False)
    )

    df['read_count'] = pd.to_numeric(df['read_count'], errors='coerce').fillna(0)

    for c in ['input_frequency', 'input_cell_frequency']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    df = df[df['cdr3_nt'].ne('')].copy()

    return df


def build_one_sample(path, output_dir, invalid_log_file, invalid_log_header_written):
    sample_id = sample_id_from_filename(path)
    nt = read_nt_table(path)

    total_reads_all = float(nt['read_count'].sum())
    nt_clone_number_all = int(nt.shape[0])

    aa_status = nt['cdr3_nt'].apply(translate_nt)
    nt['cdr3_aa'] = [x[0] for x in aa_status]
    nt['translate_status'] = [x[1] for x in aa_status]

    valid = nt[
        nt['translate_status'].eq('ok') &
        nt['cdr3_aa'].ne('')
    ].copy()

    invalid = nt[~nt['translate_status'].eq('ok')].copy()

    if not invalid.empty:
        invalid.insert(0, 'sample_id', sample_id)

        invalid.to_csv(
            invalid_log_file,
            mode='a',
            index=False,
            header=(not invalid_log_header_written)
        )

        invalid_log_header_written = True

    valid_nt_clone_number = int(valid.shape[0])
    valid_aa_clone_ratio = (
        valid_nt_clone_number / nt_clone_number_all
        if nt_clone_number_all > 0
        else 0.0
    )

    total_reads_valid = float(valid['read_count'].sum())

    if valid.empty:
        out = pd.DataFrame(columns=[
            'sample_id',
            'cdr3_aa',
            'aa_length',
            'nt_clone_number',
            'read_count',
            'read_fraction',
            'input_frequency_sum',
            'input_cell_frequency_sum',
            'frequency',
            'frequency_norm',
            'top_nt_cdr3'
        ])

        freq_source = 'none'
        frequency_sum = 0.0
        frequency_norm_sum = 0.0
        read_fraction_sum = 0.0

    else:
        valid_sorted = valid.sort_values(
            ['cdr3_aa', 'read_count'],
            ascending=[True, False]
        )

        top_nt = (
            valid_sorted
            .groupby('cdr3_aa', as_index=False)
            .first()[['cdr3_aa', 'cdr3_nt']]
            .rename(columns={'cdr3_nt': 'top_nt_cdr3'})
        )

        agg_dict = {
            'cdr3_nt': 'nunique',
            'read_count': 'sum'
        }

        if 'input_frequency' in valid.columns:
            agg_dict['input_frequency'] = 'sum'

        if 'input_cell_frequency' in valid.columns:
            agg_dict['input_cell_frequency'] = 'sum'

        out = valid.groupby('cdr3_aa', as_index=False).agg(agg_dict)

        out = out.rename(columns={
            'cdr3_nt': 'nt_clone_number',
            'input_frequency': 'input_frequency_sum',
            'input_cell_frequency': 'input_cell_frequency_sum'
        })

        out.insert(0, 'sample_id', sample_id)
        out['aa_length'] = out['cdr3_aa'].str.len()

        out['read_fraction'] = (
            out['read_count'] / total_reads_valid
            if total_reads_valid > 0
            else 0.0
        )

        if (
            'input_cell_frequency_sum' in out.columns and
            out['input_cell_frequency_sum'].sum() > 0
        ):
            out['frequency'] = out['input_cell_frequency_sum']
            freq_source = 'input_cell_frequency_sum'

        elif (
            'input_frequency_sum' in out.columns and
            out['input_frequency_sum'].sum() > 0
        ):
            out['frequency'] = out['input_frequency_sum']
            freq_source = 'input_frequency_sum'

        else:
            out['frequency'] = out['read_fraction']
            freq_source = 'read_fraction'

        frequency_sum = float(out['frequency'].sum())

        out['frequency_norm'] = (
            out['frequency'] / frequency_sum
            if frequency_sum > 0
            else 0.0
        )

        out = out.merge(top_nt, on='cdr3_aa', how='left')

        ordered_cols = [
            'sample_id',
            'cdr3_aa',
            'aa_length',
            'nt_clone_number',
            'read_count',
            'read_fraction'
        ]

        optional_cols = [
            c for c in ['input_frequency_sum', 'input_cell_frequency_sum']
            if c in out.columns
        ]

        ordered_cols = ordered_cols + optional_cols + [
            'frequency',
            'frequency_norm',
            'top_nt_cdr3'
        ]

        out = out[ordered_cols]

        out = out.sort_values(
            ['frequency_norm', 'read_count'],
            ascending=[False, False]
        )

        frequency_norm_sum = float(out['frequency_norm'].sum())
        read_fraction_sum = float(out['read_fraction'].sum())

    output_file = os.path.join(
        output_dir,
        '%s_AA_clone_table.csv' % sample_id
    )

    out.to_csv(output_file, index=False)

    aa_clone_number = int(out.shape[0])

    valid_read_ratio = (
        total_reads_valid / total_reads_all
        if total_reads_all > 0
        else 0.0
    )

    aa_clone_reduction_ratio = (
        aa_clone_number / valid_nt_clone_number
        if valid_nt_clone_number > 0
        else None
    )

    summary = {
        'sample_id': sample_id,
        'source_file': os.path.basename(path),
        'total_reads_all_nt': total_reads_all,
        'total_reads_valid_nt': total_reads_valid,
        'valid_read_ratio': valid_read_ratio,
        'nt_clone_number_all': nt_clone_number_all,
        'valid_nt_clone_number': valid_nt_clone_number,
        'valid_aa_clone_ratio': valid_aa_clone_ratio,
        'aa_clone_number': aa_clone_number,
        'aa_clone_reduction_ratio': aa_clone_reduction_ratio,
        'frequency_source': freq_source,
        'frequency_sum': frequency_sum,
        'frequency_norm_sum': frequency_norm_sum,
        'read_fraction_sum': read_fraction_sum,
        'output_file': os.path.basename(output_file)
    }

    return summary, invalid_log_header_written


def build_invalid_status_summary(invalid_log_file, output_dir):
    if not os.path.exists(invalid_log_file) or os.path.getsize(invalid_log_file) == 0:
        out = pd.DataFrame(columns=[
            'translate_status',
            'invalid_nt_clone_number',
            'invalid_read_count',
            'invalid_nt_clone_ratio',
            'invalid_read_ratio'
        ])

        out.to_csv(
            os.path.join(output_dir, '01_invalid_status_summary.csv'),
            index=False
        )

        return out

    rows = []

    for chunk in pd.read_csv(
        invalid_log_file,
        usecols=['translate_status', 'read_count'],
        chunksize=500000
    ):
        tmp = (
            chunk
            .groupby('translate_status')
            .agg(
                invalid_nt_clone_number=('translate_status', 'size'),
                invalid_read_count=('read_count', 'sum')
            )
            .reset_index()
        )

        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)

    out = (
        out
        .groupby('translate_status', as_index=False)
        .agg(
            invalid_nt_clone_number=('invalid_nt_clone_number', 'sum'),
            invalid_read_count=('invalid_read_count', 'sum')
        )
    )

    total_invalid_clone = out['invalid_nt_clone_number'].sum()
    total_invalid_reads = out['invalid_read_count'].sum()

    out['invalid_nt_clone_ratio'] = (
        out['invalid_nt_clone_number'] / total_invalid_clone
        if total_invalid_clone > 0
        else 0.0
    )

    out['invalid_read_ratio'] = (
        out['invalid_read_count'] / total_invalid_reads
        if total_invalid_reads > 0
        else 0.0
    )

    out = out.sort_values('invalid_nt_clone_number', ascending=False)

    out.to_csv(
        os.path.join(output_dir, '01_invalid_status_summary.csv'),
        index=False
    )

    return out


def add_qc_flags(summary_df, min_valid_read_ratio, min_valid_aa_clone_ratio,
                 min_nt_clones_warn, norm_tolerance):
    df = summary_df.copy()

    df['qc_pass_01'] = (
        (df['nt_clone_number_all'] >= min_nt_clones_warn) &
        (df['aa_clone_number'] > 0) &
        (df['valid_read_ratio'] > 0) &
        (df['frequency_source'].ne('none'))
    )

    reasons = []

    for _, row in df.iterrows():
        r = []

        if row['nt_clone_number_all'] < min_nt_clones_warn:
            r.append('原始NT clone数量过少')

        if row['aa_clone_number'] == 0:
            r.append('无有效AA clone')

        if row['valid_read_ratio'] == 0:
            r.append('有效reads比例为0')

        elif row['valid_read_ratio'] < min_valid_read_ratio:
            r.append('有效reads比例偏低')

        if row['valid_aa_clone_ratio'] == 0:
            r.append('有效AA clone比例为0')

        elif row['valid_aa_clone_ratio'] < min_valid_aa_clone_ratio:
            r.append('有效AA clone比例偏低')

        if row['frequency_source'] == 'none':
            r.append('无可用频率来源')

        if row['aa_clone_number'] > 0:
            if abs(row['frequency_norm_sum'] - 1.0) > norm_tolerance:
                r.append('frequency_norm求和异常')

            if abs(row['read_fraction_sum'] - 1.0) > norm_tolerance:
                r.append('read_fraction求和异常')

        reasons.append(';'.join(r))

    df['qc_warning_01'] = reasons

    df['has_warning_01'] = df['qc_warning_01'].ne('')

    return df


def print_chinese_report(summary_df, invalid_status_df, qc_df, output_dir,
                         min_valid_read_ratio, min_valid_aa_clone_ratio):
    input_n = summary_df.shape[0]
    pass_n = int(qc_df['qc_pass_01'].sum())
    fail_n = int((~qc_df['qc_pass_01']).sum())
    warn_n = int(qc_df['has_warning_01'].sum())

    print()

    print('==============================')
    print('01_AA_clone_table 运行完成')
    print('==============================')
    print('输入样本数: %d' % input_n)
    print('输出AA clone表: %d' % input_n)
    print('QC通过样本: %d' % pass_n)
    print('QC失败样本: %d' % fail_n)
    print('存在警告样本: %d' % warn_n)

    print()

    print('[无效NT序列原因汇总]')
    if invalid_status_df.empty:
        print('未发现无效NT序列。')
    else:
        for _, row in invalid_status_df.iterrows():
            print(
                '- %s: clone %d (%.2f%%), reads %.0f (%.2f%%)' %
                (
                    row['translate_status'],
                    int(row['invalid_nt_clone_number']),
                    row['invalid_nt_clone_ratio'] * 100,
                    row['invalid_read_count'],
                    row['invalid_read_ratio'] * 100
                )
            )

    print()

    print('[样本级QC概览]')
    print(
        '- valid_read_ratio: mean %.3f, median %.3f, min %.3f, max %.3f' %
        (
            summary_df['valid_read_ratio'].mean(),
            summary_df['valid_read_ratio'].median(),
            summary_df['valid_read_ratio'].min(),
            summary_df['valid_read_ratio'].max()
        )
    )

    print(
        '- valid_aa_clone_ratio: mean %.3f, median %.3f, min %.3f, max %.3f' %
        (
            summary_df['valid_aa_clone_ratio'].mean(),
            summary_df['valid_aa_clone_ratio'].median(),
            summary_df['valid_aa_clone_ratio'].min(),
            summary_df['valid_aa_clone_ratio'].max()
        )
    )

    print()

    print('[最低valid_read_ratio样本 Top 5]')
    low = (
        summary_df
        .sort_values('valid_read_ratio')
        .head(5)[['sample_id', 'valid_read_ratio', 'valid_aa_clone_ratio', 'aa_clone_number']]
    )

    for _, row in low.iterrows():
        print(
            '- %s: valid_read_ratio %.3f, valid_aa_clone_ratio %.3f, aa_clone_number %d' %
            (
                row['sample_id'],
                row['valid_read_ratio'],
                row['valid_aa_clone_ratio'],
                int(row['aa_clone_number'])
            )
        )

    fail_df = qc_df[~qc_df['qc_pass_01']].copy()

    if fail_df.empty:
        print()
        print('[QC失败样本]')
        print('无。')
    else:
        print()
        print('[QC失败样本，建议后续02/03/04排除]')
        for _, row in fail_df.iterrows():
            print('- %s: %s' % (row['sample_id'], row['qc_warning_01']))

    warn_df = qc_df[
        qc_df['has_warning_01'] &
        qc_df['qc_pass_01']
    ].copy()

    if not warn_df.empty:
        print()
        print('[其他警告样本，建议人工关注]')
        for _, row in warn_df.head(10).iterrows():
            print('- %s: %s' % (row['sample_id'], row['qc_warning_01']))

        if warn_df.shape[0] > 10:
            print('- 其余 %d 个警告样本见 01_sample_qc_status.csv' % (warn_df.shape[0] - 10))

    print()

    print('[输出文件]')
    print('- AA clone表目录: %s' % output_dir)
    print('- 总结表: %s' % os.path.join(output_dir, '01_AA_clone_table_summary.csv'))
    print('- QC检查表: %s' % os.path.join(output_dir, '01_AA_clone_table_summary_checked.csv'))
    print('- 无效NT原因汇总: %s' % os.path.join(output_dir, '01_invalid_status_summary.csv'))
    print('- 样本QC状态: %s' % os.path.join(output_dir, '01_sample_qc_status.csv'))
    print('- 后续建议排除样本: %s' % os.path.join(output_dir, '01_exclude_samples.txt'))

    print()

    print('说明:')
    print('- length_not_multiple_of_3 和 stop_codon 通常代表非生产性CDR3，不建议强行翻译。')
    print('- 02/03/04 建议默认跳过 01_exclude_samples.txt 中的样本。')


def main():
    parser = argparse.ArgumentParser(
        description='Build per-sample AA CDR3 clone tables from TRB NT CDR3 files.'
    )

    parser.add_argument(
        '--input-dir',
        default='origin_result'
    )

    parser.add_argument(
        '--output-dir',
        default='result/01_AA_clone_table'
    )

    parser.add_argument(
        '--pattern',
        default='*_TRB_CDR3_NT_frequency_error_correct.csv'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only list input files and exit without writing outputs.'
    )

    parser.add_argument(
        '--min-valid-read-ratio',
        type=float,
        default=0.70,
        help='Warning threshold for valid_read_ratio.'
    )

    parser.add_argument(
        '--min-valid-aa-clone-ratio',
        type=float,
        default=0.70,
        help='Warning threshold for valid_aa_clone_ratio.'
    )

    parser.add_argument(
        '--min-nt-clones-warn',
        type=int,
        default=10,
        help='Warning threshold for very small raw NT clone number.'
    )

    parser.add_argument(
        '--norm-tolerance',
        type=float,
        default=1e-4,
        help='Tolerance for frequency_norm_sum and read_fraction_sum.'
    )

    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))

    if not files:
        raise SystemExit(
            'No input files found: %s' %
            os.path.join(args.input_dir, args.pattern)
        )

    if args.dry_run:
        print('Dry run: 不写入结果文件。')
        print('输入目录: %s' % args.input_dir)
        print('匹配模式: %s' % args.pattern)
        print('发现文件数: %d' % len(files))
        print()
        print('前5个文件:')
        for f in files[:5]:
            print('- %s' % os.path.basename(f))
        return

    os.makedirs(args.output_dir, exist_ok=True)

    invalid_log_file = os.path.join(
        args.output_dir,
        '01_invalid_nt_translation_log.csv'
    )

    if os.path.exists(invalid_log_file):
        os.remove(invalid_log_file)

    summaries = []
    invalid_log_header_written = False

    for i, path in enumerate(files, 1):
        summary, invalid_log_header_written = build_one_sample(
            path=path,
            output_dir=args.output_dir,
            invalid_log_file=invalid_log_file,
            invalid_log_header_written=invalid_log_header_written
        )

        summaries.append(summary)

        print(
            '[%d/%d] %s -> %s' %
            (i, len(files), summary['sample_id'], summary['output_file'])
        )

    if not os.path.exists(invalid_log_file):
        pd.DataFrame(columns=[
            'sample_id',
            'cdr3_nt',
            'read_count',
            'input_frequency',
            'input_cell_frequency',
            'cdr3_aa',
            'translate_status'
        ]).to_csv(invalid_log_file, index=False)

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        os.path.join(args.output_dir, '01_AA_clone_table_summary.csv'),
        index=False
    )

    checked_file = os.path.join(
        args.output_dir,
        '01_AA_clone_table_summary_checked.csv'
    )

    summary_df.to_csv(checked_file, index=False)

    frequency_check = summary_df[[
        'sample_id',
        'aa_clone_number',
        'frequency_sum',
        'frequency_norm_sum',
        'read_fraction_sum'
    ]].copy()

    frequency_check.to_csv(
        os.path.join(args.output_dir, '01_frequency_norm_check.csv'),
        index=False
    )

    invalid_status_df = build_invalid_status_summary(
        invalid_log_file,
        args.output_dir
    )

    qc_df = add_qc_flags(
        summary_df=summary_df,
        min_valid_read_ratio=args.min_valid_read_ratio,
        min_valid_aa_clone_ratio=args.min_valid_aa_clone_ratio,
        min_nt_clones_warn=args.min_nt_clones_warn,
        norm_tolerance=args.norm_tolerance
    )

    qc_df.to_csv(
        os.path.join(args.output_dir, '01_sample_qc_status.csv'),
        index=False
    )

    qc_df.loc[~qc_df['qc_pass_01'], 'sample_id'].to_csv(
        os.path.join(args.output_dir, '01_exclude_samples.txt'),
        index=False,
        header=False
    )

    print_chinese_report(
        summary_df=summary_df,
        invalid_status_df=invalid_status_df,
        qc_df=qc_df,
        output_dir=args.output_dir,
        min_valid_read_ratio=args.min_valid_read_ratio,
        min_valid_aa_clone_ratio=args.min_valid_aa_clone_ratio
    )


if __name__ == '__main__':
    main()
