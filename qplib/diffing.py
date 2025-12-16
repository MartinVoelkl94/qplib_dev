
import pandas as pd
import copy

from .types import Container
from .pandas import deduplicate
from .excel import hide, format_excel
from .util import (
    log,
    _arg_to_list,
    ensure_unique_string,
    GREEN,
    RED,
    GREEN_LIGHT,
    ORANGE_LIGHT,
    RED_LIGHT,
    )





class Diff:
    """
    Stores differences between 2 dataframes.
    For detailed documentation see qp.diff() function.
    """

    def __init__(
            self,
            old: pd.DataFrame,
            new: pd.DataFrame,
            uid=None,
            ignore=None,
            rename=None,
            name='',
            verbosity=3,
            ):

        self.verbosity = verbosity
        self.name = name
        self.old = old.copy()
        self.new = new.copy()
        self.rename_cols(rename)
        self.ignore_cols(ignore)  #sets: self.cols_ignore
        self.set_uid(uid)  #sets: self.uid, self.old.index, self.new.index

    def rename_cols(self, rename=None):
        if isinstance(rename, dict):
            self.old = self.old.rename(columns=rename)
            self.new = self.new.rename(columns=rename)
        return self

    def ignore_cols(self, ignore=None):
        cols_ignore = (
            self.old.columns
            .union(self.new.columns)
            .intersection(_arg_to_list(ignore))
            )
        self.cols_ignore = cols_ignore
        return self

    def set_uid(self, uid=None):
        if uid is False:
            uid = ''
        elif uid is None:
            msg = 'debug: searching for suitable uid column'
            log(msg, 'qp.Diff', self.verbosity)
            uid = self._find_uid()

        if uid in self.old.columns and uid in self.new.columns:
            self.old.index = self.old[uid]
            self.new.index = self.new[uid]

        if not self.old.index.is_unique:
            self.old.index = deduplicate(
                self.old.index,
                name=f'{uid} in old df',
                verbosity=self.verbosity,
                )
        if not self.new.index.is_unique:
            self.new.index = deduplicate(
                self.new.index,
                name=f'{uid} in new df',
                verbosity=self.verbosity,
                )
        self.uid = uid
        return self

    def _find_uid(self):

        uids_potential = self.new.columns.intersection(self.old.columns)
        uids_by_uniqueness = {}
        for uid in uids_potential:
            unique_in_old = pd.Index(self.old[uid].dropna()).unique()
            unique_in_new = pd.Index(self.new[uid].dropna()).unique()
            unique_shared = unique_in_new.intersection(unique_in_old)
            uids_by_uniqueness[uid] = len(unique_shared)

        if len(uids_by_uniqueness) > 0:
            uid = sorted(
                uids_by_uniqueness.items(),
                key=lambda item: item[1],
                reverse=True,
                )[0][0]
            msg = f'debug: found uid "{uid}" for comparison'
            log(msg, 'qp.Diff', self.verbosity)
        else:
            uid = ''
            msg = 'info: no uid found. using index for comparison'
            log(msg, 'qp.Diff', self.verbosity)

        return uid

    def summary(self, detailed=False):
        """
        """
        cols_shared = (
            self.new
            .columns
            .intersection(self.old.columns)
            .difference(self.cols_ignore)
            )
        cols_added = (
            self.new
            .columns
            .difference(self.old.columns)
            .difference(self.cols_ignore)
            )
        cols_removed = (
            self.old
            .columns
            .difference(self.new.columns)
            .difference(self.cols_ignore)
            )

        rows_shared = (
            self.new
            .index
            .intersection(self.old.index)
            )
        rows_added = (
            self.new
            .index
            .difference(self.old.index)
            )
        rows_removed = (
            self.old
            .index
            .difference(self.new.index)
            )

        dtypes_changed = {}
        for col in cols_shared:
            if self.old[col].dtype != self.new[col].dtype:
                changed = {
                    'old': self.old[col].dtype,
                    'new': self.new[col].dtype
                    }
                dtypes_changed[col] = changed

        summary = Container()
        summary.name = self.name
        summary.uid = self.uid
        summary.n_cols_shared = len(cols_shared)
        summary.n_cols_added = len(cols_added)
        summary.n_cols_removed = len(cols_removed)
        summary.n_rows_shared = len(rows_shared)
        summary.n_rows_added = len(rows_added)
        summary.n_rows_removed = len(rows_removed)
        summary.n_dtypes_changed = len(dtypes_changed)
        if detailed:
            summary.cols_shared = cols_shared
            summary.cols_added = cols_added
            summary.cols_removed = cols_removed
            summary.rows_shared = rows_shared
            summary.rows_added = rows_added
            summary.rows_removed = rows_removed
            summary.dtypes_changed = dtypes_changed
        return summary

    def details(self):
        details = self.summary(detailed=True)
        return details

    def show(
            self,
            mode='mix',
            prefix_old='old: ',
            linebreak='<br>',
            ):
        """
        Generate a styled DataFrame showing differences between datasets.
        Differences are highlighted with color-coded styles,
        supporting multiple visualization modes.

        Parameters
        ----------

        mode : str, default 'mix'
            Display mode for differences:
            - 'new': Show new DataFrame with added and changed elements highlighted
            - 'new+': Show new DataFrame with old values in additional (hidden) columns
            - 'old': Show old DataFrame with removed elements highlighted
            - 'mix': Combine both DataFrames showing all changes

        prefix_old : str, default 'old: '
            Prefix for columns showing old values (used in 'new+' mode)


        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with color-coded differences, or None if sheet not found
        """

        old = self.old
        new = self.new
        uid = self.uid
        details = self.details()
        cols_added = details.cols_added
        cols_removed = details.cols_removed
        cols_shared = details.cols_shared
        rows_added = details.rows_added
        rows_removed = details.rows_removed
        rows_shared = details.rows_shared

        if mode not in ('new', 'new+', 'old', 'mix'):
            log(f'error: unknown mode: {mode}', 'qp.Diff', self.verbosity)
            raise ValueError(f'Unknown mode: {mode}')


        if new.empty:
            df_diff = old.copy()
            df_diff_style = pd.DataFrame(
                f'background-color: {RED}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, 'removed dataset')


        elif old.empty:
            df_diff = new.copy()
            df_diff_style = pd.DataFrame(
                f'background-color: {GREEN}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, 'added dataset')


        elif mode in ['new', 'new+']:
            df_diff = new.copy()
            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            #add metadata columns
            if mode == 'new+':
                cols_add = []
                cols_reorder = []
                for col in df_diff.columns:
                    cols_reorder.append(col)
                    if col != uid:
                        col_name = prefix_old + col
                        cols_reorder.append(col_name)
                        if col_name not in df_diff.columns:
                            cols_add.append(col_name)

                df_diff_add = pd.DataFrame(
                    '',
                    index=df_diff.index,
                    columns=cols_add,
                    )
                df_diff_style_add = pd.DataFrame(
                    'font-style: italic',
                    index=df_diff.index,
                    columns=cols_add
                    )

                df_diff = pd.concat(
                    [df_diff, df_diff_add],
                    axis=1,
                    )
                df_diff_style = pd.concat(
                    [df_diff_style, df_diff_style_add],
                    axis=1,
                    )

                df_diff = df_diff[cols_reorder]
                df_diff_style = df_diff_style[cols_reorder]


            df_diff_style.loc[:, cols_added] = f'background-color: {GREEN}'
            df_diff_style.loc[rows_added, :] = f'background-color: {GREEN}'

            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, '')
            df_diff.loc[rows_added, col_diff] += 'added row'


        elif mode == 'old':
            df_diff = old.copy()
            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, '')
            df_diff.loc[rows_removed, col_diff] += 'removed row'


        elif mode == 'mix':
            inds_old = old.index.difference(new.index)
            cols_old = old.columns.difference(new.columns)

            df_diff = pd.concat([new, old.loc[:, cols_old]], axis=1)
            df_diff.loc[inds_old, :] = old.loc[inds_old, :]

            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            df_diff_style.loc[:, cols_added] = f'background-color: {GREEN}'
            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_added, :] = f'background-color: {GREEN}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, '')
            df_diff.loc[rows_added, col_diff] += 'added row'
            df_diff.loc[rows_removed, col_diff] += 'removed row'



        #highlight values in shared columns

        # replace "<" and ">" with html entities to prevent interpretation as html tags
        if pd.__version__ >= '2.1.0':
            df_diff = df_diff.map(lambda x: _replace_gt_lt(x))
        else:
            df_diff = df_diff.applymap(lambda x: _replace_gt_lt(x))

        df_old_isna = old.loc[rows_shared, cols_shared].isna()
        df_new_isna = new.loc[rows_shared, cols_shared].isna()
        df_new_equals_old = (
            new.loc[rows_shared, cols_shared]
            == old.loc[rows_shared, cols_shared]
            )

        #these comparisons can result in dtype "boolean" instead of "bool"
        #"boolean" masks cannot be used to set values as str
        df_added = (df_old_isna & ~df_new_isna).astype(bool)
        df_removed = (df_new_isna & ~df_old_isna).astype(bool)
        df_changed = (~df_new_isna & ~df_old_isna & ~df_new_equals_old).astype(bool)

        df_diff_style.loc[rows_shared, cols_shared] += (
            df_added
            .mask(df_added, f'background-color: {GREEN_LIGHT}')
            .where(df_added, '')
            )

        df_diff_style.loc[rows_shared, cols_shared] += (
            df_removed
            .mask(df_removed, f'background-color: {RED_LIGHT}')
            .where(df_removed, '')
            )

        df_diff_style.loc[rows_shared, cols_shared] += (
            df_changed
            .mask(df_changed, f'background-color: {ORANGE_LIGHT}')
            .where(df_changed, '')
            )


        #summarize changes in diff column
        sum_added = df_added.sum(axis=1)
        sum_removed = df_removed.sum(axis=1)
        sum_changed = df_changed.sum(axis=1)

        added = sum_added[sum_added > 0].index
        removed = sum_removed[sum_removed > 0].index
        changed = sum_changed[sum_changed > 0].index

        df_diff.loc[added, col_diff] += 'vals added: '
        df_diff.loc[added, col_diff] += sum_added[added].astype(str)
        df_diff.loc[added.intersection(removed), col_diff] += linebreak
        df_diff.loc[added.intersection(changed), col_diff] += linebreak

        df_diff.loc[removed, col_diff] += 'vals removed: '
        df_diff.loc[removed, col_diff] += sum_removed[removed].astype(str)
        df_diff.loc[removed.intersection(changed), col_diff] += linebreak

        df_diff.loc[changed, col_diff] += 'vals changed: '
        df_diff.loc[changed, col_diff] += sum_changed[changed].astype(str)


        if mode == 'new+':
            cols_shared_metadata = [prefix_old + col for col in cols_shared]
            df_all_modifications = (df_added | df_removed | df_changed)
            df_old_changed = (
                old
                .loc[rows_shared, cols_shared]
                .where(df_all_modifications, '')
                )
            df_diff.loc[rows_shared, cols_shared_metadata] = df_old_changed.values


        if len(df_diff.columns) * len(df_diff.index) > 100_000:
            msg = (
                'warning: more than 100 000 cells are being formatted.'
                'while this might not cause performance issues for formatting,'
                'the result might be slow to render, especially in jupyter notebooks.'
                )
            log(msg, 'qp.Diff', self.verbosity)

        diff_styled = df_diff.style.apply(lambda x: df_diff_style, axis=None)
        diff_styled = diff_styled.set_properties(**{
            # 'text-align': 'left',  #not working on all rows
            'white-space': 'pre-wrap',
            })
        return diff_styled



class Diffs:
    """
    Class to calculate and store differences between
    dataframes, csv or excel files.
    Used as return type of the diff() function.
    """

    def __init__(
            self,
            old: pd.DataFrame | str,
            new: pd.DataFrame | str,
            uid=None,
            ignore=None,
            rename=None,
            verbosity=3,
            ):
        self.verbosity = verbosity
        self.all = self._get_Diffs(
            old,
            new,
            uid,
            ignore,
            rename,
            )

    def _get_Diffs(
            self,
            old,
            new,
            uid=None,
            ignore=None,
            rename=None,
            ):
        """
        Loads dataframes from various sources (pd.DataFrame, CSV files, Excel files)
        and determines whether to compare single sheets or multiple Excel sheets.
        """

        #when both inputs are excel files, they might
        #contain multiple sheets to be compared
        conditions_excel_comp = (
            isinstance(old, str)
            and isinstance(new, str)
            and old.endswith('.xlsx')
            and new.endswith('.xlsx')
            )
        if conditions_excel_comp:
            msg = 'debug: comparing all sheets from 2 excel files'
            log(msg, 'qp.Diff', self.verbosity)
            diffs = self._get_excel_Diffs(
                old,
                new,
                uid=uid,
                ignore=ignore,
                rename=rename,
                )
            return diffs


        #only 2 dfs need to be compared if input is
        #2 dfs, 2 csvs or 1 df and 1 csv/excel file
        else:
            if isinstance(old, str):
                if old.endswith('.csv'):
                    df_old = pd.read_csv(old)
                elif old.endswith('.xlsx'):
                    df_old = pd.read_excel(old)
                else:
                    msg = f'error: unknown file extension: {old}'
                    log(msg, 'qp.Diff', self.verbosity)
            elif isinstance(old, pd.DataFrame):
                df_old = old
            else:
                msg = 'error: incompatible type for old df'
                log(msg, 'qp.Diff', self.verbosity)

            if isinstance(new, str):
                if new.endswith('.csv'):
                    df_new = pd.read_csv(new)
                elif new.endswith('.xlsx'):
                    df_new = pd.read_excel(new)
                else:
                    msg = f'error: unknown file extension: {new}'
                    log(msg, 'qp.Diff', self.verbosity)
            elif isinstance(new, pd.DataFrame):
                df_new = new

            else:
                msg = 'error: incompatible type for new df'
                log(msg, 'qp.Diff', self.verbosity)

            diff = Diff(
                df_old,
                df_new,
                uid=uid,
                ignore=ignore,
                rename=rename,
                verbosity=self.verbosity,
                )
            diff.sheet = ''
            diff.in_both_datasets = 'yes'
            diffs = [diff]
            return diffs

    def _get_excel_Diffs(
            self,
            old,
            new,
            uid=None,
            ignore=None,
            rename=None,
            ):
        """
        Read all sheets from two Excel files.
        """

        diffs = []
        sheets_old = pd.ExcelFile(old).sheet_names
        sheets_new = pd.ExcelFile(new).sheet_names
        sheets_all = list(dict.fromkeys(sheets_new + sheets_old))  #preserves order

        for sheet in sheets_all:
            if sheet in sheets_old and sheet in sheets_new:
                df_old = pd.read_excel(old, sheet_name=sheet)
                df_new = pd.read_excel(new, sheet_name=sheet)
                in_both_datasets = 'yes'
            elif sheet in sheets_new:
                df_old = pd.DataFrame()
                df_new = pd.read_excel(new, sheet_name=sheet)
                in_both_datasets = 'only in new'
            elif sheet in sheets_old:
                df_old = pd.read_excel(old, sheet_name=sheet)
                df_new = pd.DataFrame()
                in_both_datasets = 'only in old'

            diff = Diff(
                df_old,
                df_new,
                uid=uid,
                ignore=ignore,
                rename=rename,
                name=sheet,
                verbosity=self.verbosity,
                )
            diff.sheet = sheet
            diff.in_both_datasets = in_both_datasets
            diffs.append(diff)

        return diffs

    def rename_cols(self, rename=None, sheet=None):
        """
        """
        for diff in self.all:
            if sheet is None or diff.name == sheet:
                diff.rename_cols(rename)
        return self

    def ignore_cols(self, ignore=None, sheet=None):
        """
        """
        for diff in self.all:
            if sheet is None or diff.name == sheet:
                diff.ignore_cols(ignore)
        return self

    def set_uid(self, uid=None, sheet=None):
        """
        """
        for diff in self.all:
            if sheet is None or diff.name == sheet:
                diff.set_uid(uid)
        return self

    def __getitem__(self, key):
        return self.all[key]

    def summary(self, detailed=False, linebreak='<br>', concat_limit=1000):
        """
        """

        sheets = [diff.name for diff in self.all]
        data = {
            'uid': [diff.uid for diff in self.all],
            'in_both_datasets': [diff.in_both_datasets for diff in self.all],
            }
        summary = pd.DataFrame(data, index=sheets)

        cols = [
            'n_cols_shared',
            'n_cols_added',
            'n_cols_removed',
            'n_rows_shared',
            'n_rows_added',
            'n_rows_removed',
            'n_dtypes_changed',
            ]
        if detailed:
            cols += [
                'cols_shared',
                'cols_added',
                'cols_removed',
                'rows_shared',
                'rows_added',
                'rows_removed',
                'dtypes_changed',
                ]

        for diff in self.all:
            diff_summary = diff.summary(detailed=detailed)
            for col in cols:
                string = _to_str(
                    diff_summary[col],
                    linebreak,
                    concat_limit,
                    )
                summary.loc[diff.name, col] = string

        summary.columns = [col.replace('_', ' ') for col in summary.columns]
        summary = summary.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap',
            })
        return summary


    def details(self, linebreak='<br>', concat_limit=1000):
        """
        """
        details = self.summary(
            detailed=True,
            linebreak=linebreak,
            concat_limit=concat_limit,
            )
        return details



    def show(
            self,
            mode='mix',
            sheet=0,
            prefix_old='old: ',
            linebreak='<br>',
            ):
        """
        """

        if isinstance(sheet, int):
            diff = self.all[sheet]
        else:
            for sheet in self.all:
                if sheet.name == sheet:
                    diff = sheet
                    break
            else:
                msg = f'error: sheet "{sheet}" not found in diffs'
                log(msg, 'qp.Diff', self.verbosity)
                return None

        result = diff.show(
            mode,
            prefix_old,
            linebreak,
            )
        return result


    # def to_excel(
    #         self,
    #         path,
    #         mode='new+',
    #         index=False,
    #         prefix_old='old: ',
    #         linebreak='\n',
    #         ):
    #     """
    #     Export diff results to an Excel file with formatting.

    #     Creates an Excel file containing a summary sheet and individual
    #     sheets for each comparison with highlighted differences. Applies
    #     Excel-specific formatting and hides old value columns.


    #     Parameters
    #     ----------
    #     path : str
    #         File path for the output Excel file
    #     mode : str, default 'mix'
    #         Display mode for differences (see show() method for options)
    #     index : bool, default False
    #         Whether to include row indices in the Excel output
    #     prefix_old : str, default 'old: '
    #         Prefix for columns showing old values (used in 'new+' mode)
    #     linebreak : str, default '\\n'
    #         String to use for line breaks in Excel cells
    #     """

    #     details = self.details(linebreak=linebreak)
    #     name = ensure_unique_string('diff', details.index)
    #     with pd.ExcelWriter(path) as writer:
    #         pd.DataFrame().to_excel(
    #             writer,
    #             sheet_name=name,
    #             )


    #     for i, diff in enumerate(self.all):


    #         result = self.show(
    #             mode=mode,
    #             sheet=i,
    #             prefix_old=prefix_old,
    #             )

    #         result.data['meta'] = result.data['meta'].str.replace('<br>', '\n')

    #         with pd.ExcelWriter(path) as writer:
    #             result.to_excel(
    #                 writer,
    #                 sheet_name=sheet,
    #                 index=index,
    #                 )

    #     with pd.ExcelWriter(path) as writer:
    #         self.summary(linebreak).to_excel(
    #             writer,
    #             sheet_name='diff_summary',
    #             index=index,
    #             )

    #     if mode == 'new+':
    #         hide(
    #             path,
    #             axis='col',
    #             patterns=f'{prefix_old}.*',
    #             hide=True,
    #             verbosity=self._verbosity,
    #             )
    #     format_excel(path)
    #     log(f'info: differences saved to "{path}"', 'qp.Diff', self._verbosity)


    # def print(self):
    #     """
    #     Print the string representation of differences to console.

    #     Convenience method that prints the output of str() method,
    #     providing a readable summary of all differences found.
    #     """
    #     print(self.str())


    # def str(self):
    #     """
    #     Get string representation of differences.

    #     Convenience method that calls __str__() to return a formatted
    #     string summary of all differences found between datasets.
    #     """
    #     return str(self)


    # def __str__(self):
    #     """
    #     Generate a human-readable string summary of all differences.

    #     Creates a detailed text summary showing differences for single
    #     DataFrame comparisons or multi-sheet Excel file comparisons.
    #     Handles cases where datasets are identical.
    #     """
    #     summary = self.summary()
    #     if len(self.sheets) == 1 and self.sheets[0] is None:
    #         if self.dfs_new[0].equals(self.dfs_old[0]):
    #             string = 'both dataframes are identical'
    #         else:
    #             string = 'Diff between 2 dataframes\n'
    #             string += _sheet_to_str(summary, 0)
    #     else:
    #         string = f'Diff between 2 excel files with {len(self.sheets)} sheets\n'
    #         for i, sheet in enumerate(self.sheets):
    #             if self.dfs_new[i].equals(self.dfs_old[i]):
    #                 string += f'\nSheet "{sheet}" is identical in both files\n'
    #             else:
    #                 string += f'\nSheet: {sheet}\n'
    #                 string += _sheet_to_str(summary, i)
    #     return string


def _to_str(obj, linebreak='<br>', concat_limit=1000):

    if isinstance(obj, int):
        if obj == 0:
            string = ''
        else:
            string = str(obj)

    elif isinstance(obj, dict):
        if len(obj) == 0:
            string = ''
        elif len(obj) > concat_limit:
            string = f'over {concat_limit} values'
        else:
            strings = [f'{k} -> {v}' for k, v in obj.items()]
            string = f';{linebreak}'.join(strings)

    elif isinstance(obj, (list, set, tuple, pd.Index)):
        if len(obj) == 0:
            string = ''
        elif len(obj) > concat_limit:
            string = f'over {concat_limit} values'
        else:
            strings = [str(x) for x in obj]
            string = f';{linebreak}'.join(strings)

    else:
        string = str(obj)

    return string


# def _sheet_to_str(summary, row):
#     string = ''
#     one_liners = [
#         'in both files',
#         'uid col',
#         'cols shared',
#         'rows shared',
#         ]
#     for col in summary.data.columns:
#         if col == 'sheets':
#             continue
#         elif col in one_liners:
#             string += f'  {col}: {summary.data.loc[row, col]}\n'
#         else:
#             string += f'  {col}:\n    {summary.data.loc[row, col]}\n'
#     return string.replace('<br>', '\n    ')


# def _cols_to_str(iter):
#     iter_new = []
#     for col in iter:
#         if col is None:
#             iter_new.append('')
#         else:
#             iter_new.append(str(col))
#     return iter_new


# def _dicts_to_str(iter, linebreak='<br>'):
#     iter_new = []
#     for dictionary in iter:
#         if dictionary:
#             iter_new.append(
#                 f';{linebreak}'.join(
#                     [f'{k} -> {v}' for k, v in dictionary.items()]
#                     )
#                 )
#         else:
#             iter_new.append('')
#     return iter_new


# def _iters_to_str(iters, linebreak='<br>'):
#     iters_new = []
#     for iter in iters:
#         iters_new.append(f';{linebreak}'.join([str(x) for x in iter]))
#     return iters_new


# def _nested_dicts_to_str(iter, linebreak='<br>'):
#     iter_new = []
#     for dictionary in iter:
#         string = ''
#         if dictionary:
#             for k, v in dictionary.items():
#                 string += f'{k}: {v["old"]} -> {v["new"]}{linebreak}'
#         iter_new.append(string)
#     return iter_new


def _replace_gt_lt(x):
    if isinstance(x, str):
        return x.replace('<', '&lt;').replace('>', '&gt;')
    elif isinstance(x, type):
        return str(x).replace('<', '&lt;').replace('>', '&gt;')
    else:
        return x


def diff(
        old: pd.DataFrame | str,
        new: pd.DataFrame | str,
        uid=None,
        ignore=None,
        rename=None,
        verbosity=3,
        ) -> Diffs:
    """
    Calculates differences between dataframes,
    csv or excel files and returns a Diff object.


    Parameters
    ----------

    old, new : pd.DataFrame of filepath to CSV or Excel file

    uid : which column to use as a unique identifier to specify
        which rows correspond to each other in old and new data.
        If None, the function will try to find a suitable column automatically.
        If no suitable column is found, the index will be used.
        If False, the index will be used.
        when comparing multiple sheets from excel files, a dictionary
        with sheet names as keys and uid column names as values can be provided.

    ignore : column name or list of column names to ignore for comparison

    rename : dictionary to rename columns before comparison.
        Note that renaming is done before uid and columns to ignore
        are determined, meaning that those must use the new column names.
        This can also be used to fix situations where corrresponding columns
        in old and new data have different names (see examples)


    Examples
    --------

    basic usage:

    >>> import qplib as qp
    >>> diff = qp.diff(df_old, df_new)
    >>> diff.show()  #returns df with highlighted differences
    >>> diff.summary()  #returns df with summary stats
    >>> diff.str()  #string version of summary stats
    >>> diff.print()  #prints the string version
    >>> diff.to_excel('diff.xlsx')  #writes summary and dfs to file


    align inconsistently named columns before comparison:

    >>> corrections = {
            'year of birth': 'yob',
            'birthyear': 'yob',
            },
    >>> qp.diff(
            df_old,
            df_new,
            rename=corrections,
            )
    """
    diffs = Diffs(
        old,
        new,
        uid,
        ignore,
        rename,
        verbosity,
        )
    return diffs
