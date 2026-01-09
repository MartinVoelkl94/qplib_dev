
import pandas as pd
import openpyxl
import os

from .types import Container
from .pandas import deduplicate
from .excel import format_excel
from .util import (
    log,
    match,
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
            rename_cols=None,
            ignore_cols=None,
            name='data',
            verbosity=3,
            ):
        self.verbosity = verbosity
        self.name = name
        self.old = old.copy()
        self.new = new.copy()
        if rename_cols:
            self.rename_cols(rename_cols)
        self.ignore_cols(ignore_cols)  #sets: self.cols_ignore
        self.set_uid(uid)  #sets: self.uid, self.old.index, self.new.index
        if not self.old.index.name:
            self.old.index.name = 'index'
        if not self.new.index.name:
            self.new.index.name = 'index'

    def rename_cols(self, mapping):
        if not isinstance(mapping, dict):
            msg = (
                'error: mapping for renaming columns must'
                f' be a dict but is {type(mapping)!r}'
                )
            log(msg, 'qp.Diff', self.verbosity)
            return self
        self.old = self.old.rename(columns=mapping)
        self.new = self.new.rename(columns=mapping)
        msg = f'trace: renamed columns for {self.name!r}'
        log(msg, 'qp.Diff', self.verbosity)
        return self

    def ignore_cols(self, cols):
        if cols is None:
            cols_ignore = pd.Index([])
        else:
            cols_ignore = (
                self.old
                .columns
                .union(self.new.columns)
                .intersection(_arg_to_list(cols))
                )
        self.cols_ignore = cols_ignore
        return self


    def set_uid(self, uid=None):
        """
        Set the unique identifier (uid) column for
        comparing rows between old and new datasets.
        """
        #in case a uid was already set before
        if hasattr(self, 'uid') and self.uid in self.cols_ignore:
            msg = f'trace: removing old uid {self.uid!r} from .cols_ignore'
            log(msg, 'qp.Diff', self.verbosity)
            self.cols_ignore = self.cols_ignore.drop(self.uid)

        if uid is False:
            msg = 'trace: using index as uid'
            log(msg, 'qp.Diff', self.verbosity)
            uid = ''
        elif uid is None:
            msg = 'trace: searching for suitable uid column'
            log(msg, 'qp.Diff', self.verbosity)
            uid = self._find_uid()

        if uid in self.old.columns and uid in self.new.columns:
            self.old.index = self.old[uid]
            self.new.index = self.new[uid]
            self.cols_ignore = self.cols_ignore.append(pd.Index([uid]))

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

        uids_by_uniqueness = sorted(
            uids_by_uniqueness.items(),
            key=lambda item: item[1],
            reverse=True,
            )
        if len(uids_by_uniqueness) > 0:
            uid = uids_by_uniqueness[0][0]
            msg = f'debug: found uid {uid!r} for {self.name!r}'
            log(msg, 'qp.Diff', self.verbosity)
        else:
            uid = ''
            msg = f'debug: no uid found. using index for {self.name!r}'
            log(msg, 'qp.Diff', self.verbosity)

        return uid


    def details(self):
        """
        Detailed information about differences between datasets.
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
                    'old': self.old[col].dtype.name,
                    'new': self.new[col].dtype.name
                    }
                dtypes_changed[col] = changed

        details = Container()

        #basic info
        details.name = self.name
        details.uid = self.uid

        #numerical summary of changes
        details.cols_shared = len(cols_shared)
        details.cols_added = len(cols_added)
        details.cols_removed = len(cols_removed)
        details.rows_shared = len(rows_shared)
        details.rows_added = len(rows_added)
        details.rows_removed = len(rows_removed)
        details.dtypes_changed = len(dtypes_changed)

        #all changes
        details.cols_shared_all = cols_shared
        details.cols_added_all = cols_added
        details.cols_removed_all = cols_removed
        details.rows_shared_all = rows_shared
        details.rows_added_all = rows_added
        details.rows_removed_all = rows_removed
        details.dtypes_changed_all = dtypes_changed

        msg = 'debug: calculated (detailed) summary of differences'
        log(msg, 'qp.Diff', self.verbosity)
        return details


    def summary(self, head=5):
        """
        Summary of differences between datasets.
        """
        details = self.details()
        summary = Container()
        for key, val in details.items():
            if key.endswith('_all'):
                continue
            summary[key] = val
        return summary


    def show(
            self,
            mode='mix',
            prefix_old='old: ',
            linebreak='\n',
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
            Styled DataFrame with color-coded differences
        """

        old = self.old
        new = self.new
        uid = self.uid
        details = self.details()
        cols_added = details.cols_added_all
        cols_removed = details.cols_removed_all
        cols_shared = details.cols_shared_all
        rows_added = details.rows_added_all
        rows_removed = details.rows_removed_all
        rows_shared = details.rows_shared_all

        if mode not in ('new', 'new+', 'old', 'mix'):
            log(f'error: unknown mode: {mode}', 'qp.Diff', self.verbosity)
            raise ValueError(f'Unknown mode: {mode}')


        if old.empty and new.empty:
            df_diff = pd.DataFrame({'diff': ['empty datasets']})
            df_diff.index.name = 'index'
            diff_styled = df_diff.style
            return diff_styled

        elif old.empty:
            df_diff = new.copy()
            df_diff_style = pd.DataFrame(
                f'background-color: {GREEN}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, 'dataset added')
            diff_styled = df_diff.style.apply(lambda x: df_diff_style, axis=None)
            diff_styled = diff_styled.set_properties(white_space='pre-wrap')
            return diff_styled

        elif new.empty:
            df_diff = old.copy()
            df_diff_style = pd.DataFrame(
                f'background-color: {RED}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            col_diff = ensure_unique_string('diff', df_diff.columns)
            df_diff.insert(0, col_diff, 'dataset removed')
            diff_styled = df_diff.style.apply(lambda x: df_diff_style, axis=None)
            diff_styled = diff_styled.set_properties(white_space='pre-wrap')
            return diff_styled


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
            df_diff.loc[rows_added, col_diff] += 'row added'


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
            df_diff.loc[rows_removed, col_diff] += 'row removed'


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
            df_diff.loc[rows_added, col_diff] += 'row added'
            df_diff.loc[rows_removed, col_diff] += 'row removed'



        #highlight values in shared columns

        #replace "<" and ">" with html entities to prevent interpretation as html tags
        #doing this also impacts dtypes, which might be more problematic than the odd
        #html tag
        # if pd.__version__ >= '2.1.0':
        #     df_diff = df_diff.map(lambda x: _replace_gt_lt(x))
        # else:
        #     df_diff = df_diff.applymap(lambda x: _replace_gt_lt(x))

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

        removed_and_changed = removed.intersection(changed)
        added_and_removed = added.intersection(removed)
        added_and_changed = added.intersection(changed)
        added_and_removed_or_changed = added_and_removed.union(added_and_changed)

        df_diff.loc[added, col_diff] += 'vals added: '
        df_diff.loc[added, col_diff] += sum_added[added].astype(str)
        df_diff.loc[added_and_removed_or_changed, col_diff] += linebreak

        df_diff.loc[removed, col_diff] += 'vals removed: '
        df_diff.loc[removed, col_diff] += sum_removed[removed].astype(str)
        df_diff.loc[removed_and_changed, col_diff] += linebreak

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

        diff_styled = (
            df_diff
            .style
            .apply(lambda x: df_diff_style, axis=None)
            .set_properties(white_space='pre-wrap')
            )
        msg = 'debug: created df with highlighted differences'
        log(msg, 'qp.Diff', self.verbosity)
        return diff_styled


    def str(self):
        string = f'Diff of {self.name!r}:\n'
        if self.old.empty and self.new.empty:
            string += '  both datasets are empty\n'
        elif self.old.empty:
            string += '  old dataset is empty\n'
        elif self.new.empty:
            string += '  new dataset is empty\n'
        elif self.old.equals(self.new):
            string += '  datasets are identical\n'
        else:
            details = self.details()
            cols_shared_str = _to_str(
                details.cols_shared_all,
                linebreak='\n  ',
                )
            cols_added_str = _to_str(
                details.cols_added_all,
                linebreak='\n  ',
                )
            cols_removed_str = _to_str(
                details.cols_removed_all,
                linebreak='\n  ',
                )
            rows_shared_str = _to_str(
                details.rows_shared_all,
                linebreak='\n  ',
                )
            rows_added_str = _to_str(
                details.rows_added_all,
                linebreak='\n  ',
                )
            rows_removed_str = _to_str(
                details.rows_removed_all,
                linebreak='\n  ',
                )
            dtypes_changed_str = _to_str(
                details.dtypes_changed_all,
                linebreak='\n  ',
                )
            string += (
                f' cols shared: {details.cols_shared}\n'
                f' cols added: {details.cols_added}\n'
                f' cols removed: {details.cols_removed}\n'
                f' rows shared: {details.rows_shared}\n'
                f' rows added: {details.rows_added}\n'
                f' rows removed: {details.rows_removed}\n'
                f' dtypes changed: {details.dtypes_changed}\n'
                f' all cols shared:\n  {cols_shared_str}\n'
                f' all cols added:\n  {cols_added_str}\n'
                f' all cols removed:\n  {cols_removed_str}\n'
                f' all rows shared:\n  {rows_shared_str}\n'
                f' all rows added:\n  {rows_added_str}\n'
                f' all rows removed:\n  {rows_removed_str}\n'
                f' all dtypes changed:\n  {dtypes_changed_str}\n'
                )
        return string

    def print(self):
        print(self.str())
        return self


class Diffs:
    """
    Stores differences between (multiple) datasets.
    Used as return type of the diff() function.
    """

    def __init__(
            self,
            old: pd.DataFrame | str,
            new: pd.DataFrame | str,
            uid=None,
            rename_cols=None,
            ignore_cols=None,
            verbosity=3,
            ):
        self.verbosity = verbosity
        self.all = self._get_Diffs(
            old=old,
            new=new,
            uid=uid,
            rename_cols=rename_cols,
            ignore_cols=ignore_cols,
            )
        self.cols_summary = [
            'uid',
            'in both datasets',
            'cols shared',
            'cols added',
            'cols removed',
            'rows shared',
            'rows added',
            'rows removed',
            'dtypes changed',
            ]


    def _get_Diffs(
            self,
            old,
            new,
            uid=None,
            rename_cols=None,
            ignore_cols=None,
            ):
        """
        Loads dataframes from various sources (pd.DataFrame, CSV files, Excel files)
        and determines whether to compare single sheets or multiple Excel sheets.
        """
        msg = 'Debug: getting data for Diffs'
        log(msg, 'qp.Diffs', self.verbosity)
        self.old = old
        self.new = new

        #when both inputs are excel files, they might
        #contain multiple sheets to be compared
        conditions_excel_comp = (
            isinstance(old, str)
            and isinstance(new, str)
            and old.endswith('.xlsx')
            and new.endswith('.xlsx')
            )
        if conditions_excel_comp:
            diffs = self._get_excel_Diffs(
                old=old,
                new=new,
                uid=uid,
                rename_cols=rename_cols,
                ignore_cols=ignore_cols,
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
                    log(msg, 'qp.Diffs', self.verbosity)
                    raise ValueError(msg)
            elif isinstance(old, pd.DataFrame):
                df_old = old
            else:
                msg = 'error: incompatible type for old df'
                log(msg, 'qp.Diffs', self.verbosity)
                raise ValueError(msg)

            if isinstance(new, str):
                if new.endswith('.csv'):
                    df_new = pd.read_csv(new)
                elif new.endswith('.xlsx'):
                    df_new = pd.read_excel(new)
                else:
                    msg = f'error: unknown file extension: {new}'
                    log(msg, 'qp.Diffs', self.verbosity)
            elif isinstance(new, pd.DataFrame):
                df_new = new

            else:
                msg = 'error: incompatible type for new df'
                log(msg, 'qp.Diffs', self.verbosity)

            diff = Diff(
                old=df_old,
                new=df_new,
                uid=uid,
                rename_cols=rename_cols,
                ignore_cols=ignore_cols,
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
            rename_cols=None,
            ignore_cols=None,
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

            if isinstance(uid, dict):
                uid_sheet = uid.get(sheet, None)
            else:
                uid_sheet = uid

            diff = Diff(
                old=df_old,
                new=df_new,
                uid=uid_sheet,
                rename_cols=rename_cols,
                ignore_cols=ignore_cols,
                name=sheet,
                verbosity=self.verbosity,
                )
            diff.sheet = sheet
            diff.in_both_datasets = in_both_datasets
            diffs.append(diff)

        msg = 'debug: created Diffs for 2 excel files'
        log(msg, 'qp.Diffs', self.verbosity)
        return diffs

    def rename_cols(self, mappings):
        for sheet, mapping in mappings.items():
            diff = self[sheet]
            if diff is not None:
                diff.rename_cols(mapping)
        return self

    def ignore_cols(self, cols):
        if isinstance(cols, dict):
            for sheet, cols_sheet in cols.items():
                diff = self[sheet]
                if diff is not None:
                    diff.ignore_cols(cols_sheet)
        else:
            for diff in self.all:
                diff.ignore_cols(cols)
        return self

    def set_uid(self, uids=None):
        if isinstance(uids, dict):
            for sheet, uid in uids.items():
                diff = self[sheet]
                if diff is not None:
                    diff.set_uid(uid)
        else:
            for diff in self.all:
                diff.set_uid(uids)
        return self

    def __getitem__(self, key):
        if isinstance(key, int):
            item = self.all[key]
        else:
            for diff in self.all:
                if diff.name == key:
                    item = diff
                    break
            else:
                msg = f'error: sheet "{key}" not found in diffs'
                log(msg, 'qp.Diff', self.verbosity)
                return None
        return item


    def info(self):
        """
        Basic information about the datasets.
        """
        if isinstance(self.old, str):
            name_old = self.old
            size_old = os.path.getsize(self.old) / 1024
        else:
            name_old = type(self.old).__name__
            size_old = self.old.memory_usage(deep=True).sum() / 1024
        if isinstance(self.new, str):
            name_new = self.new
            size_new = os.path.getsize(self.new) / 1024
        else:
            name_new = type(self.new).__name__
            size_new = self.new.memory_usage(deep=True).sum() / 1024
        data = {
            'name': [name_old, name_new],
            'size (KB)': [size_old, size_new],
            }
        info = pd.DataFrame(
            data,
            index=['old dataset', 'new dataset'],
            )
        info.index.name = 'dataset'
        return info


    def details(
            self,
            separator=',',
            linebreak='\n',
            ):
        """
        Detailed information about differences between datasets.
        """

        datasets = [diff.name for diff in self.all]
        data = {
            'uid': [diff.uid for diff in self.all],
            'in both datasets': [diff.in_both_datasets for diff in self.all],
            }
        details = pd.DataFrame(data, index=datasets)
        details.index.name = 'dataset'

        cols = {
            #numerical summary
            'cols_shared': 'cols shared',
            'cols_added': 'cols added',
            'cols_removed': 'cols removed',
            'rows_shared': 'rows shared',
            'rows_added': 'rows added',
            'rows_removed': 'rows removed',
            'dtypes_changed': 'dtypes changed',
            #all changes
            'cols_shared_all': 'all cols shared',
            'cols_added_all': 'all cols added',
            'cols_removed_all': 'all cols removed',
            'rows_shared_all': 'all rows shared',
            'rows_added_all': 'all rows added',
            'rows_removed_all': 'all rows removed',
            'dtypes_changed_all': 'all dtypes changed',
            }

        for diff in self.all:
            diff_details = diff.details()
            for key, col in cols.items():
                string = _to_str(
                    diff_details[key],
                    separator,
                    linebreak,
                    )
                details.loc[diff.name, col] = string

        details = details.style.set_properties(**{
            # 'text-align': 'left',
            'white-space': 'pre-wrap',
            })
        return details


    def summary(
            self,
            separator=',',
            linebreak='\n',
            ):
        """
        Summary of differences between datasets.
        """
        details = self.details(separator, linebreak).data
        summary = details[self.cols_summary]
        summary = summary.style.set_properties(**{
            # 'text-align': 'left',
            'white-space': 'pre-wrap',
            })
        return summary


    def show(
            self,
            mode='mix',
            sheet=0,
            prefix_old='old: ',
            linebreak='\n',
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

        sheet : str | int, default 0
            Sheet name or index to show differences for (if there are multiple sheets)

        prefix_old : str, default 'old: '
            Prefix for columns showing old values (used in 'new+' mode)

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with color-coded differences
        """
        diff = self[sheet]
        result = diff.show(
            mode,
            prefix_old,
            linebreak,
            )
        return result


    def to_excel(
            self,
            path,
            mode='new+',
            index=True,
            prefix_old='old: ',
            hide_info=True,
            hide_details=True,
            hide_summary=False,
            summary_separator=',',
            summary_linebreak='\n',
            ):
        """
        Export diff results to an Excel file with formatting.

        Creates an Excel file containing a summary sheet and individual
        sheets for each comparison with highlighted differences. Applies
        Excel-specific formatting and hides columns with old values.


        Parameters
        ----------
        path : str
            File path for the output Excel file
        mode : str, default 'mix'
            Display mode for differences (see show() method for options)
        index : bool, default False
            Whether to include row indices in the Excel output
        prefix_old : str, default 'old: '
            Prefix for columns showing old values (used in 'new+' mode)
        hide_details : bool, default True
            Whether to hide the details sheets in the Excel file
        hide_summary : bool, default False
            Whether to hide the summary sheet in the Excel file
        summary_separator : str, default ','
            Separator string for lists in summary/details
        summary_linebreak : str, default '\\n'
            String to use for line breaks in summary/details
        """

        with pd.ExcelWriter(path) as writer:

            info = self.info()
            sheet_info = ensure_unique_string(
                'info',
                info.index,
                )
            info.to_excel(
                writer,
                sheet_name=sheet_info,
                index=True,
                )

            summary = self.summary(
                summary_separator,
                summary_linebreak,
                )
            sheet_summary = ensure_unique_string(
                'summary',
                summary.index,  #contains all sheet names
                )
            summary.to_excel(
                writer,
                sheet_name=sheet_summary,
                index=True,
                )

            details = self.details(
                summary_separator,
                summary_linebreak,
                )
            sheet_details = ensure_unique_string(
                'details',
                details.index,  #contains all sheet names
                )
            details.to_excel(
                writer,
                sheet_name=sheet_details,
                index=True,
                )

            for diff in self.all:
                result = diff.show(
                    mode,
                    prefix_old,
                    summary_linebreak,
                    )
                result.to_excel(
                    writer,
                    sheet_name=diff.name,
                    index=index,
                    )

        #post-process excel file
        if hide_details or hide_summary or mode == 'new+':
            wb = openpyxl.load_workbook(path)
            for ws in wb.worksheets:
                if hide_info and ws.title == sheet_info:
                    ws.sheet_state = 'hidden'
                if hide_summary and ws.title == sheet_summary:
                    ws.sheet_state = 'hidden'
                elif hide_details and ws.title == sheet_details:
                    ws.sheet_state = 'hidden'
                else:
                    for col in ws.columns:
                        if match(f'{prefix_old}.*', col[0].value, regex=True):
                            ws.column_dimensions[col[0].column_letter].hidden = True
            wb.save(path)
            wb.close()
        format_excel(path)
        log(f'info: differences saved to "{path}"', 'qp.Diff', self.verbosity)


    def str(self):
        string = 'Diffs summary:'
        for diff in self.all:
            string += '\n  ' + diff.str().replace('\n', '\n  ')
        return string

    def print(self):
        print(self.str())
        return self

    def __iter__(self):
        for diff in self.all:
            yield diff


def _to_str(obj, separator=',', linebreak='\n'):

    if isinstance(obj, int):
        if obj == 0:
            string = ''
        else:
            string = str(obj)

    elif isinstance(obj, dict):
        if len(obj) == 0:
            string = ''
        else:
            strings = []
            for key, val in obj.items():
                val_old = val['old']
                val_new = val['new']
                strings.append(f'{key!r}: {val_old!r} -> {val_new!r}')
            string = f'{separator}{linebreak}'.join(strings)

    elif isinstance(obj, (list, set, tuple, pd.Index)):
        if len(obj) == 0:
            string = ''
        else:
            items = [f'{x!r}' for x in obj]
            string = f'{separator}{linebreak}'.join(items)

    else:
        string = str(obj)

    return string


def diff(
        old: pd.DataFrame | str,
        new: pd.DataFrame | str,
        uid=None,
        rename_cols=None,
        ignore_cols=None,
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

    rename_cols : dictionary to rename columns before comparison.
        Note that renaming is done before uid and columns to ignore
        are determined, meaning that those must use the new column names.
        This can also be used to fix situations where corrresponding columns
        in old and new data have different names (see examples)

    ignore_cols : column name or list of column names to ignore for comparison


    Examples
    --------

    basic usage:

    >>> import qplib as qp
    >>> diffs = qp.diff('old.xlsx', 'new.xlsx')
    >>> diffs.summary()  #returns df with summary stats
    >>> diffs.details()  #returns df with more detailed stats
    >>> diffs.show()  #returns df.style with highlighted differences
    >>> diffs.str()  #string version of summary stats
    >>> diffs.print()  #prints the string version
    >>> diffs.to_excel('diffs.xlsx')  #writes summary and dfs to file


    align inconsistently named columns before comparison:

    >>> corrections = {
            'year of birth': 'yob',
            'birthyear': 'yob',
            },
    >>> qp.diff(
            'old.xlsx',
            'new.xlsx',
            rename=corrections,
            )
    """
    diffs = Diffs(
        old=old,
        new=new,
        uid=uid,
        rename_cols=rename_cols,
        ignore_cols=ignore_cols,
        verbosity=verbosity,
        )
    return diffs
