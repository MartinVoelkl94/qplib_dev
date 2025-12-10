
import pandas as pd
import copy

from .pandas import _format_df, deduplicate
from .excel import hide, format_excel
from .util import (
    log,
    _arg_to_list,
    GREEN,
    RED,
    GREEN_LIGHT,
    ORANGE_LIGHT,
    RED_LIGHT,
    )



def diff(
        old: pd.DataFrame | str,
        new: pd.DataFrame | str,
        uid=None,
        ignore=None,  #col(s) to ignore for comparison
        rename=None,  #rename cols before comparison
        verbosity=3,
        ) -> 'Diff':
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
    return Diff(
        old=old,
        new=new,
        uid=uid,
        ignore=ignore,
        rename=rename,
        verbosity=verbosity,
        )


class Diff:
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
            ignore=None,  #col(s) to ignore for comparison
            rename=None,  #rename cols before comparison
            verbosity=3,
            ):

        #original args
        self._old = old
        self._new = new
        self._uid = uid
        self._ignore = _arg_to_list(ignore)
        self._rename = rename
        self._verbosity = verbosity

        #processed data
        self.dfs_old = []
        self.dfs_new = []
        self.sheets = []
        self.sheet_in_both_files = []
        self.uid_cols = []
        self.cols_renamed_old = []
        self.cols_renamed_new = []
        self.cols_ignored_old = []
        self.cols_ignored_new = []
        self.cols_ignore = []
        self.cols_added = []
        self.cols_removed = []
        self.cols_shared = []
        self.rows_added = []
        self.rows_removed = []
        self.rows_shared = []
        self.vals_added = []
        self.vals_removed = []
        self.vals_changed = []
        self.dtypes_changed = []

        #process inputs
        self._get_dfs()
        self._format_dfs()
        self._rename_cols()
        self._get_uids()
        self._ignore_cols()
        self._get_diff_stats()


    def _get_dfs(self):
        """
        Load and prepare DataFrames from input sources.

        Handles loading from various sources (DataFrames, CSV files, Excel files)
        and determines whether to compare single sheets or multiple Excel sheets.
        Populates self.dfs_old, self.dfs_new, self.sheets, and self.sheet_in_both_files.
        """

        #when both inputs are excel files, they might
        #contain multiple sheets to be compared
        conditions_excel_comp = (
            isinstance(self._old, str)
            and isinstance(self._new, str)
            and self._old.endswith('.xlsx')
            and self._new.endswith('.xlsx')
            )
        if conditions_excel_comp:
            msg = 'debug: comparing all sheets from 2 excel files'
            log(msg, 'qp.diff()', self._verbosity)
            self._read_excel_sheets()


        #only 2 dfs need to be compared if input is
        #2 dfs, 2 csvs or 1 df and 1 csv/excel file
        else:

            if isinstance(self._new, str):
                if self._new.endswith('.csv'):
                    df_new = pd.read_csv(self._new)
                elif self._new.endswith('.xlsx'):
                    df_new = pd.read_excel(self._new)
                else:
                    msg = f'error: unknown file extension: {self._new}'
                    log(msg, 'qp.diff()', self._verbosity)
            elif isinstance(self._new, pd.DataFrame):
                df_new = self._new

            else:
                msg = 'error: incompatible type for new df'
                log(msg, 'qp.diff()', self._verbosity)

            if isinstance(self._old, str):
                if self._old.endswith('.csv'):
                    df_old = pd.read_csv(self._old)
                elif self._old.endswith('.xlsx'):
                    df_old = pd.read_excel(self._old)
                else:
                    msg = f'error: unknown file extension: {self._old}'
                    log(msg, 'qp.diff()', self._verbosity)
            elif isinstance(self._old, pd.DataFrame):
                df_old = self._old
            else:
                msg = 'error: incompatible type for old df'
                log(msg, 'qp.diff()', self._verbosity)

            self.dfs_old.append(df_old)
            self.dfs_new.append(df_new)
            self.sheets.append(None)
            self.sheet_in_both_files.append(None)


    def _read_excel_sheets(self):
        """
        Read all sheets from two Excel files.

        Identifies sheets present in both files, old file only, or new file only.
        Loads DataFrames for each sheet and tracks their comparison status.
        Populates sheet-specific data in the instance lists.
        """

        sheets_old = pd.ExcelFile(self._old).sheet_names
        sheets_new = pd.ExcelFile(self._new).sheet_names
        sheets_all = list(dict.fromkeys(sheets_new + sheets_old))  #preserves order

        for i, sheet in enumerate(sheets_all):

            if sheet in sheets_old and sheet in sheets_new:
                msg = f'debug: found sheet "{sheet}" in both files'
                log(msg, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'yes'
                df_old = pd.read_excel(self._old, sheet_name=sheet)
                df_new = pd.read_excel(self._new, sheet_name=sheet)

            elif sheet in sheets_old:
                msg = f'debug: sheet "{sheet}" is only in old file.'
                log(msg, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'only in old file'
                df_old = pd.read_excel(self._old, sheet_name=sheet)
                df_new = pd.DataFrame()

            elif sheet in sheets_new:
                msg = f'debug: sheet "{sheet}" is only in new file.'
                log(msg, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'only in new file'
                df_old = pd.DataFrame()
                df_new = pd.read_excel(self._new, sheet_name=sheet)

            self.dfs_old.append(df_old)
            self.dfs_new.append(df_new)
            self.sheets.append(sheet)
            self.sheet_in_both_files.append(sheet_in_both_files)


    def _format_dfs(self):
        """
        Apply consistent formatting to all loaded DataFrames.

        Standardizes DataFrame structure by fixing headers and adding metadata columns
        to ensure consistent comparison.
        """
        for i in range(len(self.sheets)):
            self.dfs_old[i] = _format_df(
                self.dfs_old[i],
                fix_headers=False,
                add_metadata=True,
                verbosity=2,
                )
            self.dfs_new[i] = _format_df(
                self.dfs_new[i],
                fix_headers=False,
                add_metadata=True,
                verbosity=2,
                )


    def _rename_cols(self):
        """
        Apply column renaming to DataFrames before comparison.

        Uses the rename dictionary to standardize column names between
        old and new DataFrames. Tracks which columns were renamed in each
        DataFrame and warns about potential issues with renaming metadata columns.
        """
        for i in range(len(self.sheets)):
            df_old = self.dfs_old[i]
            df_new = self.dfs_new[i]
            rename = self._rename
            rename_old = rename_new = ''

            if rename is None:
                pass
            elif isinstance(rename, dict):
                if 'meta' in rename:
                    msg = 'warning: it is not advised to rename the "meta" column'
                    log(msg, 'qp.diff()', self._verbosity)
                rename_old = {
                    old: new
                    for old, new
                    in rename.items()
                    if old in df_old.columns
                    }
                rename_new = {
                    old: new
                    for old, new
                    in rename.items()
                    if old in df_new.columns
                    }
                df_old.rename(columns=rename_old, inplace=True)
                df_new.rename(columns=rename_new, inplace=True)
            else:
                msg = 'error: rename argument must be a dictionary'
                log(msg, 'qp.diff()', self._verbosity)

            self.cols_renamed_old.append(rename_old)
            self.cols_renamed_new.append(rename_new)


    def _get_uids(self):
        """
        Determine unique identifier columns for each comparison.

        For each pair of dataframes, attempts to use the specified uid column or
        automatically finds the best alternative based on uniqueness and overlap
        between dataframes. Sets dataframe indices based on the selected uid column.
        Falls back to using the existing index if no suitable uid column is found.
        """

        for i, sheet in enumerate(self.sheets):

            df_old = self.dfs_old[i]
            df_new = self.dfs_new[i]

            if isinstance(self._uid, dict) and sheet in self._uid:
                uid = self._uid[sheet]
            elif self._uid in df_old.columns and self._uid in df_new.columns:
                uid = self._uid
            elif self._uid is False:
                uid = None
            else:
                if self._uid is None:
                    msg = 'debug: searching for alternative uid column'
                elif sheet is None:
                    msg = 'warning: no valid uid column specified for comparison'
                else:
                    msg = f'warning: no valid uid column specified for sheet "{sheet}"'
                log(msg, 'qp.diff()', self._verbosity)

                uids_potential = df_new.columns.intersection(df_old.columns)
                uids_by_uniqueness = {}
                for uid in uids_potential:
                    unique_in_old = pd.Index(df_old[uid].dropna()).unique()
                    unique_in_new = pd.Index(df_new[uid].dropna()).unique()
                    unique_shared = unique_in_new.intersection(unique_in_old)
                    uids_by_uniqueness[uid] = len(unique_shared)

                if len(uids_by_uniqueness) > 0:
                    uid = sorted(
                        uids_by_uniqueness.items(),
                        key=lambda item: item[1],
                        reverse=True,
                        )[0][0]
                    msg = f'debug: found alternative uid "{uid}" for comparison'
                    log(msg, 'qp.diff()', self._verbosity)
                else:
                    uid = None
                    msg = 'info: no alternative uid found. using index for comparison'
                    log(msg, 'qp.diff()', self._verbosity)

            if uid is not None:
                df_old.index = deduplicate(
                    df_old[uid],
                    name=f'{uid} in old df',
                    verbosity=self._verbosity,
                    )
                df_new.index = deduplicate(
                    df_new[uid],
                    name=f'{uid} in new df',
                    verbosity=self._verbosity,
                    )
            self.uid_cols.append(uid)


    def _ignore_cols(self):
        """
        Identify and track columns to ignore during comparison.

        Builds lists of columns to exclude from difference calculations,
        including user-specified ignore columns, metadata columns, and
        uid columns. Tracks which ignore columns are present in each DataFrame.
        """

        for i in range(len(self.sheets)):

            cols_ignore = self._ignore.copy()

            self.cols_ignored_old.append((
                self.dfs_old[i]
                .columns
                .intersection(cols_ignore)
                .to_list()
                ))

            self.cols_ignored_new.append((
                self.dfs_new[i]
                .columns
                .intersection(cols_ignore)
                .to_list()
                ))

            cols_ignore.append('meta')
            if self.uid_cols[i] is not None:
                cols_ignore.append(self.uid_cols[i])
            self.cols_ignore.append(cols_ignore)


    def _get_diff_stats(self):
        """
        Calculate comprehensive difference statistics for each sheet.

        Analyzes DataFrames to identify added/removed columns and rows,
        shared elements, and data type changes. This method populates
        the stats lists used by .summary(), .str() and .print().
        """

        for i in range(len(self.sheets)):
            df_old = self.dfs_old[i]
            df_new = self.dfs_new[i]
            cols_ignore = self.cols_ignore[i]

            self.cols_added.append((
                df_new
                .columns
                .difference(df_old.columns)
                .difference(cols_ignore)
                ))
            self.cols_removed.append((
                df_old
                .columns
                .difference(df_new.columns)
                .difference(cols_ignore)
                ))
            self.cols_shared.append((
                df_new
                .columns
                .intersection(df_old.columns)
                .difference(cols_ignore)
                ))

            self.rows_added.append((
                df_new
                .index
                .difference(df_old.index)
                ))
            self.rows_removed.append((
                df_old
                .index
                .difference(df_new.index)
                ))
            self.rows_shared.append((
                df_new
                .index
                .intersection(df_old.index)
                ))

            different_dtypes = {}
            for col in self.cols_shared[i]:
                if df_old[col].dtype != df_new[col].dtype:
                    changed = {
                        'old': df_old[col].dtype,
                        'new': df_new[col].dtype
                        }
                    different_dtypes[col] = changed
            if len(different_dtypes) > 0:
                self.dtypes_changed.append(different_dtypes)
            else:
                self.dtypes_changed.append(None)


    def show(
            self,
            mode='mix',
            sheet=0,
            prefix_old='old: ',
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

        sheet : int or str, default 0
            Sheet index or name to display in case multiple
            sheets from excel files were compared

        prefix_old : str, default 'old: '
            Prefix for columns showing old values (used in 'new+' mode)


        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with color-coded differences, or None if sheet not found
        """

        if sheet in self.sheets:
            ind = self.sheets.index(sheet)
        elif isinstance(sheet, int) and sheet < len(self.sheets):
            ind = sheet
        else:
            log(f'error: sheet "{sheet}" not found', 'qp.diff()', self._verbosity)
            return None

        df_old = self.dfs_old[ind]
        df_new = self.dfs_new[ind]
        uid_col = self.uid_cols[ind]
        cols_added = self.cols_added[ind]
        cols_removed = self.cols_removed[ind]
        cols_shared = self.cols_shared[ind]
        rows_added = self.rows_added[ind]
        rows_removed = self.rows_removed[ind]
        rows_shared = self.rows_shared[ind]


        if df_new.empty:
            df_diff = copy.deepcopy(df_old)
            df_diff_style = pd.DataFrame(
                f'background-color: {RED}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            df_diff['meta'] += 'removed sheet'


        elif df_old.empty:
            df_diff = copy.deepcopy(df_new)
            df_diff_style = pd.DataFrame(
                f'background-color: {GREEN}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            df_diff['meta'] += 'added sheet'


        elif mode in ['new', 'new+']:
            df_diff = copy.deepcopy(df_new)
            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            #add metadata columns
            if mode == 'new+':
                cols_new = ['meta']
                cols_add = []
                for col in df_diff.columns:
                    if not col.startswith(prefix_old) and col != 'meta':
                        cols_new.append(col)
                        if col != uid_col:
                            cols_new.append(prefix_old + col)
                            if prefix_old + col not in df_diff.columns:
                                cols_add.append(prefix_old + col)

                df_diff = pd.concat(
                    [
                        df_diff,
                        pd.DataFrame('', index=df_diff.index, columns=cols_add)
                    ],
                    axis=1,
                    )
                df_diff_style = pd.concat(
                    [
                        df_diff_style,
                        pd.DataFrame(
                            'font-style: italic',
                            index=df_diff.index,
                            columns=cols_add
                            )
                    ],
                    axis=1,
                    )
                df_diff = df_diff[cols_new]
                df_diff_style = df_diff_style[cols_new]


            df_diff_style.loc[:, cols_added] = f'background-color: {GREEN}'
            df_diff_style.loc[rows_added, :] = f'background-color: {GREEN}'

            df_diff.loc[rows_added, 'meta'] += 'added row'


        elif mode == 'old':
            df_diff = copy.deepcopy(df_old)
            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            df_diff.loc[rows_removed, 'meta'] += 'removed row'

        elif mode == 'mix':
            inds_old = df_old.index.difference(df_new.index)
            cols_old = df_old.columns.difference(df_new.columns)

            df_diff = pd.concat([df_new, df_old.loc[:, cols_old]], axis=1)
            df_diff.loc[inds_old, :] = df_old.loc[inds_old, :]

            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            df_diff_style.loc[:, cols_added] = f'background-color: {GREEN}'
            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_added, :] = f'background-color: {GREEN}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            df_diff.loc[rows_added, 'meta'] += 'added row'
            df_diff.loc[rows_removed, 'meta'] += 'removed row'

        else:
            log(f'error: unknown mode: {mode}', 'qp.diff()', self._verbosity)


        #highlight values in shared columns
        #column 0 contains metadata and is skipped
        cols_shared_no_metadata = [
            col for col
            in cols_shared
            if not col.startswith(prefix_old) and col != 'meta'
            ]

        df_old_isna = df_old.loc[rows_shared, cols_shared_no_metadata].isna()
        df_new_isna = df_new.loc[rows_shared, cols_shared_no_metadata].isna()
        df_new_equals_old = (
            df_new.loc[rows_shared, cols_shared_no_metadata]
            == df_old.loc[rows_shared, cols_shared_no_metadata]
            )

        df_added = df_old_isna & ~df_new_isna
        df_removed = df_new_isna & ~df_old_isna
        df_changed = (~df_new_isna & ~df_old_isna & ~df_new_equals_old).astype(bool)
        #the previous comparison can result in dtype "boolean" instead of "bool"
        #"boolean" masks cannot be used to set values as str

        df_diff_style.loc[rows_shared, cols_shared_no_metadata] += (
            df_added
            .mask(df_added, f'background-color: {GREEN_LIGHT}')
            .where(df_added, '')
            )

        df_diff_style.loc[rows_shared, cols_shared_no_metadata] += (
            df_removed
            .mask(df_removed, f'background-color: {RED_LIGHT}')
            .where(df_removed, '')
            )

        df_diff_style.loc[rows_shared, cols_shared_no_metadata] += (
            df_changed
            .mask(df_changed, f'background-color: {ORANGE_LIGHT}')
            .where(df_changed, '')
            )



        df_added_sum = df_added.sum(axis=1)
        df_removed_sum = df_removed.sum(axis=1)
        df_changed_sum = df_changed.sum(axis=1)

        self.vals_added.append(int(df_added_sum.sum()))
        self.vals_removed.append(int(df_removed_sum.sum()))
        self.vals_changed.append(int(df_changed_sum.sum()))

        df_diff.loc[rows_shared, 'meta'] += (
            df_added_sum
            .apply(lambda x: f'<br>vals added: {x}' if x > 0 else '')
            )
        df_diff.loc[rows_shared, 'meta'] += (
            df_removed_sum
            .apply(lambda x: f'<br>vals removed: {x}' if x > 0 else '')
            )
        df_diff.loc[rows_shared, 'meta'] += (
            df_changed_sum
            .apply(lambda x: f'<br>vals changed: {x}' if x > 0 else '')
            )


        if mode == 'new+':
            cols_shared_metadata = [prefix_old + col for col in cols_shared_no_metadata]
            df_all_modifications = (df_added | df_removed | df_changed)
            df_old_changed = (
                df_old
                .loc[rows_shared, cols_shared_no_metadata]
                .where(df_all_modifications, '')
                )
            df_diff.loc[rows_shared, cols_shared_metadata] = df_old_changed.values


        if len(df_diff.columns) * len(df_diff.index) > 100_000:
            msg = (
                'warning: more than 100 000 cells are being formatted.'
                'while this might not cause performance issues for formatting,'
                'the result might be slow to render, especially in jupyter notebooks.'
                )
            log(msg, 'qp.diff()', self._verbosity)

        #replace "<" and ">" with html entities to prevent interpretation as html tags
        cols_no_metadata = [
            col for col
            in df_diff.columns
            if not col.startswith(prefix_old) and col != 'meta'
            ]

        if pd.__version__ >= '2.1.0':
            df_diff.loc[:, cols_no_metadata] = (
                df_diff
                .loc[:, cols_no_metadata]
                .map(lambda x: _try_replace_gt_lt(x))
                )
        else:
            df_diff.loc[:, cols_no_metadata] = (
                df_diff
                .loc[:, cols_no_metadata]
                .applymap(lambda x: _try_replace_gt_lt(x))
                )


        diff_styled = df_diff.style.apply(lambda x: df_diff_style, axis=None)
        return diff_styled


    def summary(self, linebreak='<br>'):
        """
        Generate a comprehensive summary dataframe of differences between datasets.
        Dataframe contains statistics about added/removed columns and rows,
        shared elements, data type changes, and other metadata.

        Parameters
        ----------
        linebreak : str, default '&lt;br&gt;'
            String to use for line breaks in the summary output.
            Use '&lt;br&gt;' for HTML display or '\\n' for plain text.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame containing summary statistics with left-aligned,
            pre-wrapped text formatting
        """
        summary = pd.DataFrame()

        if len(self.sheets) > 1 or self.sheets[0] is not None:
            summary['sheets'] = self.sheets
            summary['in both files'] = self.sheet_in_both_files

        summary['uid col'] = _cols_to_str(self.uid_cols)
        summary['cols shared'] = [len(x) for x in self.cols_shared]
        summary['cols added'] = _iters_to_str(self.cols_added, linebreak)
        summary['cols removed'] = _iters_to_str(self.cols_removed, linebreak)
        summary['rows shared'] = [len(x) for x in self.rows_shared]
        summary['rows added'] = _iters_to_str(self.rows_added, linebreak)
        summary['rows removed'] = _iters_to_str(self.rows_removed, linebreak)

        #these 3 are calculated as part of show(), so only include them
        #if all sheets have been processed with show()
        if (
            len(self.vals_added)
            == len(self.vals_removed)
            == len(self.vals_changed)
            == len(self.sheets)
                ):
            summary['vals added'] = self.vals_added
            summary['vals removed'] = self.vals_removed
            summary['vals changed'] = self.vals_changed

        summary['dtypes changed'] = _nested_dicts_to_str(self.dtypes_changed, linebreak)
        summary['cols renamed in new'] = _dicts_to_str(self.cols_renamed_new, linebreak)
        summary['cols renamed in old'] = _dicts_to_str(self.cols_renamed_old, linebreak)
        summary['cols ignored in new'] = _iters_to_str(self.cols_ignored_new, linebreak)
        summary['cols ignored in old'] = _iters_to_str(self.cols_ignored_old, linebreak)

        summary = summary.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap',
            })
        return summary


    def to_excel(
            self,
            path,
            mode='new+',
            index=False,
            prefix_old='old: ',
            linebreak='\n',
            ):
        """
        Export diff results to an Excel file with formatting.

        Creates an Excel file containing a summary sheet and individual
        sheets for each comparison with highlighted differences. Applies
        Excel-specific formatting and hides old value columns.


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
        linebreak : str, default '\\n'
            String to use for line breaks in Excel cells
        """

        #placeholder. summary is updated based on the results
        #of the individual sheet comparisons, but should still
        #be at the beginning of the file.
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame().to_excel(
                writer,
                sheet_name='diff_summary',
                )

        for i, sheet in enumerate(self.sheets):

            if sheet is None:
                sheet = 'diff'

            if sheet == 'diff_summary':
                msg = (
                    'warning: comparison for sheet "diff_summary" will not'
                    ' be written to file since this name is reserved'
                    )
                log(msg, 'qp.diff()', self._verbosity)

            else:

                result = self.show(
                    mode=mode,
                    sheet=i,
                    prefix_old=prefix_old,
                    )

                result.data['meta'] = result.data['meta'].str.replace('<br>', '\n')

                with pd.ExcelWriter(path) as writer:
                    result.to_excel(
                        writer,
                        sheet_name=sheet,
                        index=index,
                        )

        with pd.ExcelWriter(path) as writer:
            self.summary(linebreak).to_excel(
                writer,
                sheet_name='diff_summary',
                index=index,
                )

        if mode == 'new+':
            hide(
                path,
                axis='col',
                patterns=f'{prefix_old}.*',
                hide=True,
                verbosity=self._verbosity,
                )
        format_excel(path)
        log(f'info: differences saved to "{path}"', 'qp.diff()', self._verbosity)


    def print(self):
        """
        Print the string representation of differences to console.

        Convenience method that prints the output of str() method,
        providing a readable summary of all differences found.
        """
        print(self.str())


    def str(self):
        """
        Get string representation of differences.

        Convenience method that calls __str__() to return a formatted
        string summary of all differences found between datasets.
        """
        return str(self)


    def __str__(self):
        """
        Generate a human-readable string summary of all differences.

        Creates a detailed text summary showing differences for single
        DataFrame comparisons or multi-sheet Excel file comparisons.
        Handles cases where datasets are identical.
        """
        summary = self.summary()
        if len(self.sheets) == 1 and self.sheets[0] is None:
            if self.dfs_new[0].equals(self.dfs_old[0]):
                string = 'both dataframes are identical'
            else:
                string = 'Diff between 2 dataframes\n'
                string += _sheet_to_str(summary, 0)
        else:
            string = f'Diff between 2 excel files with {len(self.sheets)} sheets\n'
            for i, sheet in enumerate(self.sheets):
                if self.dfs_new[i].equals(self.dfs_old[i]):
                    string += f'\nSheet "{sheet}" is identical in both files\n'
                else:
                    string += f'\nSheet: {sheet}\n'
                    string += _sheet_to_str(summary, i)
        return string


def _sheet_to_str(summary, row):
    string = ''
    one_liners = [
        'in both files',
        'uid col',
        'cols shared',
        'rows shared',
        ]
    for col in summary.data.columns:
        if col == 'sheets':
            continue
        elif col in one_liners:
            string += f'  {col}: {summary.data.loc[row, col]}\n'
        else:
            string += f'  {col}:\n    {summary.data.loc[row, col]}\n'
    return string.replace('<br>', '\n    ')


def _cols_to_str(iter):
    iter_new = []
    for col in iter:
        if col is None:
            iter_new.append('')
        else:
            iter_new.append(str(col))
    return iter_new


def _dicts_to_str(iter, linebreak='<br>'):
    iter_new = []
    for dictionary in iter:
        if dictionary:
            iter_new.append(
                f';{linebreak}'.join(
                    [f'{k} -> {v}' for k, v in dictionary.items()]
                    )
                )
        else:
            iter_new.append('')
    return iter_new


def _iters_to_str(iters, linebreak='<br>'):
    iters_new = []
    for iter in iters:
        iters_new.append(f';{linebreak}'.join([str(x) for x in iter]))
    return iters_new


def _nested_dicts_to_str(iter, linebreak='<br>'):
    iter_new = []
    for dictionary in iter:
        string = ''
        if dictionary:
            for k, v in dictionary.items():
                string += f'{k}: {v["old"]} -> {v["new"]}{linebreak}'
        iter_new.append(string)
    return iter_new


def _try_replace_gt_lt(x):
    if isinstance(x, str):
        return x.replace('<', '&lt;').replace('>', '&gt;')
    elif isinstance(x, type):
        return str(x).replace('<', '&lt;').replace('>', '&gt;')
    else:
        return x
