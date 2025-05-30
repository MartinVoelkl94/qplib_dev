import openpyxl
from .util import log, match


def hide(
    filename,
    axis='col',  #'row', 'col', 'sheet'
    patterns=None,
    regex=True,
    hide=True,  #True=hide, False=unhide
    verbosity=3,
    ): #pragma: no cover (does not affect reading of xlsx files)
    """
    Hide or unhide columns, rows, or sheets in an Excel file.
    """

    if hide==True:
        mode = 'hidden'
    else:
        mode = 'visible'

    wb = openpyxl.load_workbook(filename)

    if axis in ['col', 'column', 'cols', 'columns', 1]:
        for ws in wb.worksheets:
            hidden = []
            for col in ws.columns:
                if match(patterns, col[0].value, regex):
                    ws.column_dimensions[col[0].column_letter].hidden = hide
                    hidden.append(col[0].value)
            if hidden:
                hidden = '\n'.join(hidden)
                log(f'debug: columns made {mode} in "{filename}" sheet "{ws.title}":\n{hidden}', 'qp.hide()', verbosity)


    elif axis in ['row', 'rows', 'index', 0]:
        for ws in wb.worksheets:
            hidden = []
            for row in ws.rows:
                if match(patterns, row[0].value, regex):
                    ws.row_dimensions[row[0].row].hidden = hide
                    hidden.append(row[0].value)
            if hidden:
                hidden = '\n'.join(hidden)
                log(f'debug: rows made {mode} in "{filename}" sheet "{ws.title}":\n{hidden}', 'qp.hide()', verbosity)


    elif axis in ['sheet', 'worksheet', 'tab', 2]:
        hidden = []
        for ws in wb.worksheets:
            if match(patterns, ws.title, regex):
                if hide:
                    ws.sheet_state = mode
                else:
                    ws.sheet_state = mode
                hidden.append(ws.title)
        if hidden:
            hidden = '\n'.join(hidden)
            log(f'debug: sheets made {mode} in "{filename}":\n{hidden}', 'qp.hide()', verbosity)

    else:
        log(f'error: unknown axis "{axis}"', 'qp.hide', verbosity)

    wb.save(filename)
