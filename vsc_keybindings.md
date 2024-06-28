# VSC_keybindings

// Place your key bindings in this file to override the defaults
[


//general snippets

//imports
{ "key": "alt+i",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {
    "snippet": "import pandas as pd\nimport numpy as np\nimport copy\nimport os\nimport sys\nimport shutil\nimport datetime\nimport qplib as qp\nfrom qplib import log\npd.set_option('display.max_columns', None)"}
},

//profiler
{ "key": "alt+p",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {
    "snippet": "import cProfile, pstats\n\nprofiler = cProfile.Profile()\nprofiler.enable()\n\n\n\nprofiler.disable()\nstats = pstats.Stats(profiler).sort_stats('tottime')\nstats.print_stats(10)"}
},






//pandas snippets

{ "key": "Alt+s",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {
    "snippet": "pd.Series([])"}
},

{ "key": "alt+d alt+d",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "pd.DataFrame()"}
},

{ "key": "alt+d alt+s",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "df = pd.DataFrame({\n    'ID': [10001, 10002, 10003, 20001, 20002, 20003, 30001, 30002, 30003, 30004, 30005],\n    'name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'eva white', 'Frank miller', 'Grace TAYLOR', 'Harry Clark', 'IVY GREEN', 'JAck Williams', 'john Doe'],\n    'date of birth': ['1995-01-02', '1990/09/14', '1985.08.23', '19800406', '05-11-2007', '06-30-1983', '28-05-1975', '1960Mar08', '1955-Jan-09', '1950 Sep 10', '1945 October 11'],\n    'age': [-25, '30', np.nan, None, '40.0', 'forty-five', 'nan', 'unk', '', 'unknown', 35],\n    'gender': ['M', 'F', 'Female', 'Male', 'Other', 'm', 'ff', 'NaN', None, 'Mal', 'female'],\n    'height': [170, '175.5cm', None, '280', 'NaN', '185', '1', '6ft 1in', -10, '', 200],\n    'weight': [70.2, '68', '72.5lb', 'na', '', '75kg', None, '80.3', '130lbs', '82', -65],\n    'bp systole': ['20', 130, 'NaN', '140', '135mmhg', '125', 'NAN', '122', '', 130, '45'],\n    'bp diastole': [80, '85', 'nan', '90mmHg', np.nan, '75', 'NaN', None, '95', '0', 'NaN'],\n    'cholesterol': ['Normal', 'Highe', 'NaN', 'GOOD', 'n.a.', 'High', 'Normal', 'n/a', 'high', '', 'Normal'],\n    'diabetes': ['No', 'yes', 'N/A', 'No', 'Y', 'Yes', 'NO', None, 'NaN', 'n', 'Yes'],\n    'dose': ['10kg', 'NaN', '15 mg once a day', '20mg', '20 Mg', '25g', 'NaN', None, '30 MG', '35', '40ml']\n    })"}
},


{ "key": "alt+d alt+q",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": ".q(\n\t'','','',\n\t'','','',\n\t'','','',\n\t'','','',\n\t'','','',\n\tdiff='mix',\n\tmax_cols=200,\n\tmax_rows=20,\n\tinplace=True,\n\tverbosity=3,\n\t)"}
},



//logging snippets

{ "key": "alt+l alt+i",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "qp.log('', level='info', source='')"}
},

{ "key": "alt+l alt+w",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "qp.log('', level='warning', source='')"}
},

{ "key": "alt+l alt+e",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "qp.log('', level='error', source='')"}
},

{ "key": "alt+l alt+c",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "qp.log(clear=True)"}
},

{ "key": "alt+l alt+l",
  "command": "editor.action.insertSnippet",
  "when": "editorTextFocus",
  "args": {"snippet": "qp.log()"}
},





//other

{ "key": "alt+l",
  "command": "-toggleFindInSelection",
  "when": "editorFocus"
},

{ "key": "alt+l",
  "command": "-toggleSearchEditorContextLines",
  "when": "inSearchEditor"
},

{ "key": "alt+v",
  "command": "jupyter.openVariableView"
},

{ "key": "alt+a",
  "command": "notebook.cell.executeCellsAbove"
},

{ "key": "alt+b",
  "command": "notebook.cell.executeCellAndBelow"
},

{
    "key": "alt+g",
    "command": "markdown-links.showGraph"
},
{
    "key": "ctrl+n",
    "command": "-workbench.action.files.newUntitledFile"
},
{
    "key": "ctrl+n",
    "command": "vscodeMarkdownNotes.newNote"
},
{
    "key": "ctrl+enter",
    "command": "-workbench.action.chat.insertCodeBlock",
    "when": "accessibilityModeEnabled && hasChatProvider && inChat"
},
{
    "key": "ctrl+enter",
    "command": "-github.copilot.generate",
    "when": "editorTextFocus && github.copilot.activated && !inInteractiveInput && !interactiveEditorFocused"
},
{
    "key": "alt+f alt+a",
    "command": "editor.foldAll",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+0",
    "command": "-editor.foldAll",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+u",
    "command": "editor.unfoldAll",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+j",
    "command": "-editor.unfoldAll",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+1",
    "command": "editor.foldLevel1",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+1",
    "command": "-editor.foldLevel1",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+2",
    "command": "editor.foldLevel2",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+2",
    "command": "-editor.foldLevel2",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+3",
    "command": "editor.foldLevel3",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+3",
    "command": "-editor.foldLevel3",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+4",
    "command": "editor.foldLevel4",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+4",
    "command": "-editor.foldLevel4",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+5",
    "command": "editor.foldLevel5",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+5",
    "command": "-editor.foldLevel5",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+6",
    "command": "editor.foldLevel6",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+6",
    "command": "-editor.foldLevel6",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "alt+f alt+7",
    "command": "editor.foldLevel7",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+k ctrl+7",
    "command": "-editor.foldLevel7",
    "when": "editorTextFocus && foldingEnabled"
},
{
    "key": "ctrl+enter",
    "command": "-github.copilot.generate",
    "when": "editorTextFocus && github.copilot.activated && !commentEditorFocused && !inInteractiveInput && !interactiveEditorFocused"
},
{
    "key": "ctrl+enter",
    "command": "python.execInREPL",
    "when": "config.python.REPL.sendToNativeREPL && editorTextFocus && !jupyter.ownsSelection && editorLangId == 'python' && activeEditor != 'workbench.editor.interactive'"
},
{
    "key": "shift+enter",
    "command": "-python.execInREPL",
    "when": "config.python.REPL.sendToNativeREPL && editorTextFocus && !jupyter.ownsSelection && editorLangId == 'python' && activeEditor != 'workbench.editor.interactive'"
},
{
    "key": "ctrl+enter",
    "command": "-editor.action.insertLineAfter",
    "when": "editorTextFocus && !editorReadonly"
},
{
    "key": "alt+enter",
    "command": "-ipython.runSection",
    "when": "editorTextFocus && ipython.isUse && !inDebugMode && editorLangId == 'python'"
},
{
    "key": "alt+enter",
    "command": "-notebook.cell.executeAndInsertBelow",
    "when": "notebookCellListFocused && notebookCellType == 'markup' || notebookCellListFocused && notebookMissingKernelExtension && !notebookCellExecuting && notebookCellType == 'code' || notebookCellListFocused && !notebookCellExecuting && notebookCellType == 'code' && notebookKernelCount > 0 || notebookCellListFocused && !notebookCellExecuting && notebookCellType == 'code' && notebookKernelSourceCount > 0"
},
{
    "key": "alt+enter",
    "command": "editor.action.inlineSuggest.commit",
    "when": "inlineSuggestionHasIndentationLessThanTabSize && inlineSuggestionVisible && !editorHoverFocused && !editorTabMovesFocus && !suggestWidgetVisible"
},
{
    "key": "tab",
    "command": "-editor.action.inlineSuggest.commit",
    "when": "inlineSuggestionHasIndentationLessThanTabSize && inlineSuggestionVisible && !editorHoverFocused && !editorTabMovesFocus && !suggestWidgetVisible"
}
]