# qplib

A query language for pandas data exploration/analysis/modification.

It is implemented as an [accessor extension](https://pandas.pydata.org/docs/development/extending.html), meaning it can be called from a dataframe without any further preparation by calling df.q() with a valid text string.  

An interactive wrapper can be called with df.qi() to help with query creation and for exploring the syntax.  

Currently published on [testpypi](https://test.pypi.org/project/qplib/).
<br>
<br>




# other utilities


## logging with qp.log()

A small logger to be used in notebooks or the REPL. Makes it easier to keep track of outputs in large notebooks by providing color coded output. Does not log to file, but instead to a dataframe which can then be viewed at the end of the notebook.
<br>
<br>


## "bashlike" wrappers

While python has functions to achieve similar results as common bash commands, they are often more verbose, less intuitive if you are already used to the bash names and spread out over different modules and different namespaces in those modules (os, os.path, shutil, sys, ...).  
These wrappers use the same names as the bash commands and offer some additional functionality.


available wrappers:  
- qp.ls()  
- qp.lsr()  
- qp.pwd()  
- qp.cd()  
- qp.cp()  
- qp.mkdir()  
- qp.isdir()  
- qp.isfile()  
- qp.ispath()  
<br>
<br>


## type conversion 

Mostly wrappers for pandas functions but with some additional functionality and generally more lenient handling of edge cases. 

available functions:  
- qp.int()  
- qp.float()  
- qp.num()  
- qp.bool()  
- qp.date()  
- qp.datetime()  
- qp.na()  
- qp.nk()  
- qp.yn()  
- qp.type()  
<br>
<br>

