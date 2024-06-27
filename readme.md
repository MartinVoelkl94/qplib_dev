# qplib

query and pandas utilities library for data exploration/analysis/modification of small to medium datasets.  
<br>
<br>
<br>


## Query function df.q()

Can be used on pandas Series and Index objects, but the main use is filtering Dataframes.  
The filtered results can then also be used to add/remove/change data and metadata in the Dataframe.  
It is implemented as an [accessor extension](https://pandas.pydata.org/docs/development/extending.html), meaning it can be called from a dataframe without any further preparation.  

An interactive wrapper can be called with df.qi() to help with the query creation and exploring the various options.  
Note that this wrapper is much slower and struggles with large datasets.  
<br>
<br>


## logging with qp.log()

A small logger to be used in notebooks. Makes it easier to keep track of outputs in large notebooks by providing color coded output and. Does not log to file, but instead to a dataframe in globals() which can then be viewed at the end of the notebook.
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
- qp.isfile()  
- qp.isdir()  
- qp.ispath()  
<br>
<br>


## type conversion 
Mostly wrappers for pandas functions but with some additional functionality. 

available functions:  
- qp.na()  
- qp.nk()  
- qp.num()  
- qp.yn()  
- qp.date()  
- qp.datetime()
<br>
<br>

