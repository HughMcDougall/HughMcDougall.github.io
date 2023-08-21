'''
_bloginit.py

A python file to assemble a tree of files available in all subdirectories here and hereafter.

TODO:
    - Generalize this to search for generic footers
    - Alternately, have a second .py file to add navigation headers / footers to all valid .md files, and a .bat to trigger this
    - Probably also want some kind of recursive trigger so I can fire these off for all sub folders if I want navigation sub-pages

    -HM 18/8
'''
#================================================================================================

import glob as glob
import os
import pandas
from copy import deepcopy as copy

#================================================================================================

recursion = 2
tree_depth = recursion

destfile = "./bloghome.md"

header  = "./_bloghead.md"
index   = "./_blogindex.md"
footer  = "./_blogfoot.md"
#========================

default_init = {
                "doc":     None,
                "title":   "n/a",
                "desc":    "n/a",
                "date":    "n/a",
                "series":  None,
                "entry":   0,
                "notebook_url": "n/a"}

#================================================================================================

def scanfol(url="./", recursion = 0, level = None, ):
    '''
    This is a recursive function that returns 3 x lists of every ._init.dat file
    to a recursive depth 'level' startin in root folder 'url'. Returns like:
    
        (url, fol, level)

    Where:

        url     [str]   Location of the init file
        fol     [str]   Folder in which the init file is located
        level   [int]   Recursion depth
    '''

    if level == None: level = recursion
    if level<0: return([],[],[])

    #----
    entries = []
    fols    = []
    levels  = []

    #----
    for entry in glob.glob(url+"*_init.dat"):
        entries.append(entry)
    
    #----
    for entry in glob.glob(url+"*/"):
        fols.append(entry)
    levels = [recursion-level]*len(entries)
    for fol in fols:
        fol_entries, fol_fols, fol_levels = scanfol(fol, recursion=recursion, level = level-1)

        for e,l in zip(fol_entries,fol_levels):
            entries.append(e)
            levels.append(l)

    return(entries,fols,levels)


def readinit(url):

    finit = open(url)
    out   = copy(default_init)
    for line in finit:
        line = line.replace(":\t","\t")
        while "\t\t" in line: line=line.replace("\t\t","\t")
        key, val = line.split("\t")
        while "\n" in val: val=val.replace("\n","")
        out = out|{key:val}
    finit.close()

    return(out)

#================================================================================================
# Build index

# Locate all folders and subfolders, to depth 'recursion', with an _init.md file
entries, folders, levels = scanfol("./", recursion)

# Write these to the blog index .md file
findex = open(index, 'w')
for entry, level in zip(entries, levels):

    folder = entry.replace("_init.dat","")
    
    # Load destination _init.dat
    init_data = readinit(entry)

    # Take data from the _init.md and write as a markdown line / url
    findex.write("\t"*(level-1)) # Readable indentation
    findex.write("  "*(level-1)) # markdown indentation
    findex.write("* ")
    findex.write("[%s](%s)" %(init_data["title"], folder+init_data["doc"].replace("./","").replace(".md",".html")))

    # Do markdown & html friendly line breaks
    findex.write("  ")
    findex.write("\n")


findex.close()

#========================
# Save everything to the actual human-readable .md file

fout = open(destfile, 'w')

#-----
# Write Header
if os.path.isfile(header):
    fheader = open(header,  'r')
    for line in fheader:
        fout.write(line)
    fout.write("  \n")
    fheader.close()

#-----
# Write index
findex  = open(index,   'r')
for line in findex:
    fout.write(line)
fout.write("  \n")
findex.close()

#-----
# Write Footer
if os.path.isfile(footer):
    ffooter = open(footer,  'r')
    for line in ffooter:
        fout.write(line)
    fout.write("  \n")
    ffooter.close()

#-----
fout.close()

#================================================================================================
# Construct navigation .md files for non-existant sub-folders 
