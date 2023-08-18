#========================

import glob as glob
import os
import pandas
from copy import deepcopy as copy

#========================
recursion = 2
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
                "entry":   0}

#========================

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

#========================

# Locate all folders and subfolders, to depth 'recursion', with an _init.md file
entries, folders, levels = scanfol("./", recursion)

# Write these to the blog index .md file
findex = open(index, 'w')
for entry, level in zip(entries, levels):

    folder = entry.replace("_init.dat","")
    
    # Load destination _init.dat
    finit = open(entry)

    # Extract data from _init.dat
    init_data = copy(default_init)
    for line in finit:
        line = line.replace(":\t","\t")
        while "\t\t" in line: line=line.replace("\t\t","\t")
        key, val = line.split("\t")
        while "\n" in val: val=val.replace("\n","")
        init_data = init_data|{key:val}
    finit.close()

    # Take data from the _init.md and write as a markdown line / url
    findex.write("\t"*(level-1))
    findex.write("  "*(level-1))
    findex.write("*")
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
    fout.write("\n")
    fheader.close()

#-----
# Write index
findex  = open(index,   'r')
for line in findex:
    fout.write(line)
fout.write("\n")
findex.close()

#-----
# Write Footer
if os.path.isfile(footer):
    ffooter = open(footer,  'r')
    for line in ffooter:
        fout.write(line)
    fout.write("\n")
    ffooter.close()

#-----
fout.close()
