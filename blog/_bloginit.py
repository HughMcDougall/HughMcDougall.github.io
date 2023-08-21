'''
_bloginit.py

A python file to assemble a tree of files available in all subdirectories here and hereafter.

TODO:
    - Change index construction to a function and trigger recursively
'''
#================================================================================================

import glob as glob
import os
import pandas
from copy import deepcopy as copy
import time

                               
#================================================================================================

recursion = 10  # Depth of folders in tree generation
tree_depth = 3  # Depth to display in navigation trees

KEYWORDS = ["NONE", "TRUE", "FALSE"]

destfile = "./bloghome.md"

#header  = "./_bloghead.md"
#index   = "./_blogindex.md"
#footer  = "./_blogfoot.md"

default_header  = "./_bloghead.md"
default_footer  = "./_blogfoot.md"

flog = open('./_flog.dat','w')

#========================

default_init = {
                "doc":      "NONE",
                "source":   "NONE",
                "title":   "NONE",
                "desc":    "NONE",
                "date":    "NONE",
                "series":  "NONE",
                "header":   "DEFAULT",
                "footer":   "DEFAULT",
                "entry":   0,
                "notebook_url": "NONE",
                "navheader": "TRUE",
                "tree": "TRUE",
                }

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
        if os.path.isfile(entry+"_init.dat"):
            fols.append(entry)
    levels = [recursion-level]*len(entries)
    newfols = []
    for fol in fols:
        fol_entries, fol_fols, fol_levels = scanfol(fol, recursion=recursion, level = level-1)
        if len(fol_fols)>0 and len(fol_entries)>0:
            newfols.append(fol_fols)

        for e,l in zip(fol_entries,fol_levels):
            entries.append(e)
            levels.append(l)
    
    if len(newfols)!=0:
        for newfol in newfols: fols.append(newfol)

    return(entries, fols, levels)

def readinit(url):
    '''
    Read a _init.md file to extract information into a dictionary
    '''

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

def flatten_list(X):
    out=[]
    if len(X)==0:
        return([])
    
    for x in X:
        if type(x)!=list:
            out.append(x)
        else:
            for y in flatten_list(x): out.append(y)
    return(out)

def _timef():
    '''Returns local time as a formatted string. For printing start/ end times'''
    
    t= time.localtime()
    wkdays = ["Mon","Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return "%02i:%02i %s %02i/%02i" %(t.tm_hour, t.tm_min, wkdays[t.tm_wday], t.tm_mday, t.tm_mon)

def relative_url(destination, source):
    absolute_path = os.path.dirname(__file__)
    relative_path = "src/lib"
    full_path = os.path.join(absolute_path, relative_path)
                               
#================================================================================================
# Build index
'''
Run through all folders and subfolders to generate
SCRAP THIS
'''

# Locate all folders and subfolders, to depth 'recursion', with an _init.md file
entries, folders, levels = scanfol("./", recursion)

'''
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
'''

#========================
# Save everything to the actual human-readable .md file
'''
SCRAP THIS
'''
'''
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
'''

#================================================================================================

flog.write(_timef()+"\n")
#Generate actual docs
for i, entry, level in zip(range(len(entries)), entries, levels):

    # Close all open files in case there was an error on the previous entry
    try: fout.close()
    except: pass
    try: f_foot.close()
    except: pass
    try: f_source.close()
    except: pass
    try: f_head.close()
    except: pass
    
    entryfol = entry.replace("_init.dat", "")
    print(entryfol)

    do_nav, do_header, do_tree, do_file, do_footer = False, False, False, False, False # Reset all

    #------------------------------------
    # LOAD & CHECK
    
    # load init data
    try:
        initdata = readinit(entry)
    except:
        flog.write("Something went wrong loading _init file %s \n" %entry)

    try:
        fout = open(entryfol + initdata["doc"], "w")            
    except:
        flog.write("unable to open destination file in entry %s \n" %(entry))
        continue

    # Locate next, previous & parent files
    prevfile, nextfile, parentfile = None, None, None
    prevfile_url, nextfile_url, parentfile_url = None, None, None
    if initdata["navheader"]!="FALSE":
        if i!=0:
            if levels[i]==levels[i-1]:
                prevfile_url = entries[i-1]
                prevfile = readinit(prevfile_url)
                prevfile_url = prevfile_url.replace("_init.dat", prevfile["doc"])
                prevfile_url=os.path.relpath(prevfile_url, entry).replace(".md",".html")
                            
            j = 0
            while i-j>=0 and levels[i-j]==levels[i]:
                j+=1
            parentfile_url = entries[i-j]
            parentfile = readinit(parentfile_url)
            parentfile_url = parentfile_url.replace("_init.dat", parentfile["doc"])
            parentfile_url =os.path.relpath(parentfile_url, entry).replace(".md",".html")
        
        if i!=len(entries)-1:
            if levels[i]==levels[i+1]:
                nextfile_url = entries[i+1]
                nextfile = readinit(nextfile_url)
                nextfile_url = nextfile_url.replace("_init.dat", nextfile["doc"])
                nextfile_url=os.path.relpath(nextfile_url, entry).replace(".md",".html")

        do_nav = True

    # Attempt to locate source and dest files
    if initdata["source"]!="NONE":
        f_source = open(entryfol+initdata["source"], "r")
        if not os.path.isfile(entryfol+initdata["source"]):
            flog.write("unable to find source file %s in entry %s \n" %(entryfol+initdata["source"], entry))
        do_file = True

    # Generate nav tree
    if initdata["tree"]!="FALSE":
        tree, treefols, tree_levels = scanfol(entryfol, tree_depth, tree_depth)
        do_tree = True

    # Attempt to load header and footer
    if initdata["header"]!="FALSE":
        # Check if a specific file is being nominated
        if initdata["header"] != "DEFAULT":
            try:
                f_head = open(entryfol+initdata["header"],'r')
                do_header = True
            except:
                flog.write("unable to find header file %s in entry %s \n" %(entryfol+initdata["header"], entry))        
        
        # Otherwise, check if there is a _header.md file
        if os.path.isfile(entryfol+"_header.md"):
            f_head = open(entryfol+"_header.md",'r')
            do_header = True
        
        # Otherwise, use the default header
        else:
            f_head = open(default_header,'r')
            do_header = True

    if initdata["footer"]!="FALSE":
        # Check if a specific file is being nominated
        if initdata["footer"] != "DEFAULT":
            try:
                f_foot = open(entryfol+initdata["footer"],'r')
                do_footer= True                
            except:
                flog.write("unable to find footer file %s in entry %s \n" %(entry, entryfol+initdata["footer"]))
        
        # Otherwise, check if there is a _header.md file
        if os.path.isfile(entryfol+"_footer.md"):
            f_foot = open(entryfol+"_footer.md",'r')
            do_footer = True
        
        # Otherwise, use the default header
        else:
            f_foot = open(default_footer,'r')
            do_footer = True
            
    #------------------------------------
    # FILE WRITE

    # write nav
    if do_nav:
        if prevfile !=None:
            fout.write("Previous File: [%s](%s)" %(prevfile["title"], prevfile_url))
            fout.write("\t &nbsp \t")
        if nextfile != None:
            fout.write("Next File: [%s](%s)" %(nextfile["title"], nextfile_url))
        fout.write("\n  ")

        if parentfile != None:
            fout.write("Parent File: [%s](%s)" %(parentfile["title"], parentfile_url))
            fout.write("\t &nbsp \t ")

        fout.write("Return to [blog home](%s)" %os.path.relpath(destfile.replace(".md",".html"), entry))
        fout.write("\n  ")
    # write header
    if do_header:
        for line in f_head:
            fout.write(line)
            fout.write("\n  ")

    # write tree
    if do_tree:
        # Scan entry's tree
        tree_entries, tree_folders, tree_levels = scanfol(entryfol, tree_depth)

        if len(tree_entries)>2:
            # Write these to the blog index .md file
            for tree_entry, tree_level in zip(tree_entries[1:], tree_levels[1:]):

                print("\t",tree_entry,tree_level)

                treeinit = readinit(tree_entry)
                tree_title  = treeinit['title']
                tree_relurl = os.path.relpath(tree_entry, entry).replace("_init.dat",treeinit['doc'])
                tree_relurl = tree_relurl.replace(".md",".html")

                print("\t\t",tree_title, tree_relurl)
                

                # Take data from the _init.md and write as a markdown line / url
                fout.write("\t"*(tree_level-1)) # Readable indentation
                fout.write("  "*(tree_level-1)) # markdown indentation
                fout.write("* ")
                fout.write("[%s](%s)" %(tree_title, tree_relurl))

                # Do markdown & html friendly line breaks
                fout.write("  ")
                fout.write("\n")
                
            
    # write doc
    if do_file:
        for line in f_source:
            fout.write(line)
            fout.write("\n  ")

    # write footer
    if do_footer:
        for line in f_foot:
            fout.write(line)
            fout.write("\n  ")

    #-----
    # FINISH & CLEANUP

    try: fout.close()
    except: pass
    try: f_foot.close()
    except: pass
    try: f_source.close()
    except: pass
    try: f_head.close()
    except: pass


#================================================================================================
flog.close()
