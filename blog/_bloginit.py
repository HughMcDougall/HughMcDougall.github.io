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

replacements = [
    (r"\begin{equation}", r"$$\begin{equation}"),
    (r"\end{equation}", r"\end{equation}$$"),
                ]
                               
#================================================================================================

recursion = 10  # Depth of folders in tree generation
tree_depth = 3  # Depth to display in navigation trees

KEYWORDS = ["NONE", "TRUE", "FALSE"] # Deprecated
tab = " "

destfile = "./bloghome.md"


# Default files to use if none specified
default_header  = "./_components/_bloghead.md"
default_footer  = "./_components/_blogfoot.md"

flog = open('./_flog.dat','w')

#========================

# Default init file
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
                "notebook": "NONE",
                "navheader": "TRUE",
                "tree": "TRUE",
                "render": False
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
        line = line.replace(": ",":\t")
        line = line.replace(":\t","\t")
        while "\t\t" in line: line=line.replace("\t\t","\t")
        key, val = line.split("\t")
        while "\n" in val: val=val.replace("\n","")
        out = out|{key:val}
    finit.close()

    return(out)

def flatten_list(X):
    '''
    Takes a list of lists and flattens to a single level
    '''
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
'''

# Locate all folders and subfolders, to depth 'recursion', with an _init.md file
entries, folders, levels = scanfol("./", recursion)

#================================================================================================
# File Writing

flog.write(_timef()+"\n") # Log message

#Generate actual docs
for i, entry, level in zip(range(len(entries)), entries, levels):

    #=====
    # Close all open files in case there was an error on the previous entry
    try: fout.close()
    except: pass
    try: f_foot.close()
    except: pass
    try: f_source.close()
    except: pass
    try: f_head.close()
    except: pass
    #=====
    
    entryfol = entry.replace("_init.dat", "")
    print(entryfol)

    do_nav, do_header, do_tree, do_file, do_footer = False, False, False, False, False # Reset all

    #------------------------------------
    # LOAD & CHECK

    #=====
    # load init data
    
    try:
        initdata = readinit(entry)
    except:
        flog.write("Something went wrong loading _init file %s \n" %entry)

    try:
        fout = open(entryfol + initdata["doc"], "w", encoding="utf8")     
    except:
        flog.write("unable to open destination file in entry %s \n" %(entry))
        continue
    #=====

    # Locate next, previous & parent files
    prevfile, nextfile, parentfile = None, None, None
    prevfile_url, nextfile_url, parentfile_url = None, None, None
    if initdata["navheader"]!="FALSE":
        if i!=0:
            if levels[i]==levels[i-1]:
                prevfile_url = entries[i-1]
                prevfile = readinit(prevfile_url)
                prevfile_url = prevfile_url.replace("_init.dat", prevfile["doc"])
                prevfile_url=os.path.relpath(prevfile_url, entry).replace(".md",".html")[1:]
                            
            j = 0
            while i-j>=0 and levels[i-j]==levels[i]:
                j+=1
            parentfile_url = entries[i-j]
            parentfile = readinit(parentfile_url)
            parentfile_url = parentfile_url.replace("_init.dat", parentfile["doc"])
            parentfile_url =os.path.relpath(parentfile_url, entry).replace(".md",".html")[1:]
        
        if i!=len(entries)-1:
            if levels[i]==levels[i+1]:
                nextfile_url = entries[i+1]
                nextfile = readinit(nextfile_url)
                nextfile_url = nextfile_url.replace("_init.dat", nextfile["doc"])
                nextfile_url=os.path.relpath(nextfile_url, entry).replace(".md",".html")[1:]

        do_nav = True

    # Attempt to locate source and dest files
    if initdata["source"]!="NONE":
        try:
            f_source = open(entryfol+initdata["source"], "r", encoding="utf8")
            do_file = True
        except:
            flog.write("unable to find source file %s in entry %s \n" %(entryfol+initdata["source"], entry))
        

    # Generate nav tree
    if initdata["tree"]!="FALSE":
        tree, treefols, tree_levels = scanfol(entryfol, tree_depth, tree_depth)
        do_tree = True

    # Attempt to load header and footer
    if initdata["header"]!="FALSE":
        # Check if a specific file is being nominated
        if initdata["header"] != "DEFAULT":
            try:
                f_head = open(entryfol+initdata["header"],'r', encoding="utf8")
                do_header = True
            except:
                flog.write("unable to find header file %s in entry %s \n" %(entryfol+initdata["header"], entry))        
        
        # Otherwise, check if there is a _header.md file
        if do_header==False and os.path.isfile(entryfol+"_header.md"):
            f_head = open(entryfol+"_header.md",'r', encoding="utf8")
            do_header = True
        
        # Otherwise, use the default header
        elif do_header==False:
            f_head = open(default_header,'r', encoding="utf8")
            do_header = True

    if initdata["footer"]!="FALSE":
        # Check if a specific file is being nominated
        if initdata["footer"] != "DEFAULT":
            try:
                f_foot = open(entryfol+initdata["footer"],'r', encoding="utf8")
                do_footer= True                
            except:
                flog.write("unable to find footer file %s in entry %s \n" %(entry, entryfol+initdata["footer"]))
        
        # Otherwise, check if there is a _header.md file
        if do_footer==False and os.path.isfile(entryfol+"_footer.md"):
            f_foot = open(entryfol+"_footer.md",'r', encoding="utf8")
            do_footer = True
        
        # Otherwise, use the default header
        elif do_footer==False:
            f_foot = open(default_footer,'r', encoding="utf8")
            do_footer = True
            
    #------------------------------------
    # FILE WRITE

    # write nav
    if do_nav:
        if prevfile !=None:
            fout.write("Previous Entry: [%s](%s)" %(prevfile["title"], prevfile_url))
            fout.write("\t&nbsp;\t")
            fout.write(tab)
        if nextfile != None:
            fout.write("Next Entry: [%s](%s)" %(nextfile["title"], nextfile_url))
        fout.write("  \n")
        fout.write("  \n")

        if parentfile != None:
            fout.write("  \n")
            fout.write("Go Back: [%s](%s)" %(parentfile["title"], parentfile_url))
            fout.write("\t&nbsp;\t")

            fout.write("Return to [Blog Home](%s)" %os.path.relpath(destfile.replace(".md",".html"), entry)[1:])
            fout.write("  \n")
            fout.write("  \n")
    
    # write header
    if do_header:
        for line in f_head:
            for replacement in replacements:
                line = line.replace(replacement[0],replacement[1])
            fout.write(line.replace("\n","  \n"))
            fout.write("  \n")

    # write tree
    if do_tree:
        # Scan entry's tree
        tree_entries, tree_folders, tree_levels = scanfol(entryfol, tree_depth)

        if len(tree_entries)>2:
            
            fout.write("**Navigation**")
            fout.write("  ")
            fout.write("\n")
            # Write these to the blog index .md file
            for tree_entry, tree_level in zip(tree_entries[1:], tree_levels[1:]):

                print("\t",tree_entry,tree_level)

                treeinit = readinit(tree_entry)
                tree_title  = treeinit['title']
                tree_relurl = os.path.relpath(tree_entry, entry).replace("_init.dat",treeinit['doc'])
                tree_relurl = tree_relurl.replace(".md",".html")[1:]

                print("\t\t",tree_title, tree_relurl)

                # Take data from the _init.md and write as a markdown line / url
                fout.write("\t"*(tree_level-1)) # Readable indentation
                fout.write("  "*(tree_level-1)) # markdown indentation
                fout.write("* ")
                fout.write("[%s](%s)" %(tree_title, tree_relurl))

                # Do markdown & html friendly line breaks
                fout.write("  ")
                fout.write("\n")
                
        fout.write("  \n")                
        fout.write("---------")
        fout.write("  \n")

        fout.write("  \n")
        fout.write("  \n")
                
            
    # write doc
    if do_file:

        lines = [line for line in f_source]
        dont_redact = [True]*len(lines)

        # Replace keywords
        for i in range(len(lines)):
            lines[i] = lines[i].replace(replacement[0],replacement[1])

        # Find redacted python script sections
        mode = True
        for i in range(len(lines)):
            if mode == True:
                if "```python" in lines[i] and "# REDACT" in lines[i+1]:
                    mode = False
            elif mode == False:
                if "```" in lines[i-1] and "```python" not in lines[i-1]:
                    mode = True
            dont_redact[i] = mode
            

        for i in range(len(lines)):
            if dont_redact[i]: fout.write(lines[i].replace("\n","  \n"))
        fout.write("  \n")

    # write footer
    if do_footer:
        fout.write("  \n")
        for line in f_foot:
            for replacement in replacements:
                line = line.replace(replacement[0],replacement[1])
            fout.write(line.replace("\n","  \n"))
            fout.write("  \n")
        fout.write("  \n")

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
