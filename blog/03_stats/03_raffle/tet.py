f = open('raffle_working.md', 'r')
fhead = open('../../_components/_bloghead.md', 'r')
ffoot = open('../../_components/_blogfoot.md', 'r')

fout = open("./page.md",'w')

for line in fhead:
    fout.write(line)

for f in fhead:
    fout.write(line)

for ffoot in fhead:
    fout.write(line)

f.close()
fhead.close()
ffoot.close()
fout.close()

print("!")
