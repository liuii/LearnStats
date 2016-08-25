import chapter01.nsfg as nsfg
df = nsfg.ReadFemPreg()
pregordr = df[df.columns[1]]
print(pregordr)