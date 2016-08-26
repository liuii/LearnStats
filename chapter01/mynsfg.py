for line in open("2002FemPreg.dct"):
    print(' '.join(line.strip().split()[4:]).lower().capitalize(), "$")
