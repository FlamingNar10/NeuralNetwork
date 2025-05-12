import numpy as np

def f(x,y):
    return (x - 5)**2 + ((y-5)/2)**2 < 1 or (x-10)**2 + (y-7)**3 > 1

with open("train.csv","w+") as theFile:
    for i in range(1000):
        x = np.random.random()*15
        y = np.random.random()*15
        if (f(x,y)):
            theFile.write(f"1,{x:.2f},{y:.2f}\n")
        else:
            theFile.write(f"0,{x:.2f},{y:.2f}\n")
    