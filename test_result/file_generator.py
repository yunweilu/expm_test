import os
tol = ["2","5","8","12","16"]
ts = ["1", "2","3","4","5","6","7","8","9","10"]

for i in range(len(tol)):
    for j in range(len(ts)):
        path = './state_propagator'
        file1 = "yunwei_"+tol[i]+"_"+"t"+ts[j]+".py"
        file = "sci_" + tol[i] + "_" + "t" + ts[j]+".py"
        dir_list = os.listdir(path)
        with open(os.path.join(path, file), 'w') as fp:
            pass
        with open(os.path.join(path, file1), 'w') as fp:
            pass
