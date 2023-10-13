import glob
DATA_DIR = "./field_data/"
fieldPaths = glob.glob(DATA_DIR + "*.csv")

for fieldPath in fieldPaths:
    with open(fieldPath) as f:
        data = f.read()
        data = data.replace("a", "temp").replace("b", "a").replace("temp", "b")
        
    fileName = fieldPath.split("/")[-1]
    with open(DATA_DIR + "inv_" + fileName, mode='w') as f:
        f.write(data)