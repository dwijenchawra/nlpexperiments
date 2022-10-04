import os

files = os.listdir("/home/x-dchawra/nlpexperiments/rf_svm_imdb/.rfjob/out")

for i in files:
    with open("/home/x-dchawra/nlpexperiments/rf_svm_imdb/.rfjob/out/" + i, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("Params:" + i.rstrip(".out").rstrip('\r\n') + ":" + content)


