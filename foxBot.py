import pickle
import numpy as np
import pandas as pd
from util import clean

with open("model.bin", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.bin", "rb") as f:
    vectorizer = pickle.load(f)

while(True):
    req = input("Olá eu sou o Fox Bot o que você quer?\n")
    counts = vectorizer.transform([clean(req)])
    prob = model.predict_proba(counts)[0]
    max_prob = np.amax(prob)

    if max_prob < 0.5:
        print("Não entendi poderia reformular\n")
        continue
    
    res = model.classes_[np.where(prob == max_prob)[0]][0]
    certo = input("Eu entendi: {} \nEsta certo? ([S]im/Não) ".format(res)).lower()
    if certo == "s" or certo == "sim" or certo == "":
        pass
    elif certo == "não" or certo == "nao" or certo == "n":
        for idx, i in enumerate(model.classes_):
            print(f"{idx} - {i}")
        print(f"{idx+1} - Nenhuma das anteriores")
        correto = int(input("Qual o certo então? "))
        
        if correto == idx+1:
            continue
        else:
            res = model.classes_[correto]
    else:
        print("Não entendi \n")
        continue
    
    with open("sentencas.csv", "a") as f:
        f.write(f'\n,"{req}","{res}"')
    
    model.partial_fit(counts, [res])
    
    with open("model.bin","wb+") as f:
        pickle.dump(model, f)
