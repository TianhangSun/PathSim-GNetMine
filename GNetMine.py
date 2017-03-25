import numpy as np
from PathSim import find_max
import scipy.sparse as sp
import time

# the main algorithm;
# S: PA, AP, PC, CP, PT, TP, AC, CA
# ij: A, P, C, T; 
def GNetMine(S, ij, label, t = 100, k = 4, a = 0.1, l = 0.2):
    # label the items
    y = {n:[np.zeros((ij[n],1)) for _ in range(k)] for n in ij}
    for x in label:
        for i in label[x]:
            y[x][label[x][i]-1][i-1,0] = 1
            
    others = {i:[i+j for j in ij if j != i and i+j in S] for i in ij}
    
    # iterate until converge
    old_f = y
    for _ in range(t):
        new_f = {n:[np.zeros((ij[n],1)) for _ in range(k)] for n in ij}
        for fi in old_f:
            for j,fk in enumerate(old_f[fi]):
                sf = sum([S[i] * old_f[i[1]][j] for i in others[fi]])
                new_f[fi][j] = (l * sf + 2 * l * fk + a * y[fi][j]) \
                    / (len(others[fi]) * l + 2 * l + a)
        old_f = new_f
        # print new_f
    
    # decide the labels
    f = {i:[j.tolist() for j in old_f[i]] for i in old_f}
    out = {n:np.zeros(ij[n]) for n in ij}
    for fi in f:
        for i in range(len(f[fi][0])):
            out[fi][i] = np.argmax([f[fi][j][i] for j in range(k)]) + 1
            
    # print [out[i].tolist() for i in out]
    return out

# calculate S using files
def S(filenames, m, t = [0]):
    matrices = []
    for i, filename in enumerate(filenames):
        matrix = np.zeros((m[i], m[i+1]), dtype = np.int)
        with open(filename, "r") as f:
            for line in f:
                entry = [int(j) for j in line.split()]
                matrix[entry[0+t[i]]-1][entry[1-t[i]]-1] = entry[2]
        matrices.append(sp.csc_matrix(matrix))
    
    R = matrices[0]        
    for i in matrices[1:]:
        R *= i
        
    # generate diagnol matrix
    D1 = np.diag(np.array(np.power(R.sum(axis = 1), -0.5)).flatten())
    D2 = np.diag(np.array(np.power(R.sum(axis = 0), -0.5)).flatten())
    
    out = sp.csc_matrix(D1) * R * sp.csc_matrix(D2)
    return out, sp.csc_matrix.transpose(out)

# get existing labels
def get_label():
    author, paper = {}, {}
    with open("./data/author_label.txt", "r") as f1:
        for line in f1:
            a, l = [int(i) for i in line.split()]
            author[a] = l
    with open("./data/paper_label.txt", "r") as f2:
        for line in f2:
            p, l = [int(i) for i in line.split()]
            paper[p] = l
    
    def get_num(filename):
        out = []
        with open(filename, "r") as f:
            for line in f:
                out.append(int(line))
        return out
                
    train_author = get_num("./data/trainId_author.txt")
    train_paper = get_num("./data/trainId_paper.txt")
    test_author = get_num("./data/testId_author.txt")
    test_paper = get_num("./data/testId_paper.txt")
    
    return {"A":{a:author[a] for a in train_author},\
        "P":{p:paper[p] for p in train_paper}},\
        {"A":{a:author[a] for a in test_author},\
        "P":{p:paper[p] for p in test_paper}}

# generate results
def get_accuracy(result, label):
    label["C"] = {}
    with open("./data/conf_label.txt", "r") as f:
        for line in f:
            l = [int(i) for i in line.split()]
            label["C"][l[0]] = l[1]
            
    out = {"A":[], "P":[], "C":[]}
    for i in label:
        for j in label[i]:
            if result[i][j-1] == label[i][j]:
                out[i].append(1.0)
            else:
                out[i].append(0.0)
    print "\nAccuracy:\n", "author:",sum(out["A"])/len(out["A"]),\
        "\tpaper:",sum(out["P"])/len(out["P"]),\
        "\tconf:",sum(out["C"])/len(out["C"]),"\n"

def main():
    A, P, C, T = find_max()
    start = time.time()
    PA, AP = S(["./data/PA.txt"], [P, A])
    PC, CP = S(["./data/PC.txt"], [P, C])
    PT, TP = S(["./data/PT.txt"], [P, T])
    AC, CA = S(["./data/PA.txt", "./data/PC.txt"], [A, P, C], [1, 0])
    mid = time.time()
    
    train_label, test_label = get_label()
    r = GNetMine({"PA":PA, "AP":AP, "PC":PC, "CP":CP, "PT":PT, "TP":TP,\
        "AC":AC, "CA":CA}, {"A":A, "P":P, "C":C, "T":T}, train_label, 10)
    get_accuracy(r, test_label)
    end = time.time()
    
    print "Time:\ngenerate S:", mid - start, "\tGNetMine:", end - mid

if __name__ == "__main__":
    main()