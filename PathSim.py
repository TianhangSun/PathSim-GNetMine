import numpy as np
import scipy.sparse as sp
import time

# function to construct a single matrix
def construct_matrix(filename, m1, m2):
    matrix = np.zeros((m1, m2), dtype = np.int)   
    with open(filename, "r") as f:
        for line in f:
            entry = [int(i) for i in line.split()]
            matrix[entry[0]-1][entry[1]-1] = 1
    out = sp.csc_matrix(matrix)
    return out, sp.csc_matrix.transpose(out)

# the algorithm
def PathSim(x, m, k, num):
    """ # this is significantly slower
    candidates = set()
    for i in matrix.getrow(x-1).indices:
        for j in matrix.getrow(i-1).indices:
            candidates.add(j)
    """     
    out = []
    
    for i in range(num):
        value = 2.0 * m[x-1,i] / (m[x-1,x-1] + m[i,i])
        out.append((value, i+1))
        
    out.sort()
    out.reverse()
    return out[:5]

# find how many of these items
def find_max():
    author,paper,venue,term = 0,0,0,0
    with open("./data/PA.txt", "r") as f1:
        for line in f1:
            entry = line.split()
            paper = max(paper, int(entry[0]))
            author = max(author, int(entry[1]))
            
    with open("./data/PC.txt", "r") as f2:
        for line in f2:
            entry = line.split()
            paper = max(paper, int(entry[0]))
            venue = max(venue, int(entry[1]))
            
    with open("./data/PT.txt", "r") as f3:
        for line in f3:
            entry = line.split()
            paper = max(paper, int(entry[0]))
            term = max(term, int(entry[1]))
            
    return author, paper, venue, term

def main():
    start = time.time()  
    m = find_max()
    PA, AP = construct_matrix("./data/PA.txt", m[1], m[0])
    PC, CP = construct_matrix("./data/PC.txt", m[1], m[2])
    PT, TP = construct_matrix("./data/PT.txt", m[1], m[3])
    
    APCPA = (AP * PC * CP * PA).todense()
    APTPA = (AP * PT * TP * PA).todense()   
    mid = time.time()
    
    print PathSim(7696, APCPA, 5, m[0])
    print PathSim(7696, APTPA, 5, m[0])   
    end = time.time()
    print "\nTime:\ngenerate Mp:", mid - start, "\tPathSim:", end - mid

if __name__ == "__main__":
    main()
    
    """ cnt = {} # check if answer make sense
    with open("./data/PA.txt") as f:
        for line in f:
            author = int(line.split()[1])
            if author not in cnt:
                cnt[author] = 1
            else:
                cnt[author] += 1
    print [(i,cnt[i]) for i in cnt if cnt[i] > 80]
    """