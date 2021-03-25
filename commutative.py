import numpy as np
from numpy.core.defchararray import equal 
import pandas as pd

# determine if two items are commutative
def is_commutative(r1, r2):
    div_1, div_2 = " ", " "
    
    # if the responses are exactly the same, they are not commutative
    if r1 == r2:
        return False

    # take care of issues with division expressions having equal results but not being commutative
    stop, sec = 0, 0
    if (")" not in r1 and ")" not in r2) and ("/" in r1 and "/" in r2):
        # check how many divison symbols there are and store the divisor and dividend
        for i in range(len(r1)):
            if i == 1 and r1[i] == "/":
                sec = 1
                break
            if r1[i] == "/":
                if stop == 1:
                    stop = 2
                    break

                div_1 = r1[i - 1]
                div_2 = r1[i + 1]

                stop = 1

        for i in range(len(r2)):
            # ignore resps with 2+ division symbols or resps with division as the second character
            if stop == 2 or sec == 1:
                break
            
            # if one of the divisors or dividends doesn't match the other, responses aren't commutative
            if r2[i] == "/":
                if div_1 != r2[i - 1]:
                    return False
                if div_2 != r2[i + 1]:
                    return False 

    # sort each response in ASCII order 
    rearr_r1 = ''.join(sorted(r1))
    rearr_r2 = ''.join(sorted(r2)) 

    # if these responses are equal, they have to be commutative (results are guaranteed to be equal)
    if rearr_r1 == rearr_r2:
        return True

    return False

# create a matrix show which response pairs have the relation "commutativity" 
def commutative_results(R):
    c_d = pd.DataFrame(data=np.zeros((len(R), len(R)), dtype=int), index=R, columns=R)

    for r1 in R:
        for r2 in R:
            if is_commutative(r1, r2):
                # record that these responses are commutative
                c_d.loc[r1, r2] = 1
                c_d.loc[r2, r1] = 1

    # save commutative relation data to excel file
    #c_d.to_excel("ANK_commutative_relations.xlsx")
    
    return c_d