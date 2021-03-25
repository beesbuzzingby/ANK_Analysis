import numpy as np
from numpy.core.defchararray import equal, isdigit
from numpy.lib.function_base import append 
import pandas as pd

#from analyze_2_6 import item_check

# r is only FIRST response
def get_operands(r, operator):
    # op_list: list of operands for either "*"" or "+""
    op_list = []

    for i in operator:
        if isdigit(i):
            op_list.append(i)
    
    return op_list

def is_decomp(r1, r2, operator):
    # if the two responses aren't in the same item, can't be decomposed
    #if item_check(r1, r2) == "N":
        #return False

    # must have operator in the potentially "decomposed" response
    if operator not in r2:
        return False

    decomp = "Maybe"
    decomp_len = 0

    # search for decomposition from r1 -> r2 (as in: R1 = 2 + 2, R2 = 1 + 1 + 2)
    # therefore, r2 cannot be shorter than r1
    if len(r2) < len(r1):
        return False

    for i in range(len(r1)):
        sub_exp = ""
        # start at i's position in j
        # if decomp found this will change
        for j in range(i, len(r2)):
            # potential for decomposition
            if (r2[j] != r1[i] and j == i) and ((j + 1 < len(r2) and r2[j + 1] == operator) or (j - 1 < len(r2) and r2[j - 1] == operator and r2[j + 1] != operator)):
                k = j
                while k < len(r2):
                    sub_exp += r2[k]
                    print(sub_exp)
                    
                    # do the stuffs
                    if isdigit(sub_exp[-1]):
                        if pd.eval(sub_exp) == int(r1[i]):
                            decomp_len += len(sub_exp)
                            decomp = "True"
                            break
                        elif (k == len(r2) - 1 or (isdigit(r2[k]) and r2[k + 1] != "+")) and decomp == "Maybe":
                            decomp = "False"
                            j += k - j
                            break
                        else:
                            decomp = "Maybe"     
                    k += 1
            if decomp == "True":
                break

            # These expressions don't match up, can't be decomposed
            if decomp == "False" and ((j == i and r2[j] != r1[i]) or (j == i + len(sub_exp))):
                return False
            # These expressions have at least one case of decomposition
            # but we need to make sure there are no other differences
            elif decomp == True and j == i + len(sub_exp):
                if r2[j] != r1[i]:
                    # the expressions have another difference
                    return False

    # The expressions only differ by one or more cases of decomposition, all good
    return True

def decomp_results(R, operator, op_name):
    d_d = pd.DataFrame(data=np.zeros((len(R), len(R)), dtype=int), index=R, columns=R)

    i = 0
    ignore = []
    
    for r1 in R:
        for r2 in R:
            print(i)
            if [r1, r2] in ignore:
                    continue
            if is_decomp(r1, r2, operator):
                # record that these responses are commutative and don't rerun with them reversed
                d_d.loc[r1, r2] = 1
                d_d.loc[r2, r1] = 1
                ignore.append([r2, r1])
            i += 1

    # save commutative relation data to excel file
    d_d.to_excel("ANK_" + op_name + "_decomp_relations.xlsx")
    
    return d_d

print(is_decomp("2+2", "1+1+2", "+"))