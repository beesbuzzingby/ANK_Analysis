import numpy as np, pandas as pd
from numpy.core.defchararray import isdigit
import re # for regular expressions
from collections import Counter

# load raw data
raw_dat = pd.read_excel("ANK_raw_data.xlsx", index_col=0)

# create a dict items containing source numbers and target number for each item
items = {
    'A': {'opts': [1,2,3,4],       'targ': 6,   'type': 'dense'},
    'B': {'opts': [2,4,8,12,32],   'targ': 16,  'type': 'dense'},
    'C': {'opts': [1,2,3,5,30],    'targ': 59,  'type': 'sparse'},
    'D': {'opts': [2,4,6,16,24],   'targ': 12,  'type': 'dense'},
    'E': {'opts': [3,5,30,120,180],'targ': 12,  'type': 'sparse'}} 

''' Check whether r is a valid response for item '''
def validate(r, item):
    I = items[item]
    numbers = [int(n) for n in re.findall('(\d+)', r)]
    if len([n for n in numbers if n not in I['opts']])>0:
        return(False)
    else:
        try:
            return eval(r)==I['targ']
        except (SyntaxError, ZeroDivisionError):
            return(False)

''' Remove parentheses from a response if they don't affect the result '''
def check_p(r):
    if "(" not in r and ")" not in r:
        return r 

    r_no_paren = re.sub(r"[\(\)]", "", r)
    
    # parentheses do not affect the result, remove them
    if pd.eval(r) == pd.eval(r_no_paren):
        return r_no_paren
    # parentheses do affect the result, don't remove them
    else:
        return r

''' Check if two responses have the same target value (are in the same item) '''
def item_check(r1, r2):
    targ_1, targ_2 = pd.eval(r1), pd.eval(r2)

    # the two responses are part of the same item
    if targ_1 == targ_2:
        if targ_1 == 6:
            return "A"
        elif targ_1 == 16:
            return "B"
        elif targ_1 == 59:
            return "C"
        elif targ_1 == 12:
            # D and E have equal targets, so need to differentiate
            num1, num2 = 0, 0

            # get first number that appears in each response
            for char in r1:
                if char.isnumeric():
                    num1 = int(char)
                    break 
            for char in r2:
                if char.isnumeric():
                    num2 = int(char)
                    break

            # both belong to D or both belong to E
            if num1 in items["D"]["opts"] and num2 in items["D"]["opts"]:
                return "D"
            elif num1 in items["E"]["opts"] and num2 in items["E"]["opts"]:
                return "E"
            # one belongs to each
            else:
                return "N"
        
    # the two responses aren't part of the same item
    return "N"

''' R: a list of all valid responses '''
L = [items[item]['valid_resps'] for item in items.keys()]
R_0 = [r for resps in L for r in resps]
R_1 = [check_p(resp) for resp in R_0 if "+-" not in resp]

R = []
for resp in R_1:
    if resp not in R:
        R.append(resp)

''' Prepare data for analysis with given features, store in "regression_dat.csv"
    cooc_dat: dataframe of overall coocurrence
    succ_dat: datafram of successive coocurrence
    feature, feature2: features to add (currently using commutativity, additive decomposition)
    f_name, f_name2: feature names to be used in csv'''
def analysis_dat(cooc_dat, succ_dat, feature, feature2, f_name, f_name2):
    # dat: overall coocurrence, commutativity, successive cooccurrence, item
    # rows: each response pair
    dat = []
    rows = []

    for r1 in R:
        for r2 in R: 
            # only add two responses in the same item (have the same target)
            item = item_check(r1, r2)
            if item == "N":
                continue
            else:
                dat.append([cooc_dat.loc[r1, r2], feature.loc[r1, r2], \
                    feature2.loc[r1, r2], succ_dat.loc[r1, r2], item])
                rows.append(str(r1) + ", " + str(r2))

    # 'df': contains relevant data for analysis, as well as predictions and residuals
    df = pd.DataFrame(dat, columns = ['Overall', f_name, f_name2, 'Successive', 'Item'], index=rows)
    df.to_csv("regression_dat.csv")
            
''' For each item in items, add
    'resp_lists', a list containing each participant's answers in list format
    'valid_resp_counts', a list of valid individual answers and their frequencies
    'valid_resps', a list of valid individual answers without frequencies '''
def calc_cooccurrence():
    for item in raw_dat.columns:
        v = list(raw_dat.loc[:,item])
        w = [([] if pd.isnull(resp) else re.findall("\[(.*?)\]", resp)) for resp in v]
        items[item]['resp_lists'] = w
        x = [r for r in [r for resps in w for r in resps] if validate(r, item)]

        items[item]['valid_resp_counts'] = Counter(x).most_common()
        items[item]['valid_resps'] = [resp for (resp, count) in items[item]['valid_resp_counts']]

    ''' Create two co-occurrence matrices, each with one row and column per valid response
        'cooc_dat': number of co-occurrences for each valid response pair
        'succ_dat': number of co-occurrences in immediate succession '''
    cooc_dat = pd.DataFrame(data=np.zeros((len(R), len(R)), dtype=int), index=R, columns=R)
    succ_dat = pd.DataFrame(data=np.zeros((len(R), len(R)), dtype=int), index=R, columns=R)

    for item in items.keys():
        for resps in items[item]['resp_lists']:
            if len(resps)>1:
                for i in range(0, len(resps)-1):
                    for j in range(i+1, len(resps)):
                        if resps[i] in items[item]['valid_resps'] and resps[j] in items[item]['valid_resps']:
                            if "+-" not in resps[i] and "+-" not in resps[j]:
                                r1 = check_p(resps[i])
                                r2 = check_p(resps[j])

                                cooc_dat.loc[r1, r2] += 1
                                cooc_dat.loc[r2, r1] += 1
                                if j==i+1:
                                    succ_dat.loc[r1, r2] += 1
                                    succ_dat.loc[r2, r1] += 1

    # save the co-occurrence matrices
    cooc_dat.to_excel("ANK_cooccurrence_all.xlsx")
    succ_dat.to_excel("ANK_cooccurrence_successive.xlsx")

''' Determine if two items are commutative
    Commutativity definition: 
    '''
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

''' Create a dataframe that shows which response pairs have the relation "commutativity" '''
def commutative_results():
    c_d = pd.DataFrame(data=np.zeros((len(R), len(R)), dtype=int), index=R, columns=R)

    for r1 in R:
        for r2 in R:
            if is_commutative(r1, r2):
                # record that these responses are commutative
                c_d.loc[r1, r2] = 1
                c_d.loc[r2, r1] = 1

    # save commutative relation data to excel file
    c_d.to_excel("ANK_commutative_relations.xlsx")
    
    return c_d

''' Determine if two items are related by additive decomposition.  
    Additive decomposition definition: 
    Two responses, which can be broken up into valid sub-expressions (or "nodes"), and which differ by one or more nodes.  
    For one response, this node(s) contains a sub-expression of only a single number. 
    For the other response, this same node(s) must be a sub-expression involving only addition that is equal to the sub-expression in the other response's same node(s).  
    The order of these sub-expressions matters, and sub-expressions must be in the same location for both responses.'''
def is_decomp(r1, r2, operator):
    # if the two responses aren't in the same item, can't be decomposed
    if item_check(r1, r2) == "N":
        return False

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

''' Create a dataframe that shows which response pairs have the relation "additive decomposition" '''
def decomp_results(operator, op_name):
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