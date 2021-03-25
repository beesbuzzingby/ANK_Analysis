import numpy as np, pandas as pd
from numpy.core.defchararray import add
import re # for regular expressions
from collections import Counter

# load raw data
raw_dat         = pd.read_excel("ANK_raw_data.xlsx", index_col=0)

# create a dict items containing source numbers and target number for each item
# added type (dense/sparse)
items = {
    'A': {'opts': [1,2,3,4],       'targ': 6,   'type': 'dense'},
    'B': {'opts': [2,4,8,12,32],   'targ': 16,  'type': 'dense'},
    'C': {'opts': [1,2,3,5,30],    'targ': 59,  'type': 'sparse'},
    'D': {'opts': [2,4,6,16,24],   'targ': 12,  'type': 'dense'},
    'E': {'opts': [3,5,30,120,180],'targ': 12,  'type': 'sparse'}} 

def validate(r, item):
    """ Check whether r is a valid response for item"""
    I = items[item]
    numbers = [int(n) for n in re.findall('(\d+)', r)]
    if len([n for n in numbers if n not in I['opts']])>0:
        return(False)
    else:
        try:
            return eval(r)==I['targ']
        except (SyntaxError, ZeroDivisionError):
            return(False)

# remove parentheses from a response if they don't affect the result
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

# check if two responses have the same target value (are in the same item)
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
            # options are mutually exclusive, so just need to check first number of each
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

# prepare data for analysis with given feature
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
            
# for each item in items, add
# 'resp_lists', a list containing each participant's answers in list format
# 'valid_resp_counts', a list of valid individual answers and their frequencies
# 'valid_resps', a list of valid individual answers without frequencies
for item in raw_dat.columns:
    v = list(raw_dat.loc[:,item])
    w = [([] if pd.isnull(resp) else re.findall("\[(.*?)\]", resp)) for resp in v]
    items[item]['resp_lists'] = w
    x = [r for r in [r for resps in w for r in resps] if validate(r, item)]

    items[item]['valid_resp_counts'] = Counter(x).most_common()
    items[item]['valid_resps'] = [resp for (resp, count) in items[item]['valid_resp_counts']]

# create two co-occurrence matrices, each with one row and column per valid response
# 'cooc_dat': number of co-occurrences for each valid response pair
# 'succ_dat': number of co-occurrences in immediate succession
L = [items[item]['valid_resps'] for item in items.keys()]
R_0 = [r for resps in L for r in resps]
R_1 = [check_p(resp) for resp in R_0 if "+-" not in resp]

# used to create analysis dataframe of unique responses
R = []
for resp in R_1:
    if resp not in R:
        R.append(resp)

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
#cooc_dat.to_excel("ANK_cooccurrence_all.xlsx")
#succ_dat.to_excel("ANK_cooccurrence_successive.xlsx")