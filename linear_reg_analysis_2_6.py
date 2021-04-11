import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pprint import pprint
from csv import reader
from csv import writer
import numpy as np
import random

#from analyze_2_6 import R
#from analyze_2_6 import analysis_dat
#from analyze_2_6 import cooc_dat
#from analyze_2_6 import succ_dat
#from commutative import commutative_results
#from decomposition import decomp_results

''' print responses with overall co-occurence > 100 and specified relation = 1'''
def top_resps(df, relation):
    return df.loc[df[relation] == 1,:].loc[df["Overall"] > 100,:]

'''print random 10 responses with a relation with value (0 or 1)'''
def rand_ten(df):
    add_val_r = df.sample(n=10)
    indices = add_val_r.index.value

    return indices

# statsmodels.formula.api ols() results
def results(file_name, df):
    with open(file_name, "w") as f:
        print("MULTIPLE LINEAR REGRESSION RESULTS (All)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall + C(Commutativity) + C(Add_Decomp)", data=df).fit()
        print(comm_sm.summary(), file=f)
        #print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only add_decomp)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall + C(Add_Decomp)", data=df).fit()
        print(comm_sm.summary(), file=f)
        #print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only Add_decomp + Interaction)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall * C(Add_Decomp)", data=df).fit()
        print(comm_sm.summary(), file=f)
        #print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only Comm)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall + C(Commutativity)", data=df).fit()
        print(comm_sm.summary(), file=f)
        #print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only Comm + Interaction)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall * C(Commutativity)", data=df).fit()
        print(comm_sm.summary(), file=f)
        #print(comm_sm.rsquared)
        print("\nSIMPLE LINEAR REGRESSION RESULTS", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall", data=df).fit()
        print(comm_sm.summary(), file=f)
        #print(comm_sm.rsquared)
        if file_name == "All_results.txt":
            # return predictions using one feature
            return comm_sm.predict()

'''calculate means for overall, successive co-occurrence, and predicted s.c. for response pairs split on specified relation'''
def means(df, value):
    means = dict(overall_mean = df["Overall"].mean(), 
                 successive_mean = df["Successive"].mean(),
                 predicted_mean = df["Predictions"].mean())
    return means

''' run analysis on specified item (A, B, C, D, E, or All)'''
def print_results(file, df):
    predictions = results(file, df)
    return predictions

# graph cooc_all and cooc_succ
def visualize(values):
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(3)

    all_zeros = values[0]["Overall"].to_numpy()
    succ_zeros = values[0]["Successive"].to_numpy()

    all_ones = values[1]["Overall"].to_numpy()
    succ_ones = values[1]["Successive"].to_numpy()

    plt.scatter(all_zeros, succ_zeros, s=6, c='navy', marker='o')
    plt.scatter(all_ones, succ_ones, s=6, c='dodgerblue', marker='o')

    reg_ones = np.poly1d(np.polyfit(all_ones, succ_ones, 1))
    reg_zeros = np.poly1d(np.polyfit(all_zeros, succ_zeros, 1))

    plt.plot(all_zeros, reg_zeros(all_zeros), "r--", color='royalblue')
    plt.plot(all_ones, reg_ones(all_ones),"r--", color='dodgerblue')

    plt.title("Simple Linear Regression")
    plt.xlabel("Frequency of overall co-occurrence")
    plt.ylabel("Frequency of successive cooccurrence")
    plt.show()

# 'comm_dat': dataframe that shows if a relationship between two responses is commutative
# 'add_d_dat': dataframe that shows if a relationship between two responses is additively decomposed
# 'mult_d_dat': dataframe that shows if a relationship between two responses is multiplicatively decomposed

# comm_dat = commutative_results(R)
# add_d_dat = decomp_results(R, "+", "add")

# analysis_dat(cooc_dat, succ_dat, comm_dat, add_d_dat, "Commutativity", "Add_Decomp")
# analysis_dat(add_d_dat, "Additive Decomposition")
# analysis_dat(mult_d_dat, "Multiplicative Decomposition")

''' df: dataframe of all the data necessary for analysis
    columns: Overall, Commutativity, Add_Decomp, Successive, Items
    index: string with value "resp1, resp2 '''
df = pd.read_csv("regression_dat.csv", header=0, index_col=0)

# set types 
df = df.astype({'Overall':str, 'Commutativity':int, 'Add_Decomp':int, 'Successive':int, 'Item':str})

#create a dictionary of dataframes, split on Item (A, B, C, D, or E)
item_spec = dict(tuple(df.groupby("Item")))

# example use of print_results
predictions = print_results("All_results.txt", df)
print_results("Item_A_results.txt", item_spec["A"])

# add predictions of successive coocurrence frequency based off of overall coocurrence frequency
df["Predictions"] = predictions

# add residuals of predictions 
#df["Residuals"] = df["Successive"] - df["Predictions"]

# create a dictionary of dataframes, split on relations(0 or 1)
comm_val = dict(tuple(df.groupby("Commutativity"))) 
add_val = dict(tuple(df.groupby("Add_Decomp")))

# example usage of rand_ten and top_resps
print(rand_ten(comm_val[1]))
print(top_resps(df, "Commutativity"))

# example use of means
print(means(comm_val[1]))

visualize(comm_val)
visualize(add_val)

