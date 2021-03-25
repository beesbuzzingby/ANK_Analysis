import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pprint import pprint
from csv import reader
from csv import writer
import numpy
import random

#from analyze_2_6 import R
#from analyze_2_6 import analysis_dat
#from analyze_2_6 import cooc_dat
#from analyze_2_6 import succ_dat
#from commutative import commutative_results
#from decomposition import decomp_results

# add later features without having to rerun the entire code
# feature will be dataframe, need to fig that out 
def add_dat(name, feature, R):
    i = 0
    new_feat = []
    for r1 in R:
        for r2 in R:
            new_feat.append(feature.loc[r1, r2])

    with open(name, 'r') as read_obj, open(name, 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)

        for row in csv_reader:
            if row[0] == "O":
                row.append("Add_Decomp")

            row.append(new_feat[i])
            csv_writer.writerow(row)
            i += 1


# statsmodels.formula.api ols() results
def results(file_name, df):
    with open(file_name, "w") as f:
        print("MULTIPLE LINEAR REGRESSION RESULTS", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall + C(Commutativity) + C(Add_Decomp)", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (WITH INTERACTION)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall * C(Add_Decomp) + C(Commutativity)", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only add_decomp)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall + C(Add_Decomp)", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only Add_decomp + Interaction)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall * C(Add_Decomp)", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only Comm)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall + C(Commutativity)", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        print("\nMULTIPLE LINEAR REGRESSION RESULTS (Only Comm + Interaction)", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall * C(Commutativity)", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        print("\nSIMPLE LINEAR REGRESSION RESULTS", file=f)
        comm_sm = smf.ols(formula="Successive ~ Overall", data=df).fit()
        print(comm_sm.summary(), file=f)
        print(comm_sm.rsquared)
        if file_name == "All_results.txt":
            # return predictions using one feature
            return comm_sm.predict()

# 'comm_dat': dataframe that shows if a relationship between two responses is commutative
# 'add_d_dat': dataframe that shows if a relationship between two responses is additively decomposed
# 'mult_d_dat': dataframe that shows if a relationship between two responses is multiplicatively decomposed
# comm_dat = commutative_results(R)
#comm_dat = pd.read_excel("ANK_commutative_relations.xlsx", index_col=0)
#add_d_dat = decomp_results(R, "+", "add")
#mult_d_dat = decomp_results(R, "*", "mult")
#add_d_dat = pd.read_excel("ANK_add_decomp_relations.xlsx", index_col=0)
#add_dat("mult_decomp", mult_d_dat, R)

#analysis_dat(cooc_dat, succ_dat, comm_dat, add_d_dat, "Commutativity", "Add_Decomp")
# analysis_dat(add_d_dat, "Additive Decomposition")
#analysis_dat(mult_d_dat, "Multiplicative Decomposition")

# "df": dataframe of all the data necessary for analysis
# columns are "Overall", "Comm", "Successive", "Predictions", "Residuals", "Add_Decomp", "Mult_decomp"
# df = pd.read_csv("regression_dat.csv", header=0, index_col=0)

df = pd.read_csv("regression_dat.csv", index_col=0)

# run analysis on all items
predictions = results("All_results.txt", df)

#create a dictionary of dataframes, split on Item (A, B, C, D, or E)
#item_spec = dict(tuple(df.groupby("Item")))

# run analysis for each individual item
#results("Item_A_results.txt", item_spec["A"])
#results("Item_B_results.txt", item_spec["B"])
#results("Item_C_results.txt", item_spec["C"])
#results("Item_D_results.txt", item_spec["D"])
#results("Item_E_results.txt", item_spec["E"])

# add predictions of successive coocurrence frequency based off of overall coocurrence frequency
#df["Predictions"] = predictions

# add residuals of predictions 
#df["Residuals"] = df["Successive"] - df["Predictions"]

# create a dictionary of dataframes, split on Commutativity (0 or 1)
#comm_val = dict(tuple(df.groupby("Commutativity"))) 

# calculate means for overall and successive co-occurrence for commutative and non-commutative response pairs
#nc_means = dict(overall_mean = comm_val[0]["Overall"].mean(), successive_mean = comm_val[0]["Successive"].mean())
#c_means = dict(overall_mean = comm_val[1]["Overall"].mean(), successive_mean = comm_val[1]["Successive"].mean())

# calculate predicted mean and residual mean for commutative and non-commutative response pairs
#nc_pr_means = dict(predict_mean = comm_val[0]["Predictions"].mean(), res_mean = comm_val[0]["Residuals"].mean())
#c_pr_means = dict(predict_mean = comm_val[1]["Predictions"].mean(), res_mean = comm_val[1]["Residuals"].mean())

f = plt.figure()
f.set_figwidth(6)
f.set_figheight(3)

# graph cooc_aland cooc_succ
x = df["Overall"].to_numpy() # cooc_all
y = df["Successive"].to_numpy() # cooc_succ

plt.scatter(x, y, s=6, marker='o') # scatterplot
plt.plot(x, 0.3905 * x - 0.0205, color='navy') # regression line
plt.title("Simple Linear Regression")
plt.xlabel("Frequency of overall co-occurrence")
plt.ylabel("Frequency of successive cooccurrence")
plt.show()

