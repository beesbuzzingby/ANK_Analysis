import pandas as pd, numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#from analyze_2_6 import analysis_dat
#from analyze_2_6 import commutative_results
#from analyze_2_6 import decomp_results

''' Run analysis on specified item (A, B, C, D, E, or All) 
    uses statsmodel.formula.api regression 
    file_name: file results are printed to
    df: dataframe of data from regression_dat.csv, or a subsection of that data split on Item '''
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

''' Calculate means for overall, successive co-occurrence, and predicted s.c. 
   for response pairs split on specified relation '''
def means(df):
    means = dict(overall_mean = df["Overall"].mean(), 
                 successive_mean = df["Successive"].mean(),
                 predicted_mean = df["Predictions"].mean())
    return means

''' Graph successive coocurrence based on overall coocurrence
    values: the relation to split the points (ex: comm, or add_decomp)
    navy/royalblue: pairs without specified relation and their regression line
    dodgerblue: pairs with specified relation and their regression line'''
def visualize(values):
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(3)

    all_zeros = values[0]["Overall"].to_numpy()
    succ_zeros = values[0]["Successive"].to_numpy()

    all_ones = values[1]["Overall"].to_numpy()
    succ_ones = values[1]["Successive"].to_numpy()

    # plot points without specified relation
    plt.scatter(all_zeros, succ_zeros, s=6, c='navy', marker='o')
    # plot points with specified relation
    plt.scatter(all_ones, succ_ones, s=6, c='dodgerblue', marker='o')

    reg_ones = np.poly1d(np.polyfit(all_ones, succ_ones, 1))
    reg_zeros = np.poly1d(np.polyfit(all_zeros, succ_zeros, 1))

    # plot regression line for points without specified relation
    plt.plot(all_zeros, reg_zeros(all_zeros), "r--", color='royalblue')
     # plot regression line for points with specified relation
    plt.plot(all_ones, reg_ones(all_ones),"r--", color='dodgerblue')

    plt.title("Simple Linear Regression")
    plt.xlabel("Frequency of overall co-occurrence")
    plt.ylabel("Frequency of successive cooccurrence")
    plt.show()


''' 'comm_dat': dataframe that shows if a relationship between two responses is commutative'''
# comm_dat = commutative_results(R)

''' 'add_d_dat': dataframe that shows if a relationship between two responses is additively decomposed '''
# add_d_dat = decomp_results(R, "+", "add")

# add data to dataframe and "regression_dat.csv" for analysis
#analysis_dat(cooc_dat, succ_dat, comm_dat, add_d_dat, "Commutativity", "Add_Decomp")

''' df: dataframe of all the data necessary for analysis
    columns: Overall, Commutativity, Add_Decomp, Successive, Items
    index: string with value "resp1, resp2 '''
df = pd.read_csv("regression_dat.csv", header=0, index_col=0)

# set types of each column 
df = df.astype({'Overall':str, 'Commutativity':int, 'Add_Decomp':int, 'Successive':int, 'Item':str})

''' item_spec: create a dictionary of dataframes, split on Item (A, B, C, D, or E) '''
item_spec = dict(tuple(df.groupby("Item")))

# run the analysis
predictions = results("All_results.txt", df)

# add predictions of successive coocurrence frequency based off of overall coocurrence frequency
#df["Predictions"] = predictions

# run the analysis for each item individually
# print_results("Item_A_results.txt", item_spec["A"])
# print_results("Item_A_results.txt", item_spec["B"])
# print_results("Item_A_results.txt", item_spec["C"])
# print_results("Item_A_results.txt", item_spec["D"])
# print_results("Item_A_results.txt", item_spec["E"])

# add residuals of predictions
#df["Residuals"] = df["Successive"] - df["Predictions"]

# create a dictionary of dataframes, split on relations(0 or 1)
comm_val = dict(tuple(df.groupby("Commutativity"))) 
add_val = dict(tuple(df.groupby("Add_Decomp")))

# example use of printing out the mean co-occurrence values 
print(means(comm_val[1]))

# show a graph of commutative pairs + regression line, and non-commutative pairs + regression line
visualize(comm_val)
# show a graph of add-decomp pairs + regression line, and non-add-decomp pairs + regression line
visualize(add_val)