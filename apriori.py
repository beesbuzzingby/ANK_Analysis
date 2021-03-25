from apyori import apriori

import analyze_1_24

items = analyze_1_24.items

def write_results(final_results, conf, sup, lift, filename):
    with open(filename, "a") as f:
        print("MINIMUM VALUES: Confidence = ", conf, " Support = ", sup, " Lift = ", lift, file=f)
        print("========================", file=f)
        if(filename == 'Apriori_per_item.txt'):
            print("ITEM ", item, " RESULTS", file=f)
            print("========================", file=f)

        for result in final_results:    
            print("Rule:", end=" ", file=f)

            for x in result[0]:
                print(x, end=" ", file=f)

            print("\nConfidence: ", "{:.3f}".format(result[2][0][2]), file=f)
            print("Support: ", "{:.3f}".format(result[1]), file=f)
            print("Lift: ", "{:.3f}".format(result[2][0][3]), file=f)

            print("========================", file=f)
        print("\n", file=f)

'''Use apriori algorithm to find frequent sets of responses (including frequent 1-itemsets)
    The parameters (minimum support, minimum confidence, and minimum lift) can be adjusted below '''
# suppport(A => B) = P(A ∩ B)
min_sup = 0.4 
# confidence(A => B) = P(B | A)
min_conf = 0.7 
# lift(A => B) = P(B | A) / P(B)
min_lift = 1

# Find frequent itemsets for responses in each item A, B, C, D, and E
for item in items:
    final_rule = apriori(items[item]['resp_lists'], min_support=min_sup, min_confidence=min_conf, min_lift=min_lift)
    final_results=list(final_rule)
    final_results.sort(reverse=True, key=lambda x:x[2][0][2])
    write_results(final_results, min_conf, min_sup, min_lift, 'Apriori_per_item.txt')

# 'all_resps': all resps for each participant regardless of item
all_resps = []

for i in range(1072):
    all_resps.append([])

for item in items:
    counter = 0
    for resps in items[item]['resp_lists']:
        all_resps[counter].extend(resps)
        counter += 1

# Find frequent itemsets for all responses across items
final_rule = apriori(all_resps, min_support=min_sup, min_confidence=min_conf, min_lift=min_lift)
final_results=list(final_rule)
final_results.sort(reverse=True, key=lambda x:x[2][0][2])
write_results(final_results, min_conf, min_sup, min_lift, 'Apriori_across_items.txt')