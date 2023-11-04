import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

lod = 'Length of Discussion'
cs = 'Current Score'
filename_discussion = './data/discussions.csv'
filename_grade = './data/gradebook.csv'
moderate_corr = 0.3
strong_corr = 0.5
is_corr = 0.05
# because total characters of a discussion are too many
hundred_thousand = 1_00_000

corr = 0
p_value = 0


# function to get the discussion dict from the file
def get_discussion_dict(discussion_dict) :
    # get data from the discussions.csv and gradebook.csv
    # only get learner's data
    with open(filename_discussion, 'r') as file :
        for row in csv.reader(file) :
            if row[1] == '["Learner"]' :
                discussion_dict[int(row[0][8:])] = [row[0], (row[7] + row[8]), row[9]]
    with open(filename_grade, 'r') as file :
        for row in csv.reader(file) :
            if row[0].startswith('LEARNER_') :
                if int(row[0][8:]) in discussion_dict.keys() :
                    discussion_dict[int(row[0][8:])].append(row[12])

    # sort the dict based on key
    discussion_dict = dict(sorted(discussion_dict.items()))

    # clear dictionary without grade
    cleared_dict = {}
    for d in discussion_dict.keys() :
        if len(discussion_dict[d]) == 4 :
            cleared_dict[d] = discussion_dict[d]

    return cleared_dict


# plot the scatter plot of length of discussion and grade
def scatter_plot_length_grade(discussion_dict) :
    x = []
    y = []
    for d in discussion_dict.values() :
        x.append(float(d[1])/hundred_thousand)
        y.append(float(d[3]))
    
    plt.title('Scatter Plot between Discussion and Grade')
    plt.xlabel(lod)
    plt.ylabel(cs)
    plt.scatter(x, y)
    plt.savefig('./graphs/scatter_plot_discussion_length_grade.png')


# make join plot pf length of discussion and grade
def join_plot_length_grade(discussion_dict) :
    x = []
    y = []
    for d in discussion_dict.values() :
        x.append(float(d[1])/hundred_thousand)
        y.append(float(d[3]))

    # make into pandas data frame
    data = {
        'Length of Discussion' : x,
        'Current Score' : y
    }

    df = pd.DataFrame(data)
    sns.jointplot(data=df, x=lod, y=cs, kind='reg', dropna=True)
    plt.savefig('./graphs/joint_plot_discussion_length_grade.png')


# Hypothesis testing, null hypothesis there is a correlation between x and y
# alternate hypothesis there is no correlation between x and y
# get the correlation between number of likes and length of discussion
def get_correlation(discussion_dict) :
    x = []
    y = []
    global corr
    global p_value
    for d in discussion_dict.values() :
        x.append(float(d[1])/hundred_thousand)
        y.append(float(d[3]))
    corr, p_value = pearsonr(x, y)


# running all the functions
def run() :
    global corr
    global p_value
    # dict structure
    # key : learner id
    # value : list of actor_id, total length of discussion (discussion and post message), count of likes, and current score
    discussion_dict = {}
    discussion_dict = get_discussion_dict(discussion_dict)
    get_correlation(discussion_dict)
    if p_value > is_corr :
        print("Correlation does exist with Pearson correlation coefficient of", corr)
    else :
        print("Correlation doesn't exist since ", p_value, " is less than threshold (0.05)")
    scatter_plot_length_grade(discussion_dict)
    join_plot_length_grade(discussion_dict)


# the program driver
def main() :
    run()


if __name__ == '__main__' :
    main()
