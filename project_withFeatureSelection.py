"""
STATS PROJECT 2 - DECODING

Created on Thu Jul 29 15:50:22 2021

@author: tydingsmcclary
"""

#loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
from nilearn import datasets
from nilearn import plotting, image, masking
from nilearn.input_data import NiftiMasker
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
#from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

#fetching subjects 1 to 4
haxby_dataset = datasets.fetch_haxby(subjects=[1, 2, 3, 4])
haxby_path = haxby_dataset.func
prt1_bhv = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
prt2_bhv = pd.read_csv(haxby_dataset.session_target[1], sep=" ")
prt3_bhv = pd.read_csv(haxby_dataset.session_target[2], sep=" ")
prt4_bhv = pd.read_csv(haxby_dataset.session_target[3], sep=" ")

#subsetting data
## RUNS
sum(prt1_bhv['chunks'] == prt4_bhv['chunks']) == sum(prt1_bhv['chunks'] == prt3_bhv['chunks']) == sum(prt1_bhv['chunks'] == prt2_bhv['chunks'])
#chunks are ordered the same across participants
runs_complete = prt1_bhv['chunks'].values
nruns = len(np.unique(runs_complete)) #number of runs

## CONDITIONS
sum(prt1_bhv['labels'] == prt2_bhv['labels']) == sum(prt1_bhv['labels'] == prt3_bhv['labels']) == sum(prt1_bhv['labels'] == prt4_bhv['labels'])
#conditions are not ordered the same, so we need individually for every participant
prt1_cond_compl = prt1_bhv['labels'].values
prt2_cond_compl = prt2_bhv['labels'].values
prt3_cond_compl = prt3_bhv['labels'].values
prt4_cond_compl = prt4_bhv['labels'].values

## Selecting all but rest conditions
prt1_conditions_idx = prt1_cond_compl != 'rest'
prt2_conditions_idx = prt2_cond_compl != 'rest'
prt3_conditions_idx = prt3_cond_compl != 'rest'
prt4_conditions_idx = prt4_cond_compl != 'rest'

#prt1
prt1_runs = runs_complete[prt1_conditions_idx]
prt1_conditions = prt1_cond_compl[prt1_conditions_idx]
#prt2
prt2_runs = runs_complete[prt2_conditions_idx]
prt2_conditions = prt2_cond_compl[prt2_conditions_idx]
#prt3
prt3_runs = runs_complete[prt3_conditions_idx]
prt3_conditions = prt3_cond_compl[prt3_conditions_idx]
#prt2
prt4_runs = runs_complete[prt4_conditions_idx]
prt4_conditions = prt4_cond_compl[prt4_conditions_idx]



## Using NiftiMasker for additional preprocessing steps
# we can use the same mask as the runs are all ordered the same across participants
# we just have to fit it again every time to the correct data of the corresponding participant

#in loop
#define empty lists first
mask = []
epi = []
fmri_masked = []
cdata = []
cond_ind = [prt1_conditions_idx, prt2_conditions_idx, prt3_conditions_idx, prt4_conditions_idx]

for p in range(0,len(haxby_path)):
    #defining masker
    masker = NiftiMasker(standardize = True, runs = runs_complete, detrend = True, 
                    mask_strategy='epi', smoothing_fwhm=4, high_pass=1/128, t_r=2.5)
    #masking + image
    masker.fit(haxby_path[p])
    mask.append(masker.mask_img_)
    epi.append(image.mean_img(haxby_path[p]))
    plotting.plot_epi(epi[p], cut_coords=(0,0,0))
    plotting.plot_roi(mask[p], epi[p], cut_coords=(0,0,0))
    #data
    fmri_masked.append(masker.transform(haxby_path[p]))
    cdata.append(fmri_masked[p][np.where(cond_ind[p]), :])
    cdata[p] = cdata[p][0]
#add: cdata[p] = cdata[p][0] !!!! (correct format of cdata !!)
    


### TASK 1                                                                 ###
### a) Multinomial Logistic Regression vs. SVM (one vs. rest approach)     ###
### b) Decoding accuracy of individual classes                             ###
### c) Nested Cross Validation for Parameter Search                        ###


#SVC
svc = make_pipeline(StandardScaler(), SelectKBest(f_classif, k = 1500),
                    svm.LinearSVC(C=1.,max_iter=5000, penalty='l2'
                                  , multi_class='ovr'
                                  , class_weight='balanced', random_state=0)
                    )
                    
                         
#LogReg
logreg = make_pipeline(StandardScaler(), SelectKBest(f_classif, k = 1500),
                       LogisticRegression(C=1., penalty='l2'
                                          , multi_class='ovr'
                                          , max_iter=5000, solver='lbfgs', 
                                          class_weight='balanced', random_state=0)
                       )
                            


svc_acc = np.empty([len(cdata), nruns])
logreg_acc = np.empty([len(cdata), nruns])
runs = [prt1_runs, prt2_runs, prt3_runs, prt4_runs]
conditions = [prt1_conditions, prt2_conditions, prt3_conditions, prt4_conditions]

#for b) -> making an additional list for each category to store accuracy per fold
unique_cond = np.unique(conditions[0])
category_list = []
for c in range(0, len(unique_cond)):
    #for every category we create a list inside the list
    category_list.append(np.empty([2 #SVC vs. LogReg 
                                   ,4 #4 participants
                                   ,nruns #12 folds
                                   ])
                         )




for p in range(0, len(cdata)): 
    #every participant
    print('\nSVC vs Multinomial Logistic Regression\nSubject {} \n__________________'.format(p+1))
    for fold in range(0, nruns): 
        #fold
        x_test  = cdata[p][np.where(runs[p]==fold),:][0] 
        #since we are working with everything stored in lists, we need to specify again that this is the list inside the list
        Y_test  = conditions[p][np.where(runs[p]==fold)]
        x_train = cdata[p][np.where(runs[p]!=fold),:][0] 
        Y_train = conditions[p][np.where(runs[p]!=fold)]
        f_svc   = svc.fit(x_train, Y_train)
        svc_acc[p][fold] = f_svc.score(x_test, Y_test)
        f_logreg = logreg.fit(x_train, Y_train)
        logreg_acc[p][fold] = f_logreg.score(x_test, Y_test)
        
        # gathering category specific accuracy for b)
        for c in range(0,len(category_list)):
            #SVC score
            category_list[c][0][p][fold] = f_svc.score(x_test[Y_test == unique_cond[c],:], Y_test[Y_test==unique_cond[c]])
            #LogReg score
            category_list[c][1][p][fold] = f_logreg.score(x_test[Y_test == unique_cond[c],:], Y_test[Y_test==unique_cond[c]])
            
        
        print('Fold: {0:3d}'.format(fold), end='')
        print(' | SVC: {0:6.2f}'.format(svc_acc[p][fold]*100), end='%')
        print(' | LogReg: {0:6.2f}'.format(logreg_acc[p][fold]*100), end='%')
        if fold+1 < nruns:
            print(' | Next fold.')
        else:
            print('\nDONE!')
    print('*****')
    print('SVC Accuracy for Subject %d: {0:6.2f}'.format(np.mean(svc_acc[p])*100)%(p+1), end='%')
    print('\n*****')
    print('Logistic Regression Accuracy for Subject %d: {0:6.2f}'.format(np.mean(logreg_acc[p])*100)%(p+1), end='%')
    print('\n_________________________________________________________\n')
    
        
    
        # ... #to be continued here!!        
        

#assigning the category information to individual variables
bottle = category_list[0]
cat = category_list[1]
chair = category_list[2]
face = category_list[3]
house = category_list[4]
scissors = category_list[5]
scramblepix = category_list[6]
shoe = category_list[7]
        

## FIGURES 
# a) individual subjects
for p in range(0, len(cdata)):
    plt.figure(figsize=(10, 7))
    plt.boxplot([svc_acc[p], logreg_acc[p]], showmeans=True, 
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"green"})
    plt.scatter(np.random.normal(1, 0.02, len(svc_acc[p])), svc_acc[p], s = 30, color = 'b')
    plt.scatter(np.random.normal(2, 0.02, len(logreg_acc[p])), logreg_acc[p], s = 30, color = 'r')
    plt.xticks([1, 2], ['SVC', 'LogisticRegression'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Subject %d'%(p+1))
    plt.show()

# a) complete    
plt.figure(figsize=(10, 7))
plt.boxplot([np.mean(svc_acc, axis=1), np.mean(logreg_acc, axis=1)], showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"green"})
plt.scatter(np.random.normal(1, 0.01, len(cdata)), np.mean(svc_acc, axis=1), s = 30, color='b')
plt.scatter(np.random.normal(2, 0.01, len(cdata)), np.mean(logreg_acc, axis=1), s = 30, color='r')
plt.xticks([1, 2], ['SVC', 'LogisticRegression'])
plt.ylabel('Accuracy')
plt.title('Accuracy Total')
plt.show()


# b) for the individual classes

plt.figure(figsize=(8, 6))

classifiers = ['SVC', 'LogisticRegression']
all_categories = unique_cond
tick_position = np.arange(len(all_categories))
plt.yticks(tick_position + 0.25, all_categories)
height = 0.1

for i, (color, classifier_name) in enumerate(zip(['b','r'],
                                                 classifiers)):
    score_means = [
        np.mean(category_list[c][i])*100
        for c in range(0,len(category_list))
    ]

    plt.barh(tick_position, score_means,
             label=classifier_name,
             height=height, color=color)
    tick_position = tick_position + height

plt.xlabel('Accuracy Score in % True Positives')
plt.ylabel('Category')
plt.axvline(x=50, color='m', linestyle='--')
plt.xlim(xmax=100)
plt.legend(loc='lower right', ncol=1)
plt.title(
    'Classification Accuracy for Different Categories')
plt.tight_layout()        

        
# c) Searching the Grid for best C hyperparameter

#we are going to use the GridSearchCV function from sklearn
from sklearn.model_selection import GridSearchCV, GroupKFold 

"""
First, let us set up the overall test and train data per participant. For this
we will just take runs 0-5 & 9-11 (9 runs, so 75% of our data) as our training 
data inside the cross validation scheme to find the best C hyperparameter (& 
maybe the best k for SelectKBest) and then train our model with the entire 
train data with these parameters and test on runs 6-8 (3 runs, the other 25%).
"""

train_runs_idx = []
test_runs_idx = []
train_runs = []
test_runs = []
train_conditions = []
test_conditions = []
train_cdata = []
test_cdata = []


for p in range(0, len(runs)):
    #making index to correctly split all the data
    test_runs_idx.append(np.logical_not(np.logical_or(runs[p]<6, runs[p]>8)))
    train_runs_idx.append(np.logical_or(runs[p]<6, runs[p]>8))
    #splitting data accordingly
    #runs
    test_runs.append(runs[p][test_runs_idx[p]])
    train_runs.append(runs[p][train_runs_idx[p]])
    #conditions
    test_conditions.append(conditions[p][test_runs_idx[p]])
    train_conditions.append(conditions[p][train_runs_idx[p]])
    #cdata
    train_cdata.append(cdata[p][train_runs_idx[p], :])
    test_cdata.append(cdata[p][test_runs_idx[p], :])
    
#we forget about the test data for now, as we only use the train data for our 
#parameter search. we will use our svc classifier from above again:

#SVC
svc = make_pipeline(StandardScaler(), SelectKBest(f_classif, k = 1500),
                    svm.LinearSVC(C=1.,max_iter=5000, penalty='l2'
                                  , multi_class='ovr'
                                  , class_weight='balanced', random_state=0)
                    )

#now we will specify our cross validation method with GroupKFold
gcv = GroupKFold(n_splits=9)

#and the parameter grid to search
p_grid = {'selectkbest__k': np.linspace(1000,2000,9, dtype='int'),
          'linearsvc__C': np.logspace(-5,3,num=9,base=10.)
          }

#and empty lists to append our results of interest to
best_parameters=[]
best_cv_score = []
svc_score = []


# looping over every participant, first doing the GridSearch, then taking the
# best parameters to fit a model on the entire training data and then get the
# score for the performance on the test data !

for p in range(0,len(haxby_path)):
    
    print('\nSearch for best SVC C Parameter & best k Features \nSubject {} \n__________________'.format(p+1))
    
    #first we define the GridSearch
    g_search = GridSearchCV(svc
                            , param_grid = p_grid #, verbose=2
                            , cv = gcv.split(train_cdata[p],train_conditions[p],train_runs[p])
                            )
    
    #now we use the grid search on the train data
    g_search.fit(train_cdata[p], train_conditions[p])
    
    #once the best parameters are found
    best_parameters.append(g_search.best_params_)
    best_cv_score.append(g_search.best_score_)
    print('*****')
    print('Best C Parameter for Subject %d: {}'.format(best_parameters[p]['linearsvc__C'])%(p+1), end='')
    print('\nBest k Features for Subject %d: {}'.format(best_parameters[p]['selectkbest__k'])%(p+1), end='')
    print('\n*****')
    print('Best CV Score for Subject %d: {0:6.2f}'.format(best_cv_score[p]*100)%(p+1), end='%')
    
    #now we make a new classifier with the best parameters
    svc_bp = svc.set_params(selectkbest__k = g_search.best_params_['selectkbest__k']
                            , linearsvc__C = g_search.best_params_['linearsvc__C']
                            )
    print('\nClassifier with best parameters:\n{}'.format(svc_bp))
    #and fit to complete training set and use to predict test set
    svc_bp.fit(train_cdata[p], train_conditions[p])
    svc_score.append(svc_bp.score(test_cdata[p], test_conditions[p]))
    print('\n*****')
    print('Accuracy for Subject %d on test set: {0:6.2f}'.format(svc_score[p]*100)%(p+1), end='%')
    print('\n_________________________________________________________\n')
    


    
    


















