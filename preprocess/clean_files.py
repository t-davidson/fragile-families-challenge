import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from fancyimpute import KNN


# , index_col='challengeID')
bg = pd.read_csv('../../ff_data/background.csv', low_memory=False)
bg.cf4fint = ((pd.to_datetime(bg.cf4fint) -
               pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
# Loading bg such that some bugs in labels are fixed, see https://github.com/fragilefamilieschallenge/open-source-submissions/blob/master/rfjz%20-%2011%20submission/FF%20Pre-Imputation.ipynb
# , index_col='challengeID')
train = pd.read_csv('../../ff_data/train.csv', low_memory=False)

bg.index = bg.challengeID
train.index = train.challengeID
del bg['challengeID']
del train['challengeID']
outcomes = list(train.columns)

print("Merging background and outcomes so that missing obs can be imputed")
bg = pd.concat([bg, train], axis=1, join='outer')
print(bg.shape)

# Replace missing codes with NaN
print("Replacing all missing values with NaNs")
bg = bg.replace({-1: np.NaN, -2: np.NaN, -3: np.NaN,
                 -4: np.NaN, -5: np.NaN, -6: np.NaN,
                 -7: np.NaN, -8: np.NaN, -9: np.NaN,
                 "Missing": np.NaN, np.inf: np.NaN,
                 -np.inf: np.NaN})

print("Input matrix shape ", bg.shape)

print("Identifying columns with high missingness")
threshold = 0.7
high_missingness = 0
columns_to_use = []
for c in bg.columns:
    if c not in outcomes:
        missing_prop = (bg[c].isnull().sum() / bg.shape[0])
        if missing_prop > threshold:
            high_missingness += 1
        else:
            columns_to_use.append(c)
    else:  # if it is an outcome column then always keep it
        columns_to_use.append(c)

print(str(high_missingness), " columns with over ",
      str(threshold * 100), " percent missing")

bg_ = bg[columns_to_use]  # Only keeping valid columns

print("Pruned matrix shape ", bg_.shape)

#print("Identifying column types and replacing missing values accordingly")
print("Factorizing string fields")
cat_cols = []
cont_cols = []
cat_vals_dict = {}
for i, c in enumerate(bg_.columns):
    is_categorical = False
    vals = set(list(bg_[c]))
    # Removes nans, otherwise treated as unique
    vals = {x for x in vals if x == x}
    if bg_[c].dtype == 'float64':  # unfortunately this doesn't gaurantee a continuous variable
        if len(vals) < 50:  # heuristic: if less than 50 unique vals then prob cat
            is_categorical = True
            #mode = bg_[c].mode()[0]
            #bg_[c] = bg_[c].fillna(mode)
        else:
            is_categorical = False
            #mean = bg_[c].mean()
            #bg_[c] = bg_[c].fillna(mean)
    else:
        is_categorical = True
        factors, values = pd.factorize(bg_[c])  # factorize it to numeric
        bg_[c] = factors
        #mode = bg_[c].mode()[0]
        #bg_[c] = bg_[c].fillna(mode)
    # Finally, store a list of categorical and non-categorical columns
    if is_categorical:
        cat_cols.append(c)
        cat_vals_dict[c] = vals  # Also store the unique values in a dict
    else:
        cont_cols.append(c)

# Store categorical and non-categorical variables in separate dataframes
categorical = bg_[cat_cols]
continuous = bg_[cont_cols]

# Save the index and cols of the new matrix.
index = bg_.index
cols = bg_.columns

k = 5
print("Now unning KNN imputation using ", k, " nearest neighbours...")
bg_imputed = KNN(k=k).complete(bg_)

print("Converting back to dataframe")
bg_imputed = pd.DataFrame(bg_imputed)
bg_imputed.index = index
bg_imputed.columns = cols

# Just get continuous cols since categorical columns badly imputed
bg_imputed_cont = bg_imputed[cont_cols]
bg_imputed_cat = bg_imputed[cat_cols]


def convert_continuous_to_categorical(categories, orig_col, imputed_col):
    """"Takes a set of categories, the original column, and a column
    that has been imputed using the mean value of the KNN.

    Returns a new column where the imputed values are transformed to the
    nearest numeric category.

    Note: This is quite a rough way to recover the imputed categories,
    consider testing and potentially improving it."""
    new_col = list(orig_col)  # Take vals of orig col, including NAs
    for i, j in enumerate(list(orig_col)):  # iterate through them
        if j != j:  # if val is missing then get its imputed value
            predicted = list(imputed_col)[i]
            # Now distance of pred from all categories
            diffs = []
            for c in categories:
                diffs.append(abs(predicted - c))
            min_idx = diffs.index(min(diffs))  # get index of min distance
            closest_cat = categories[min_idx]  # find corresponding cat
            new_col[i] = closest_cat  # assign cat to value
    return new_col


print("Manually cleaning imputed categorical variables")
for c in bg_imputed_cat.columns:
    categories = list(cat_vals_dict[c])
    orig_col = bg_[c]
    imputed_col = bg_imputed_cat[c]
    new_col = convert_continuous_to_categorical(
        categories, orig_col, imputed_col)
    bg_imputed_cat[c] = new_col

# Now concat them together
bg_ = pd.concat([bg_imputed_cont, bg_imputed_cat], axis=1)

print("Finding remaining missing values and filling with zeros")
bg_ = bg_.fillna(0)

print("Finding columns with zero variance and removing them")
vt = VarianceThreshold()
vfs = vt.fit(bg_)
bg_ = bg_.loc[:, vfs.get_support()]

print("Clipping negative values to 0")
bg_ = bg_.clip_lower(0)

print("Picking outputs...")
bg_.to_pickle('full_imputed.p')
