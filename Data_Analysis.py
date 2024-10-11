import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

df.describe()
df.describe(include=object)

# TODO) Measure skewness for all numeric attribute and sort by value. (hint, please refer document of pandas)
numericColumns = df.describe().columns
skewness = df[numericColumns].skew().sort_values(ascending = False)

# TODO) Select the Attributes that skewness value is the largest, smallest and the closet to zero.
#       * For zero skewness attribue, you can specify attribute. e.g., df['RPM']
highSkewAttribute = df[skewness.idxmax()]
lowSkewAttribute = df[skewness.idxmin()]
zeroSkewAttribute = df['Length']


f, axes = plt.subplots(3, 1, figsize=(10, 15))

# set the string
strInPlot = "Skewness: %f"

# Plot distplot (highSkew)
ax = axes[0]
sns.histplot(data=df, x=highSkewAttribute, color="skyblue", ax=ax, kde=True)

# Add legend
ax.text(x=0.97, y=0.97, transform=ax.transAxes, s=strInPlot % highSkewAttribute.skew(),\
    fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right')

# Plot distplot (zeroSkew)
ax = axes[1]
sns.histplot(data=df, x=zeroSkewAttribute, color="khaki", ax=ax, kde=True)

# Add legend
ax.text(x=0.97, y=0.97, transform=ax.transAxes, s=strInPlot % zeroSkewAttribute.skew(),\
    fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right')

# Plot distplot (lowSkew)
ax = axes[2]
sns.histplot(data=df, x=lowSkewAttribute, color="salmon", ax=ax, kde=True)

# Add legend
ax.text(x=0.97, y=0.97, transform=ax.transAxes, s=strInPlot % lowSkewAttribute.skew(),\
    fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()


## Compelete the code
# TODO) Select `Price`, `Horsepower`, `Rev.per.mile` attribte from dataframe, and compute `pearson` correlation value.
selectedDataFrame = df[['Price', 'Horsepower', 'Passengers','Length', 'Weight']]
selectedDataFrame.corr(method='pearson')

# TODO) Plot *pair plot*. The plot must satisify following conditions.
g = sns.pairplot(selectedDataFrame, diag_kind='kde', plot_kws={'alpha':0.5})
plt.show()

# TODO)
# We would like the test that is there a significant `Passengers`  difference between `small` and `midsize` type car.
# Test our hypothesis with proper statistical test with siginificance level as 0.05.
# If there is missing value (`nan`) in `pd.Series`, test will not be working properly.
siginificantLevel = 0.05
smallCar = df[df['Type']=='Small']['Passengers'].dropna()
midsizeCar = df[df['Type']=='Midsize']['Passengers'].dropna()

stats, p = ttest_ind(smallCar, midsizeCar)
print('P-value', format(p, ".19f"))

if p < siginificantLevel:
  print("The Weight of the two types shows a significant difference.")
else:
  print("The Weight of the two types shows no significant difference.")


# TODO) Exclude the outlier (cars) with too much expensive.
## * Assume that `price` of outlier is greater than `Q3 + (1.5 * IQR)`.
attribute = 'Price'

Q1 = df[attribute].quantile(.25)
Q3 = df[attribute].quantile(.75)
IQR = Q3 - Q1
outlierStep = 1.5 * IQR

filtered_df = df[df[attribute] <= Q3 + outlierStep]
    

f, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

sns.histplot(data=df, x='Price', ax=axes[0], kde=True,
             bins=np.arange(0,80,5))
sns.histplot(data=filtered_df, x='Price', color='Red',
             ax=axes[1], kde=True, bins=np.arange(0,80,5))
f.tight_layout()


# TODO) Rescale the numeric attribute using z-score. You can use Scikit-learn or Pandas
numericDf = df.select_dtypes(include='number')
from sklearn.preprocessing import StandardScaler

# create a scaler object
stdScaler = StandardScaler()

# fit and transform the data
std_df = pd.DataFrame(stdScaler.fit_transform(numericDf), columns=numericDf.columns)
std_df.head()


# TODO) Impute the `std_df` with missing value with KNN imputation with parameter `n_neighbors = 2` and `weights` with `distance`.
from sklearn.impute import KNNImputer

imp_KNN = KNNImputer(n_neighbors = 2, weights = 'distance')

impute_df = pd.DataFrame(imp_KNN.fit_transform(std_df), columns=std_df.columns)
impute_df

catergorical_df = df.select_dtypes(exclude='number').drop(
    columns=['Manufacturer', 'Model', 'Make'])
catergorical_df.head()


# TODO) Encode remaining categorical attiribute into numeric attribute using label encoding.
from sklearn.preprocessing import LabelEncoder

# create a encoder object
enc = LabelEncoder()

# create empthy dataframe
enc_df = pd.DataFrame()

# fill up data frame with encoded attribute using iteration
for col in catergorical_df.columns:
  enc_df[col] = enc.fit_transform(catergorical_df[col].astype(str))

enc_df.head()