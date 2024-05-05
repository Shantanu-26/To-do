#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Practising all programs here:


# In[1]:


import numpy as np
import pandas as pd
import seaborn as ss
import matplotlib.pyplot as plot


# In[10]:


#Linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dp=pd.read_csv('salary - Sheet.csv')

x=dp[['YearsExperience']].values
y=dp['Salary'].values

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=1/3, random_state=0)

lin_reg=LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred=lin_reg.predict(x_test)

plot.scatter(x_train, y_train, color='blue', label='Scattered data')
plot.plot(x_train, lin_reg.predict(x_train), color='red', label='Linear data')
plot.xlabel("Years of experience")
plot.ylabel("Salary")
plot.title("YearsExperience vs Salary")
plot.legend()


# In[19]:


#non-linear regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

polynomial_degree=2

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=1/3, random_state=0)

polynomial_reg=make_pipeline(PolynomialFeatures(degree=polynomial_degree), LinearRegression())

polynomial_reg.fit(x_train, y_train)

y_poly_pred=polynomial_reg.predict(x_test)

plot.scatter(x_train, y_train, color='red', label='training data')
plot.plot(x_train, polynomial_reg.predict(x_train), color='blue', label='testing data')
plot.xlabel('Years of eperience')
plot.ylabel('Salary')
plot.title('TearsExperience vs Salary')
plot.legend()


# In[23]:


#ridge and lasso:
from sklearn.linear_model import Ridge, Lasso
rd=Ridge(alpha=3)
rd.fit(x_train, y_train)
rd.score(x_test, y_test)


# In[25]:


ls=Lasso(alpha=3)
ls.fit(x_train, y_train)
ls.score(x_test, y_test)


# In[37]:


#Logistics regression:
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dp=pd.read_csv('User_Data.csv')

x=dp.iloc[:, [2,3]].values
y=dp.iloc[:, [4]].values

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.25, random_state=0)
print(y_train)


# In[38]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_train[0:10])


# In[40]:


classifier=LogisticRegression(random_state=42)
classifier.fit(x_train, y_train)


# In[41]:


y_pred=classifier.predict(x_test)
y_pred


# In[42]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)

print("Confusion matrix:", cm)
print("Accuracy score:", accuracy)


# In[ ]:


#bayesian classifier:
from matplotlib.colors import ListedColormap

x_set, y_set=x_test, y_test

x1, x2=np.meshgrid()


# In[ ]:





# In[55]:


#K-means clustering
from sklearn.cluster import KMeans
dp=pd.read_csv('Mall_customers.csv')

x = dp[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Initialize K-means clustering with 5 clusters, using 'k-means++' initialization
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

# Fit and predict the clusters
y_predict = kmeans.fit_predict(x)

colors=['red', 'blue', 'green', 'orange', 'violet']
for i in range(5):
    plt.scatter(x[y_predict==i,0], x[y_predict==i,1], s=100, c=colors[i], label=i)
    
plot.title("K-means clustering")
plot.xlabel("Annual Income (k$)")
plot.ylabel("Spending Score (1-100)")
plot.legend()
plot.grid()
plot.show()


# In[56]:


#Heirarchical clustering
from sklearn.cluster import AgglomerativeClustering

df=pd.read_csv('Mall_customers.csv')

hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

x=dp[['Annual Income (k$)', 'Spending Score (1-100)']].values

y_predict=hc.fit_predict(x)

colors=['pink', 'yellow', 'cyan', 'magenta', 'orange']
for i in range(5):
    plt.scatter(x[y_predict==i,0], x[y_predict==i,1], s=100, c=colors[i], label=i)
    
plot.title("Heirarchical clustering")
plot.xlabel("Annual Income (k$)")
plot.ylabel("Spending Score (1-100)")
plot.legend()
plot.grid()
plot.show()


# In[62]:


#Monte carlo area question:
circle_radius=int(input("Enter radius of circle:"))
circle_center=(0,0)

rectangle_length=int(input("Enter length of rectangle:"))
rectangle_breadth=int(input("Enter breadth of rectangle:"))
rectangle_center=(0,0)

total_points=int(input("Enter total number of points"))

points=np.random.uniform(-6, 6, size=(total_points, 2))

points_in_circle=[]
points_in_rectangle=[]

for point in points:
    x, y=point
    if(x-circle_center[0])**2+(y-circle_center[1])**2<=circle_radius**2:
        points_in_circle.append(point)
    if(abs(x-rectangle_center[0])<=rectangle_breadth/2) and (abs(y-rectangle_center[0]<=rectangle_length/2)):
        points_in_rectangle.append(point)
        
area_circle=np.pi*circle_radius**2
area_rectangle=rectangle_length*rectangle_breadth

num_circle=len(points_in_circle)
num_rectangle=len(points_in_rectangle)

print("Area of circle:", area_circle)
print("Area of rectangle:", area_rectangle)
print("Number of points in circle:", num_circle)
print("Number of points in rectangle:", num_rectangle)
print("Ratio of areas:", area_circle/area_rectangle)
print("Ratio of number of points:", num_circle/num_rectangle)


# In[70]:


#Card colors
num_colors=int(input("Enter number of colors:"))
colors=[]
card_counts=[]

for i in range(num_colors):
    color_name=input(f"Enter name for color{i+1}:")
    num_cards=int(input("Enter number of cards of this color:"))
    colors.append(color_name)
    card_counts.append(num_cards)
    
total_cards=sum(card_counts)

total_draws=int(input("Enter number of cards to be drawn:"))
drawn_cards=np.random.choice(colors, size=total_draws, p=[count/total_cards for count in card_counts])

unique_colors, color_drawn_counts=np.unique(drawn_cards, return_counts=True)
drawn_color_distribution=dict(zip(unique_colors, color_drawn_counts))

print("Result of Monte Carlo Simulation:")
for color, count in drawn_color_distribution.items():
    estimated_probability=count/total_draws
    print(color, ";", count, "Estimated probability:", estimated_probability)


# In[81]:


#Coin toss:
total_flips=int(input("Enter number of times coin is to be tossed"))

coin_flips=np.random.choice(["H", "T"], size=total_flips)

num_heads=np.sum(coin_flips=="H")
num_tails=np.sum(coin_flips=="T")

prob_heads=num_heads/total_flips
prob_tails=num_tails/total_flips

print("Result of Monte Carlo simulation:")
print("Number of heads:", num_heads)
print("Number of tails:", num_tails)
print("Probability of heads:", num_heads)
print("probability of tails:", num_tails)


# In[87]:


#Dice problem:
total_rolls=int(input("Enter number of times a dice is rolled:"))

dice_rolls=np.random.randint(1, 7, size=total_rolls)

unique_faces, face_counts=np.unique(dice_rolls, return_counts=True)
face_distribution=dict(zip(unique_faces, face_counts))

estimated_probabilities={face: count/total_rolls for face, count in face_distribution.items()}

print("Monte Carlo simulation for rolling a dice:")
print("Total dice rolls", "total_rolls")
for face, count in face_distribution.items():
    probability=estimated_probabilities[face]
    print("Face", {face}, ":", {count}, "times", "Estimated probability:", probability)


# In[88]:


#SVM:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Preprocessing
# Drop irrelevant columns or columns with too many missing values
titanic_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# Handling missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data.dropna(inplace=True)

# Convert categorical variables to numerical
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# Split the data into features and target
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions
y_pred = svm_classifier.predict(X_test)

y_test=y_test.values
print("Actual Outcomes = ",y_test,'\n')
print("Predictions = ", y_pred,'\n')

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy,'\n')

from sklearn.metrics import precision_score, recall_score, f1_score

# Precision, Recall, and F1-score for testing dataset
precision_test = precision_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred)

print("Testing Dataset Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1 Score:", f1_test)

# Predictions on training data
y_train_pred = svm_classifier.predict(X_train)

# Precision, Recall, and F1-score for training dataset
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

print("\nTraining Dataset Metrics:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1 Score:", f1_train)


# In[ ]:




