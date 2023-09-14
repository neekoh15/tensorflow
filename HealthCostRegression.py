# Load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

dataset = pd.read_csv('insurance.csv')
dataset.tail()

# Convert categorical data to numbers
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Pop off the "expenses" column to create train_labels and test_labels
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

#train_dataset.head()

# Create a Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)

# Train the regressor with the training data
gbr.fit(train_dataset, train_labels)

# Predict on the test set and calculate MAE
test_predictions = gbr.predict(test_dataset)
mae = mean_absolute_error(test_labels, test_predictions)

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)


