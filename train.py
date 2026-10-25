from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset with diabetes patients
diabetes = load_diabetes()

# Isolate the features from the target variable
X = diabetes.data
y = diabetes.target

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Instantiate Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save and export the trained model 
trained_model = model

# Define an exportable function to retrain model on remaining test data
def retrain_model(model, X_test, y_test):
    '''
    Retrains model using remaining test data.
    Args:
      model: The current model
      X_test: The features of the test dataset
      y_test: The labeled target of the test dataset
    Returns:
      retrained_model: The retrained model trained on the test data
    '''
    retrained_model = model.fit(X_test, y_test)
    return retrained_model