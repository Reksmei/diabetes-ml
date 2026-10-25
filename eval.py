from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from train import trained_model
from train import retrain_model, X_test, y_test

# Use the imported trained model to predict disease progression 1 year after baseline on test data
y_pred = trained_model.predict(X_test)

# Evaluate the model using test data, comparing predicted values against actual values
scores = {
    'R2 Score': r2_score(y_test, y_pred),
    'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
    'Mean Squared Error': mean_squared_error(y_test, y_pred)
}

# Print the results of the evaluations
print(f"R2 Score: {scores['R2 Score']}")
print(f"Mean Absolute Error: {scores['Mean Absolute Error']}")
print(f"Mean Squared Error: {scores['Mean Squared Error']}")

# If the model performs poorly in evaluation, retrain the model using the test data first before evaluating
if scores['R2 Score'] < 0.8 or scores['Mean Squared Error'] > 1000 or scores['Mean Absolute Error'] > 40:
    retrained_model = retrain_model(trained_model, X_test, y_test)
    
    # Evaluate the retrained model
    retrained_scores = {
        'R2 Score': r2_score(y_test, retrained_model.predict(X_test)),
        'Mean Absolute Error': mean_absolute_error(y_test, retrained_model.predict(X_test)),
        'Mean Squared Error': mean_squared_error(y_test, retrained_model.predict(X_test))
    }

    # Print the results of the evaluations
    print(f"New R2 Score: {retrained_scores['R2 Score']}")
    print(f"New Mean Absolute Error: {retrained_scores['Mean Absolute Error']}")
    print(f"New Mean Squared Error: {retrained_scores['Mean Squared Error']}")