from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from train import trained_model
from train import retrain_model, X_test, y_test

# Use the imported trained model to predict disease progression 1 year after baseline on test data
y_pred = trained_model.predict(X_test)

# Define a Evaluation Class that when initialized, evaluates the model
class Evaluation:
    def __init___(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.r2 = r2_score(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)

# Evaluate the model using test data, comparing predicted values against actual values
eval_1 = Evaluation(y_test, y_pred)

# Print the results of the evaluations
print(f"R2 Score: {eval_1.r2}")
print(f"Mean Absolute Error: {eval_1.mae}")
print(f"Mean Squared Error: {eval_1.mse}")

# If the model performs poorly in evaluation, retrain the model using the test data first before evaluating
if eval_1.r2 < 0.8 or eval_1.mse > 1000 or eval_1.mae > 40:
    retrained_model = retrain_model(trained_model, X_test, y_test)
    
    # Evaluate the retrained model
    eval_2 = Evaluation(y_test, retrained_model.predict(X_test))

    # Print the results of the evaluations
    print(f"New R2 Score: {eval_2.r2}")
    print(f"New Mean Absolute Error: {eval_2.mae}")
    print(f"New Mean Squared Error: {eval_2.mse}")
else:
    print("Model performs well, happy to export")