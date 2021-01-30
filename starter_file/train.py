from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run

print("Getting run object")
run = Run.get_context()

print("Reading data from CSV file")
dataframe = pd.read_csv('./breast_cancer_dataset.csv')
x = dataframe.drop('diagnosis', axis=1)
y = dataframe['diagnosis']

print("Splitting data")
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', 
                        type=float, 
                        default=1.0, 
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    
    parser.add_argument('--max_iter', 
                        type=int, 
                        default=100, 
                        help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    print("Calling LogisticRegression")
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    print("Accuracy is {}".format(accuracy))
    
    # Store model
    os.makedirs('output', exist_ok=True)
    joblib.dump(value=model, filename='output/model.pkl')
    
if __name__ == '__main__':
    main()