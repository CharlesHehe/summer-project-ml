# summer-project-ml

0. prepare raw data, named "raw.txt"
1. ```pip install -r requirements.txt```
2. ```python preprocessing.py``` to process raw data
3. ```python scikit-learn-logistic-regression.py``` to train model
4. ```mlserver start .``` to start a listening server
5. make requests for prediction ```python test.py```


| Algorithm | training accuracy on buttons.txt | testing accuracy on bandsaw.txt |
| ----------- | ----------- | ----------- |
| RandomForestClassifier | 0.9996324556373954 | 0.9985213846472742 |
| DecisionTreeClassifier | 0.9999852982254959 | 0.9985213846472742 |
| KNeighborsClassifier | 0.9880474573280995 | 0.8084571270641885 |
| SGDClassifier | 0.9607462620738323 | 0.8758377668762524 |
| LogisticRegression | 0.9607315602993282 | 0.8758377668762524 |
| SVM | 0.959673032535027 | 0.8758377668762524 |
| GaussianNB | 0.554492127199753 | 0.28114419954397846 |
