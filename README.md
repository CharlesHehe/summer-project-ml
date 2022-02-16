# summer-project-ml

0. run ```pip install -r requirements.txt```



1. all raw datasets should be put into "raw_data" directory

<img src="https://user-images.githubusercontent.com/45914103/153745493-497b32b7-f7cb-420c-9533-b079b88519dc.png" width="200" height="100">

2. run ```python main.py``` to train all models. All model names are documented in model_list.txt

<img src="https://user-images.githubusercontent.com/45914103/153745798-aa0d7470-d2ca-44b5-8ea1-e7fd5c6570f6.png" width="200" height="150">

3. Models that have trained will be stored under "models" directory.

<img src="https://user-images.githubusercontent.com/45914103/153745553-1b3da0a4-b5be-4f7d-90e4-306f50286431.png" width="200" height="150">

4. replace the "model_name".joblib with models under "model" directory.

<img src="https://user-images.githubusercontent.com/45914103/154376956-cc07d5f1-8c84-49f6-a13c-d6005e927159.png" width="300" height="150">


5. ```mlserver start .``` to start a listening server
6. make requests for prediction ```python test.py``` using mlserver


| Algorithm | accuracy |
| ----------- | ----------- |
| DecisionTreeClassifier | 0.9999852982254959 |
| RandomForestClassifier | 0.9996324556373954 |
| KNeighborsClassifier | 0.9880474573280995 |
| SGDClassifier | 0.9607462620738323 |
| LogisticRegression | 0.9607315602993282 |
| SVM | 0.959673032535027 |
| GaussianNB | 0.554492127199753 |
