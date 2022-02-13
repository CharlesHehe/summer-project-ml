# summer-project-ml

0. all raw datasets should be put into "raw_data" directory

![image](https://user-images.githubusercontent.com/45914103/153745493-497b32b7-f7cb-420c-9533-b079b88519dc.png)

1. run ```pip install -r requirements.txt```
2. run ```python preprocessing.py``` This is to process all raw datasets and output "train_data.npz" for further training.
3. run ```python main.py knn``` to train knn model. The command "python main.py + model_name" need to use model name as argument. All model names are documented in model_list.txt

![image](https://user-images.githubusercontent.com/45914103/153745596-6700584a-bf62-4ff1-91f7-09f6e75b14ba.png)

4. Models that have trained will be stored under "models" directory.

![image](https://user-images.githubusercontent.com/45914103/153745553-1b3da0a4-b5be-4f7d-90e4-306f50286431.png)

5. ```mlserver start .``` to start a listening server
6. make requests for prediction ```python test.py```


| Algorithm | training accuracy on buttons.txt | testing accuracy on bandsaw.txt |
| ----------- | ----------- | ----------- |
| DecisionTreeClassifier | 0.9999852982254959 | 0.9985213846472742 |
| RandomForestClassifier | 0.9996324556373954 | 0.9985213846472742 |
| KNeighborsClassifier | 0.9880474573280995 | 0.8084571270641885 |
| SGDClassifier | 0.9607462620738323 | 0.8758377668762524 |
| LogisticRegression | 0.9607315602993282 | 0.8758377668762524 |
| SVM | 0.959673032535027 | 0.8758377668762524 |
| GaussianNB | 0.554492127199753 | 0.28114419954397846 |
