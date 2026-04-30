import mlflow
import dagshub
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("https://dagshub.com/RosyPaul/mlflow-dagshub-demo.mlflow")
dagshub.init(repo_owner='RosyPaul', repo_name='mlflow-dagshub-demo', mlflow=True)


iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state= 42)

max_depth = 10

mlflow.set_experiment('iris-dt')
with mlflow.start_run():
    dt=DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)
    y_pred= dt.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.ylabel('actutal')
    plt.xlabel('Predicted')
    plt.title('COnfusion Matrix')

    plt.savefig("Confusion_matrix")

    mlflow.log_artifact('Confusion_matrix.png')

    mlflow.log_artifact(__file__)
    # mlflow.sklearn.log_model(dt,"decision tree")

    mlflow.set_tag('author','rosy')
    

    print('accuracy',accuracy)



