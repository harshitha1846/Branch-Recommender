from flask import Flask,request,render_template
import pandas
import numpy
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# for creating confusion matrix 
# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt      
app = Flask(__name__) 



@app.route('/') 
def home():  
   return render_template('index.html')
@app.route('/question') 
def question():  
   return render_template('question.html')
@app.route('/submit',methods = ['GET'])
def get_value():
    #collecting answers
    li=[]
    li.append(request.args.get('ans1'))
    li.append(request.args.get('ans2'))
    li.append(request.args.get('ans3'))
    li.append(request.args.get('ans4'))
    li.append(request.args.get('ans5'))
    li.append(request.args.get('ans6'))
    li.append(request.args.get('ans7'))
    li.append(request.args.get('ans8'))
    li.append(request.args.get('ans9'))
    li.append(request.args.get('ans10'))
    li.append(request.args.get('ans11'))
    li.append(request.args.get('ans12'))
    li.append(request.args.get('ans13'))
    li.append(request.args.get('ans14'))
    li.append(request.args.get('ans15'))
    li.append(request.args.get('ans16'))
    li.append(request.args.get('ans17'))
    li.append(request.args.get('ans18'))
    li.append(request.args.get('ans19'))
    li.append(request.args.get('ans20'))
    li.append(request.args.get('ans21'))
    li.append(request.args.get('ans22'))
    li.append(request.args.get('ans23'))
    li.append(request.args.get('ans24'))
    li.append(request.args.get('ans25'))
    li.append(request.args.get('ans26'))
    li.append(request.args.get('ans27'))
    li.append(request.args.get('ans28'))
    li.append(request.args.get('ans29'))
    li.append(request.args.get('ans30'))
    #print(li)
    data(li)
    return render_template('submit.html')
    
def data(ans):
    global ypred, model, le
    
    #converting ans to numpy array and reshaping it to required shape
    ans=numpy.asarray(ans).reshape(1,30)
    
    #converting numpy array to dataframe
    df_ans=pandas.DataFrame(ans)
    
    #fitting answers to the model
    df_ans=df_ans.apply(le.transform)
    
    #store predict value
    ypred=model.predict(df_ans)
    
    #convert predict value into list
    ypred=list(ypred)
    
    'print(len(le.classes_))'

@app.route('/branch')
def label():
    global ypred
    num=ypred
    #for debug purpose
    '''if num[0]==0:
        print('CHEMICAL ENGINEERING')
    elif num[0]==1:
        print('CIVIL ENGINEERING')
    elif num[0]==2:
        print('COMPUTER SCIENCE ENGINEERING')
    elif num[0]==3:
        print('ELECTRONICS AND COMMUNICATION ENGINEERING')
    elif num[0]==4:
        print('MECHANICAL ENGINEERING')
    elif num[0]==5:
        print('METALLURGICAL AND MATERIALS ENGINEERING')
    else:
        print("something is wrong")
        print("sorry for inconvienance")'''
    return render_template('branch.html',prediction=num[0])

if __name__=='__main__':
    #declaring variables
    global x_train, x_test, y_train, y_test,model,ypred,le
    
    #reading excel files as dataframes
    df1=pandas.read_excel('dataset1_append.xlsx')
    df2=pandas.read_excel('dataset22_append.xlsx')
    
    #appending dataframes into one
    df=df1.append(df2,ignore_index=True)
    
    #store features in X and target_class in Y
    X=df[df.columns[0:30]]
    Y=df[['Branch']]
    
    #defining encoders for X and Y
    le=preprocessing.LabelEncoder()
    yle=preprocessing.LabelEncoder()
    
    #fitting label encoders for X and Y
    le.fit(numpy.unique(X.values))
    yle.fit(numpy.unique(Y.values))
    
    #Transforming X and Y
    X=X.apply(le.transform)
    Y=Y.apply(yle.transform)
    
    #spliting dataset into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=99)   
    
    # define model
    model = LogisticRegression(multi_class='ovr',max_iter=1000)
    
    # fit model
    model.fit(x_train,y_train.values.ravel())
    
    #uncomment below code to generate confusion matrix
    '''plot_confusion_matrix(model, x_test, y_test,
                                 display_labels=yle.classes_)
    
    plt.show()'''
    
    #running web app
    app.run()
    