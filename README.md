# CS-145-Project
Class project to predict COVID-19 cases and deaths

## Team9 Members
- Jakin Wang (team leader)
- Melina Diaz
- Jiyuan Xiao
- Diego Fernandez-Salvador

# To Run
In this project, there are three regression models can be run :
- Linear Regression model
- Quadratic Regression model
- Cubic Regression model

## Run by locally downloaded Python
- You can run this program with the Python compiler in you computer.
- But you have to make sure numpy and pandas package have been installed beforehand.
```
You can run Linear Regression model by double clicking LinearRegression.py

You can run Quadratic Regression model by double clicking QuadraticRegression.py

You can run Cubic Regression model by double clicking CubicRegression.py
```
- Result csv files show have been saved to current directory.

## Run by Visual Studio Code

### evnironment setting
1. First open Visual Studio Code, opne a new folder, then select our folder "CS-145-Project".

2. After opening the folder in Visual Studio Code, open the terminal in Visual Studio Code.

"package-lock.json" and "node_modules" for Visual Studio Code has been include in the folder.

All models program should be able to run in Visual Studio Code immediately if you have download Python(Extension) in Visual Studio Code.

Try:
```
python ./LinearRegression.py
```
If there is no error occur, please ignore the following step, and skip to "run programs".

Sometime you may need to select Python Interpreter first, you can do it by:
```
Ctrl+Shift+P
```
search and select your python environment. 

More information about python enviroment in Visual Studio Code: [https://code.visualstudio.com/docs/python/environments]

### run programs

To run Linear, Quadratic, or Cubic Regression model
```
python ./LinearRegression.py
or
python3 ./LinearRegression.py

python ./QuadraticRegression.py
or
python3 ./QuadraticRegression.py

python ./CubicRegression.py
or
python3 ./CubicRegression.py
```
If there is error, please try:
```
npm install
```
programs should be able to run after packages installation.

If programs are run successfully, result data should have been saved in csv files 
```
"result_Linear_Reg.csv" 

"result_Quadratic_Reg.csv" 

"result_Cubic_Reg.csv" 
```
which will locate at current directory.

csv files created above are in format 'submssion.csv', in which data about States and Date will not be included.

If you wan to have result in format 'test.csv' (data about States and Date will be included), you can open and edit three .py program:
1. Go to the last line of the code
2. replace the second strig parameter 'submission' by 'test'
3. save and run 

Now reuslt should be in format 'test.csv'.

## "SIRModel.py"
- The "SIRModel.py" is not a runable program, but there are functions in it which are our attempt on SIR model.
- Unfortunately, we failed to implement SIR bcause we were not able to make good prediction on the missing data.

## "Visualizations to Evaluate Results"
- This is an example of how we exaluate our results by print tables and graphs.
- Due to some unknown issues, for 3 out of our 9 submissions, the order of data was shuffled after submitting through Kaggle, including the highest score submission.
- We visualized both the submitted data in Kaggle and the data produced by our programs in thses visualization ipynb files.