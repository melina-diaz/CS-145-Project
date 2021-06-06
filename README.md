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
"package-lock.json" and "node_modules" for Visual Studio Code has been include in the folder.

All models program should be able to run in Visual Studio Code immediately if you have download Python(Extension) in Visual Studio Code.

Sometime you may need to select Python Interpreter first, you can do it by:
```
Ctrl+Shift+P
```
search and select yout python environment. 

More information about python enviroment in Visual Studio Code: [https://code.visualstudio.com/docs/python/environments]

1. First open Visual Studio Code, opne a new folder, then select our folder "CS-145-Project".

2. After opening the folder in Visual Studio Code, open the terminal in Visual Studio Code.

To run Linear, Quadratic, and Cubic Regression model:
```
python ./LinearRegression.py

python ./QuadraticRegression.py

python ./CubicRegression.py
```
If there is error, please try:
```
npm install
```
progeams should be run after packages installation.

The result csv files "result_Linear_Reg.csv", "result_Quadratic_Reg.csv", "result_Cubic_Reg.csv" should have been saved to current directory.

csv files created above are in format 'submssion.csv', in which data about States and Date will not be included.

If you wan to have result in format 'test.csv' (data about States and Date will be included), you can open and edit three .py program:
1. Go to the last line of the code
2. replace the second strig parameter 'submission' by 'test'
3. save and run 

Now reuslt should be in format 'test.csv'.

## "SIRModel.py"
- The "SIRModel.py" is not a runable program, but there are functions in it which are our attempt on SIR model.
- Unfortunately, we failed to implement SIR bcause we were not able to make good prediction on the missing data.

## "Visualizations to Evaluate Linear Regression Results.ipynb"
- This is an example of how we exaluate our results by print tables and graphs.