# CS-145-Project
Class project to predict COVID-19 cases and deaths

## Team9 Members
- Jakin Wang (team leader)
- Melina Diaz
- Jiyuan Xiao
- Diego Fernandez-Salvador

# To Run
In this project, we tried three regression models:
- Linear Regression model
- Quadratic Regression model
- Cubic Regression model

## Run by Visual Studio Code
Settings.json file for Visual Studio Code has been include in the folder.

All models program should be able to run in Visual Studio Code immediately if you have download Python in Visual Studio Code.

First open Visual Studio Code, opne a new folder, then select our folder "CS-145-Project".

After opening the folder in Visual Studio Code, open the terminal in Visual Studio Code.

To run Linear Regression model:
```
python ./LinearRegression.py
```
The result csv file "result_Linear_Reg.csv" should be save into the current directory.

To run Quadratic Regression model:
```
python ./QuadraticRegression.py
```
The result csv file "result_Quadratic_Reg.csv" should be save into the current directory.

To run Cubic Regression model:
```
python ./CubicRegression.py
```
The result csv file "result_Cubic_Reg.csv" should be save into the current directory.

csv files created above are in format 'submssion.csv', in which data about States and Date will not be included.

If you wan to have result in format 'test.csv' (data about States and Date will be included), you can open and edit three .py program:
1. Go to the last line of the code
2. replace the second strig parameter 'submission' by 'test'
3. save and run 

Now reuslt should be in format 'test.csv'.

The "SIRModel.py" is not a runable program, but there are functions in it which are our attempt on SIR model.

Unfortunately, we failed to implement SIR bcause we were not able to make good prediction on the missing data.