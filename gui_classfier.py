import PySimpleGUI as sg
import classifier

sg.theme('DarkGrey11')
layout = [  [sg.Text('Модел: ',size=(15, 1), font='Lucida',justification='right'),
            sg.Combo(['DecisionTreeClassifier','GaussianNB', 'KNeighborsClassifier'],default_value='DecisionTreeClassifier',key='board')],
            [sg.Text("Sepal length: ",size=(15, 1), font='Lucida',justification='right'),sg.Input()],
            [sg.Text("Sepal width: ",size=(15, 1), font='Lucida',justification='right'),sg.Input()],
            [sg.Text("Petal length: ",size=(15, 1), font='Lucida',justification='right'),sg.Input()],
            [sg.Text("Petal width: ",size=(15, 1), font='Lucida',justification='right'),sg.Input()],
            [sg.Button("Үр дүнг харуулах")],
            [sg.Text("",size=(15, 1), font='Lucida',justification='center', key="result")]]

window = sg.Window("Iris", layout)

while True:
    event, values = window.read()
    if event == "Үр дүнг харуулах":
        window["result"].update(classifier.classify(values[0], values[1], values[2], values[3], values["board"]))

    if event == sg.WIN_CLOSED:
        break

window.close()