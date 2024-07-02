import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


data = pd.read_csv('data.csv')
irrelevant = ['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
data = data.drop(irrelevant, axis=1)
const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']
data = data[(data['batting_team'].isin(const_teams)) & (data['bowling_team'].isin(const_teams))]
data = data[data['overs'] >= 5.0]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# One-hot encode teams
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
data = np.array(columnTransformer.fit_transform(data))


cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab', 'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals', 'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad', 'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab', 'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals', 'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(data, columns=cols)
features = df.drop(['total'], axis=1)
labels = df['total']


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)


forest = RandomForestRegressor()
forest.fit(train_features, train_labels)


def predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model=forest):
    prediction_array = []
    if batting_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Delhi Daredevils':
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kings XI Punjab':
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif batting_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif batting_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif batting_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif batting_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif batting_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]
    
    if bowling_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Daredevils':
        prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kings XI Punjab':
        prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]
    
    prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)
    return int(round(pred[0]))


def create_app():
    def get_prediction():
        batting_team = batting_team_var.get()
        bowling_team = bowling_team_var.get()
        runs = int(runs_var.get())
        wickets = int(wickets_var.get())
        overs = float(overs_var.get())
        runs_last_5 = int(runs_last_5_var.get())
        wickets_last_5 = int(wickets_last_5_var.get())
        
        score = predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5)
        result_label.config(text=f"Predicted Score: {score}")
    
    root = tk.Tk()
    root.title("IPL-ML")
    
    
    image = Image.open("cricket_image.jpg") 
    image = image.resize((900, 300), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    label_image = tk.Label(root, image=photo)
    label_image.grid(row=0, column=0, columnspan=2, pady=10)


    ttk.Label(root, text="Batting Team:").grid(row=1, column=0, sticky=tk.W, pady=2)
    batting_team_var = tk.StringVar()
    batting_team_entry = ttk.Combobox(root, textvariable=batting_team_var)
    batting_team_entry['values'] = const_teams
    batting_team_entry.grid(row=1, column=1, pady=2)

    ttk.Label(root, text="Bowling Team:").grid(row=2, column=0, sticky=tk.W, pady=2)
    bowling_team_var = tk.StringVar()
    bowling_team_entry = ttk.Combobox(root, textvariable=bowling_team_var)
    bowling_team_entry['values'] = const_teams
    bowling_team_entry.grid(row=2, column=1, pady=2)

    ttk.Label(root, text="Runs:").grid(row=3, column=0, sticky=tk.W, pady=2)
    runs_var = tk.StringVar()
    runs_entry = ttk.Entry(root, textvariable=runs_var)
    runs_entry.grid(row=3, column=1, pady=2)

    ttk.Label(root, text="Wickets:").grid(row=4, column=0, sticky=tk.W, pady=2)
    wickets_var = tk.StringVar()
    wickets_entry = ttk.Entry(root, textvariable=wickets_var)
    wickets_entry.grid(row=4, column=1, pady=2)

    ttk.Label(root, text="Overs:").grid(row=5, column=0, sticky=tk.W, pady=2)
    overs_var = tk.StringVar()
    overs_entry = ttk.Entry(root, textvariable=overs_var)
    overs_entry.grid(row=5, column=1, pady=2)

    ttk.Label(root, text="Runs Last 5 Overs:").grid(row=6, column=0, sticky=tk.W, pady=2)
    runs_last_5_var = tk.StringVar()
    runs_last_5_entry = ttk.Entry(root, textvariable=runs_last_5_var)
    runs_last_5_entry.grid(row=6, column=1, pady=2)

    ttk.Label(root, text="Wickets Last 5 Overs:").grid(row=7, column=0, sticky=tk.W, pady=2)
    wickets_last_5_var = tk.StringVar()
    wickets_last_5_entry = ttk.Entry(root, textvariable=wickets_last_5_var)
    wickets_last_5_entry.grid(row=7, column=1, pady=2)

    
    predict_button = ttk.Button(root, text="Predict Score", command=get_prediction)
    predict_button.grid(row=8, column=0, columnspan=2, pady=10)

    
    result_label = ttk.Label(root, text="Predicted Score: ")
    result_label.grid(row=9, column=0, columnspan=2, pady=10)

    root.mainloop()


create_app()
