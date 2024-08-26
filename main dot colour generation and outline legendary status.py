import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\busha\OneDrive\Desktop\Pokemon.csv")

# Define colors for each generation
generation_colors = {
    1: '#E6AACE',
    2: '#1E91D6',
    3: '#F24333',
    4: '#5E239D',
    5: '#315c2b',
    6: '#bdfffd'
}

# Function to get color for a Pokémon based on its generation
def get_generation_color(row):
    return generation_colors[row['Generation']]

# Function to perform linear regression for legendaries and non-legendaries
def linear_regression_by_legendary_status(df):
    models = {}
    for status in [True, False]:
        status_df = df[df['Legendary'] == status]
        X = status_df[['Defense']].values
        y = status_df['Attack'].values
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        models[status] = (model, r2)
    return models

# Function to plot linear regression results by legendary status
def plot_regression_by_legendary_status(df):
    models = linear_regression_by_legendary_status(df)

    plt.figure(figsize=(10, 6))
    for status, (model, r2) in models.items():
        status_df = df[df['Legendary'] == status]
        X = status_df[['Defense']].values
        y = status_df['Attack'].values
        predictions = model.predict(X)
        label = 'Legendary' if status else 'Non-Legendary'
        color = 'gold' if status else 'silver'
        plt.plot(X, predictions, label=f'{label} (R²={r2:.2f})', color=color)

        # Display line equation
        coef = model.coef_[0]
        intercept = model.intercept_
        plt.text(0.05, 0.95 - 0.05 * status, f'{label}: y = {coef:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes)

    for i, row in df.iterrows():
        color = get_generation_color(row)
        edge_color = 'gold' if row['Legendary'] else 'silver'
        plt.scatter(row['Defense'], row['Attack'], color=color, edgecolor=edge_color, linewidth=2.2)

    plt.xlabel('Defense')
    plt.ylabel('Attack')
    plt.title('Linear Regression by Legendary Status')
    plt.legend()
    plt.show()

# Example usage
plot_regression_by_legendary_status(df)
