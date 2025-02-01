from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a secure key in production

combined_df = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global combined_df, model
    files = request.files.getlist('files[]')
    dataframes = []

    for file in files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            flash(f"Couldn't read {file.filename}. Error: {str(e)}", "error")
            return redirect(url_for('index'))

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        flash('Files uploaded successfully! 🎉 Let’s make some predictions!')

        # Assuming there is a 'time' column; otherwise, you'll need to adjust
        if 'time' in combined_df.columns:
            combined_df['time'] = pd.to_datetime(combined_df['time'], errors='coerce')
            combined_df['Year'] = combined_df['time'].dt.year
            combined_df = combined_df.dropna(subset=['time', 'prcp'])
        else:
            # Create a Year column if no 'time' column is present
            combined_df['Year'] = np.arange(2000, 2000 + len(combined_df))  # Example year generation
            combined_df = combined_df.dropna(subset=['prcp'])

        # Train the linear regression model
        X = combined_df['Year'].values.reshape(-1, 1)
        y = combined_df['prcp'].values

        model = LinearRegression()
        model.fit(X, y)
    else:
        flash('Oops! No files uploaded. Let’s try this again!', 'error')

    return redirect(url_for('index'))

def predict_rainfall_for_year(year):
    """Predict rainfall for a given year using the trained model."""
    if model is None:
        return None
    return model.predict(np.array([[year]]))[0]

@app.route('/predict', methods=['POST'])
def predict_rainfall():
    global combined_df
    if combined_df is None:
        flash("No data available. Please upload files first!", "error")
        return redirect(url_for('index'))

    year = int(request.form.get('year'))

    predicted_value = predict_rainfall_for_year(year)

    if predicted_value is None:
        flash(f"No data available for prediction for {year}.", "error")
        return redirect(url_for('index'))

    flash(f'Predicted Rainfall for {year}: {predicted_value:.2f} mm')

    # Create a bar chart for visualization
    plt.figure(figsize=(12, 7))

    # Plot historical rainfall
    historical_data = combined_df.groupby('Year')['prcp'].mean().reset_index()
    plt.bar(historical_data['Year'], historical_data['prcp'], color='lightgray', label='Historical Rainfall (mm)', alpha=0.5)

    # Plot predicted rainfall
    plt.bar(year, predicted_value, color='blue', label='Predicted Rainfall (mm)', alpha=0.9)

    # Add data labels on top of the bars
    plt.text(year, predicted_value, f'{predicted_value:.2f} mm', ha='center', va='bottom')

    plt.xlabel('Year')
    plt.ylabel('Rainfall (mm)')
    plt.title('Predicted vs Historical Rainfall')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding a legend
    plt.legend()
    plt.tight_layout()

    # Save the plot to a BytesIO object and encode it to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('results.html', plot_url=plot_url, predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
