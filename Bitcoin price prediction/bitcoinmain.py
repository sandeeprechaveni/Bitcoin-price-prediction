
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CryptoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cryptocurrency Price Predictor")

        self.model = None
        self.scaler = None
        self.data = None  # To store the downloaded data

        # Create models directory if it does not exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Cryptocurrency Selection
        self.crypto_options = ["BTC-USD", "ETH-USD", "LTC-USD"]
        self.selected_crypto = tk.StringVar(root)
        self.selected_crypto.set(self.crypto_options[0])  # default value

        tk.Label(root, text="Select Cryptocurrency:", font=("Arial", 12)).pack(pady=5)
        self.crypto_combobox = ttk.Combobox(root, textvariable=self.selected_crypto, values=self.crypto_options, font=("Arial", 12))
        self.crypto_combobox.pack(pady=5)

        # Load and Train Data Button
        self.load_train_button = tk.Button(root, text="Load and Train Data", command=self.load_and_train, font=("Arial", 12), width=25)
        self.load_train_button.pack(pady=10)

        # Download and Save Data Button
        self.download_save_button = tk.Button(root, text="Download and Save Data", command=self.download_and_save, font=("Arial", 12), width=25)
        self.download_save_button.pack(pady=10)

        # Predict Button
        self.predict_button = tk.Button(root, text="Predict", command=self.predict, font=("Arial", 12), width=25, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        # Predict Next Day Rate Button
        self.live_predict_button = tk.Button(root, text="Predict Next Day's Rate", command=self.predict_next_day, font=("Arial", 12), width=25, state=tk.DISABLED)
        self.live_predict_button.pack(pady=10)

        # Show Data Button
        self.show_data_button = tk.Button(root, text="Show Data", command=self.show_data, font=("Arial", 12), width=25, state=tk.DISABLED)
        self.show_data_button.pack(pady=10)

        # Fields for Prediction
        self.fields = {
            'Open': tk.Entry(root, font=("Arial", 12)),
            'High': tk.Entry(root, font=("Arial", 12)),
            'Low': tk.Entry(root, font=("Arial", 12)),
            'Volume': tk.Entry(root, font=("Arial", 12))
        }

        

        # Prediction Output Text Box
        self.output_text = tk.Text(root, height=4, width=50, font=("Arial", 12))
        self.output_text.pack(pady=10)
        self.output_text.insert(tk.END, "Prediction results will be displayed here...\n")
        self.output_text.config(state=tk.DISABLED)

        # Canvas for Matplotlib Plot
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(pady=10)

    def load_model(self):
        # Load the model and scaler if they exist
        if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
            self.model = joblib.load('models/model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.predict_button.config(state=tk.NORMAL)
            self.live_predict_button.config(state=tk.NORMAL)
            print("Model and scaler loaded successfully.")

    def load_and_train(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            # Load data
            data = pd.read_csv(file_path)
            features = ['Open', 'High', 'Low', 'Volume']
            X = data[features]
            y = data['Close']

            # Normalize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Split data and train model
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.model = RandomForestRegressor()
            self.model.fit(X_train, y_train)

            # Save the model and scaler
            joblib.dump(self.model, 'models/model.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')

            messagebox.showinfo("Info", "Model trained and saved successfully!")
            self.predict_button.config(state=tk.NORMAL)
            self.live_predict_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading and training the data: {e}")

    def download_and_save(self):
        crypto_symbol = self.selected_crypto.get()

        try:
            # Fetch historical data from Yahoo Finance
            crypto_data = yf.download(crypto_symbol, period="1y", interval="1d")  # Fetch last 1 year data
            if crypto_data.empty:
                raise ValueError("No data available for the selected cryptocurrency.")

            # Save data to CSV
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not save_path:
                return

            crypto_data.to_csv(save_path)
            self.data = crypto_data
            self.show_data_button.config(state=tk.NORMAL)
            messagebox.showinfo("Info", f"Data for {crypto_symbol} saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while downloading and saving data: {e}")

    def show_data(self):
        if self.data is not None:
            try:
                # Clear previous plots
                for widget in self.canvas_frame.winfo_children():
                    widget.destroy()

                # Plot data
                fig, ax = plt.subplots(figsize=(8, 5))
                self.data[['Open', 'High', 'Low', 'Close']].plot(kind='bar', ax=ax)
                ax.set_title('Cryptocurrency Price Data')
                ax.set_ylabel('Price')
                ax.set_xlabel('Date')

                # Add plot to Tkinter canvas
                canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while displaying data: {e}")

    def predict(self):
        if not self.model:
            messagebox.showerror("Error", "Model is not trained yet.")
            return

        try:
            # Extract user inputs
            features = {
                'Open': float(self.fields['Open'].get()),
                'High': float(self.fields['High'].get()),
                'Low': float(self.fields['Low'].get()),
                'Volume': float(self.fields['Volume'].get())
            }

            # Prepare features for prediction
            input_data = [[
                features['Open'],
                features['High'],
                features['Low'],
                features['Volume']
            ]]

            # Scale the input data
            input_data_scaled = self.scaler.transform(input_data)

            # Make prediction
            prediction = self.model.predict(input_data_scaled)
            predicted_price = prediction[0]

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Predicted Closing Price: ${predicted_price:.2f}\n")
            self.output_text.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

    def predict_next_day(self):
        if not self.model:
            messagebox.showerror("Error", "Model is not trained yet.")
            return

        crypto_symbol = self.selected_crypto.get()

        try:
            # Fetch historical data from Yahoo Finance
            crypto_data = yf.download(crypto_symbol, period="3mo", interval="1d")  # Fetch last 3 months data
            if len(crypto_data) < 30:  # Ensure at least 30 days of data
                raise ValueError("Not enough data to make a prediction. Please download more data.")

            # Use the last available day's data to predict the next day's rate
            latest_data = crypto_data.iloc[-1]
            features = {
                'Open': latest_data['Open'],
                'High': latest_data['High'],
                'Low': latest_data['Low'],
                'Volume': latest_data['Volume']
            }

            # Prepare features for prediction
            input_data = [[
                features['Open'],
                features['High'],
                features['Low'],
                features['Volume']
            ]]

            # Scale the input data
            input_data_scaled = self.scaler.transform(input_data)

            # Make prediction
            prediction = self.model.predict(input_data_scaled)
            predicted_price = prediction[0]

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Predicted Next Day's Closing Price: ${predicted_price:.2f}\n")
            self.output_text.config(state=tk.DISABLED)
        except ValueError as ve:
            messagebox.showerror("Error", f"Value error: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while fetching live data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CryptoApp(root)
    root.mainloop()
