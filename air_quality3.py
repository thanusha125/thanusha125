import pandas as pd
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r"C:\Users\thanu\OneDrive\Desktop\2 VITC S5\G Big data analytics\DA2\Cleaned_AirQualityUCI3.csv")


df_numeric = df.select_dtypes(include=[np.number])


scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

som_x, som_y = 10, 10  # 10x10 grid of neurons
input_len = df_normalized.shape[1]  # Number of features in the dataset

som = MiniSom(x=som_x, y=som_y, input_len=input_len, sigma=1.0, learning_rate=0.5)


som.random_weights_init(df_normalized.values)
som.train_random(df_normalized.values, num_iteration=100)  # 100 iterations for training


plt.figure(figsize=(12, 12))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plot the U-Matrix
plt.colorbar(label='Distance from Neurons')

for i, x in enumerate(df_normalized.values):
    w = som.winner(x)  # Get the winning node for the input
    # Limit the number of markers to avoid cluttering
    if i % 10 == 0:  # Display every 10th marker
        plt.text(w[0] + 0.5, w[1] + 0.5, str(i), fontsize=14, color='black', ha='center', va='center',
                 fontweight='bold')  # Removed path_effects for simplicity

plt.title("Self-Organizing Map (SOM) for Air Quality Data", fontsize=24, fontweight='bold')


plt.savefig(r"C:\Users\thanu\OneDrive\Desktop\2 VITC S5\G Big data analytics\DA2\som_output.png", bbox_inches='tight', dpi=300)

plt.show()
