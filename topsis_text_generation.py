import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def topsis(data, weights, impacts):
    # Normalize the dataset
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    
    # Multiply by weights
    weighted_data = norm_data * weights
    
    # Determine ideal and negative-ideal solutions
    ideal_best = np.max(weighted_data, axis=0) * impacts + np.min(weighted_data, axis=0) * (1 - impacts)
    ideal_worst = np.min(weighted_data, axis=0) * impacts + np.max(weighted_data, axis=0) * (1 - impacts)
    
    # Compute separation measures
    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))
    
    # Compute TOPSIS scores
    scores = dist_worst / (dist_best + dist_worst)
    
    return scores

# Example dataset with models and their performance metrics
data = pd.DataFrame({
    'Model': ['GPT-3', 'T5', 'GPT-2', 'XLNet', 'BART'],
    'Accuracy': [0.9, 0.85, 0.8, 0.75, 0.78],
    'Fluency': [0.88, 0.83, 0.79, 0.74, 0.76],
    'Coherence': [0.87, 0.82, 0.78, 0.73, 0.75]
})

# Define weights and impacts
weights = np.array([0.4, 0.3, 0.3])  # Example weights
impacts = np.array([1, 1, 1])  # 1 for benefit criteria

# Apply TOPSIS
scores = topsis(data.iloc[:, 1:].values, weights, impacts)
data['TOPSIS Score'] = scores

data.sort_values(by='TOPSIS Score', ascending=False, inplace=True)

# Print ranked models
print(data[['Model', 'TOPSIS Score']])

# Save results to CSV
data.to_csv('topsis_text_generation_results.csv', index=False)

plt.figure(figsize=(10,6))
plt.barh(data['Model'], data['TOPSIS Score'], color='skyblue')
plt.xlabel('TOPSIS Score')
plt.ylabel('Model')
plt.title('Model Ranking Based on TOPSIS Score')
plt.gca().invert_yaxis()
plt.savefig('topsis_score_plot.png')
plt.show()