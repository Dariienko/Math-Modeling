import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_a = {'x': [1.0, 2.3, 3.7, 4.2, 6.1, 7.0],
          'y': [3.6, 3.0, 3.2, 5.1, 5.3, 6.8]}
df_a = pd.DataFrame(data_a)

data_b = {'x': [29.1, 48.2, 72.7, 92.0, 118, 140, 165, 199],
          'y': [0.0493, 0.0821, 0.123, 0.154, 0.197, 0.234, 0.274, 0.328]}
df_b = pd.DataFrame(data_b)

data_c = {'x': [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
          'y': [4.32, 4.83, 5.27, 5.74, 6.26, 6.79, 7.23]}
df_c = pd.DataFrame(data_c)

def y (a, b, x):
    y = a * x + b 
    return y
    
def task_answer(data,m):
    data['xy'] = data.x * data.y
    data['x^2'] = data.x**2
    
    a = (m * data.xy.sum() - (data.x.sum() * data.y.sum()))/(m * data['x^2'].sum() - data.x.sum()**2)
    b =  (data['x^2'].sum() * data.y.sum() - (data.xy.sum() * data.x.sum()))/(m *data['x^2'].sum()- data.x.sum()**2)

    d_sum = 0
    d_max = 0
    y_pred = []
    d_i = []
    
    for _, row in data.iterrows():
        predicted_y = y(a, b, row.x)
        y_pred.append(predicted_y)
        d_i.append(np.abs(predicted_y - row.y)) 
        d_sum += (predicted_y - row.y)**2
        d_max = max(d_max, np.abs(predicted_y - row.y))
    
    data["d_i"] = d_i
    data["y^"] = y_pred
    
    D = np.sqrt(d_sum/data.y.count())
    
    print(f"y = {a:.5f} * x + {b:.5f}")
    print(f"D = {D:.8f} <= c_max < d_max = {d_max:.4f}")
    print("-" * 50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.x, data["y^"], color ='blue', marker="o", label="predict line", zorder = 1)
    plt.scatter(data.x, data.y, color ='orange',marker="o" , label="value", zorder = 2)
    for _, row in data.iterrows():
        if row["d_i"] == d_max:
             plt.text(row.x, row.y,f"  d_max = {d_max:.4f}", ha='left', color="red", zorder = 3)
    for _, row in data.iterrows():
         plt.plot([row.x, row.x],[row.y, row["y^"]], color='red', ls='--', zorder= 0)
   
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()  
    
task_answer(df_a,df_a.x.count())
task_answer(df_b,df_b.x.count())
task_answer(df_c,df_c.x.count())