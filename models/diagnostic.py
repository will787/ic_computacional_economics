# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_history.csv")
df.head()
# %%

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Count_Def_D'], label='Count Def D')
plt.plot(df['Time'], df['Count_Def_U'], label='Count Def U')
plt.xlabel('Time')
plt.ylabel('Count Def')
plt.title('Count Def over Time')
plt.legend()
plt.grid()
plt.show()
# %%
