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

plt.hist(df['Count_Def_Z'], bins=20, alpha=0.7, label='Count Def Z')
plt.hist(df['Count_Def_U'], bins=20, alpha=0.7, label='Count Def U')
plt.hist(df['Count_Def_D'], bins=20, alpha=0.7, label='Count Def D')

plt.xlabel('Count Def')
plt.ylabel('Frequency')
plt.title('Distribution of Count Def')
plt.legend()
plt.grid()
plt.show()


# %%
