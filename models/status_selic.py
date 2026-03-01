# %%
from bcb import sgs
import pandas as pd
import datetime as dt

date = dt.today().strftime('%Y-%m-%d')
selic = sgs.get({'selic': 432}, start='2000-01-01')
# %%
