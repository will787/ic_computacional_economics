# %%
from bcb import sgs
import pandas as pd
import datetime as dt

today = dt.datetime.now().strftime("%Y-%m-%d")

# pegamos a ultima data da selic, pegamos a variavel do momentum selic atual
selic = sgs.get(({'selic': 432}), start = today, end = today)
