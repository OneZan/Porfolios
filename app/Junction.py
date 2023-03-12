#!/usr/bin/env python
# coding: utf-8

# ## Junction Risk + Porfolio

# In[2]:


import dash
#import dash_core_components as dcc
#import dash_html_components as html
import dash.dcc as dcc
import dash.html as html
from dash.dependencies import Input,Output,State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_daq as daq
from pickle import load
import cvxopt as opt
from cvxopt import blas, solvers
import sklearn

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage,risk_matrix

from pypfopt import plotting
import copy


# In[ ]:





# In[3]:


assets = pd.read_csv('../assets.csv',index_col=0)
filename = 'risk_model.p'
loaded_model = load(open(filename, 'rb'))


# In[13]:


edad = int(input("Edad:"))


# In[14]:


nivel_educativo = int(input("Seleccione su nivel educativo:\n0- Sin educación\n1- Ed. diferencial\n2- Ed. Básica\n3- Ed. Media\n4- CFT o IP\n5- Universitaria\n6- Postgrado\n"))


# In[12]:


nivel_educativo


# In[18]:


estado_civil = int(input("Seleccione su estado civil:\n1- Casado\n2- Conviviente\n3- Anulado\n4- Separado\n5 Viudo\n6- Soltero\n7- Divorciado\n"))


# In[20]:


ocup = int(input("Seleccione su estado de oupación actual:\n0- Desocupado\n1- Ocupado\n"))


# In[21]:


tr_numh = int(input("Seleccione el número de habitantes en su hogar:\n1- 1 o 2 personas\n2- 3 o 4 personas\n3- 5 o 6 personas\n5 más de 6 personas\n"))


act_tot = int(input("Ingrese un estimado de tu total de activos en pesos chilenos:\n"))
estrato = int(input("Ingrese su estrato:\n1- \n6-\n9-"))
rci_dt  = float(input("Ingrese rci ( entre 0 y 1):\n"))


# In[22]:


X_imput = {"edad_ent" : edad,"neduc_ent" : nivel_educativo, "est_civil_ent": estado_civil, "ocup_ent":ocup, "estrato":estrato, "tr_numh": tr_numh,"act_toth":act_tot,"rci_dt":rci_dt}

X_imput = {"edad_ent" : 21,"neduc_ent" : 4.0, "est_civil_ent": 6.0, "ocup_ent":1.0, "estrato":9.0, "tr_numh": 2.0,"act_toth":9.090000e+08,"rci_dt":0.34}
# In[23]:


X_imput = pd.DataFrame(np.array(list(X_imput.values())).reshape(1,-1),columns = list(X_imput.keys()))


# In[24]:


risk = loaded_model.predict(X_imput)


# In[26]:


risk


# In[32]:


def get_asset_allocation(risk,stock_ticker = assets.columns,initial_inv = 100):
    #ipdb.set_trace()
    assets_selected = assets.loc[:,stock_ticker]
    #return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    mu = mean_historical_return(assets,compounding = False)
    #S = CovarianceShrinkage(assets).ledoit_wolf()
    S = risk_matrix(assets)#.sample_cov()
    ef = EfficientFrontier(mu, S)

    #risk = np.sqrt(risk)
    #print(risk[0])
    weights = ef.efficient_risk(risk[0])
    ef.portfolio_performance(verbose=True)

    returns_final= np.dot(assets.loc[:, weights.keys()], np.array(list(weights.values())))
    #returns_sum = np.cumsum(returns_final)
    #print(returns_sum)
    returns_sum_pd = pd.DataFrame(returns_final, index = assets.index )
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + initial_inv   
    return weights,returns_sum_pd


# In[34]:


w,r = get_asset_allocation(risk)
print(w)


# In[ ]:




