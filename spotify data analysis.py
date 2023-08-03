#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings 
filterwarnings ('ignore')


# In[2]:


df_track = pd.read_csv("C:/Users/barsh/Downloads/archive (11)/tracks.csv")


# In[3]:


df_track.head(
)


# In[4]:


pd.isnull(df_track).sum()


# In[5]:


df_track.info()


# In[6]:


sorted_df = df_track.sort_values('popularity',ascending = True).head(10)
sorted_df


# In[7]:


df_track.describe().transpose()


# In[8]:


most_popular = df_track.query('popularity>90',inplace = False ).sort_values('popularity',ascending= False)
most_popular[:10]


# In[9]:


df_track.set_index("release_date" , inplace = True)
df_track.index = pd.to_datetime(df_track.index)
df_track.head()


# In[10]:


df_track[["artists"]].iloc[18]


# In[11]:


df_track["duration"]=df_track["duration_ms"].apply(lambda x: round(x/1000))
df_track.drop("duration_ms" ,inplace = True ,axis = 1)


# In[12]:


df_track.duration.head()


# In[13]:


corr_df=df_track.drop(["key","mode","explicit"],axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap=sns.heatmap(corr_df,annot=True, fmt=".1g", vmin=-1, vmax=1, center=0, cmap="inferno",linewidths=1 ,linecolor="Black")
heatmap.set_title("correlation heatmap between variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)


# In[14]:


sample_df = df_track.sample(int(0.004*len(df_track)))


# In[15]:


print(len(sample_df))


# In[16]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y="loudness", x="energy", color = "#f26b15").set(title = "loudness vs energy correlation")


# In[17]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y="popularity", x="acousticness", color = "#fdc48f").set(title = "popularity x acoustickness")


# In[18]:


df_track['dates']=df_track.index.get_level_values('release_date')
df_track.dates=pd.to_datetime(df_track.dates)
years=df_track.dates.dt.year


# In[19]:


df_track.head()


# In[20]:


pip install --user seaborn==0.11.0


# In[21]:


sns.displot(years,discrete=True,aspect=2,height=5,kind="hist").set(title="Number of songs per year")


# In[26]:


total_dr = df_track.duration
fig_dims = (18,7)
fig,ax = plt.subplots(figsize = fig_dims)
fig = sns.barplot(x= years,y=total_dr,ax=ax,   = False).set(title="year vs duration")
plt.xticks(rotation=90)


# In[30]:


total_dr=df_track.duration
sns.set_style("whitegrid")
fig_dims=(10,5)
fig,ax=plt.subplots(figsize=fig_dims)
fig=sns.lineplot(x=years,y=total_dr,ax=ax).set(title="year vs duration")
plt.xticks(rotation=60)


# In[33]:


df_genre=pd.read_csv("C:/Users/barsh/Downloads/spotifytrack/SpotifyFeatures.csv")


# In[35]:


df_genre.head()


# In[41]:


plt.title("Duration of the songs in diffrent genres")
sns.color_palette("rocket",as_cmap=True)
sns.barplot(y='genre',x='duration_ms',data=df_genre)
plt.xlabel("duration in milli seconds")
plt.ylabel("genres")


# In[42]:


sns.set_style(style="darkgrid")
plt.figure(figsize=(10,5))
famous = df_genre.sort_values("popularity",ascending =False).head(10)
sns.barplot(y='genre',x='popularity',data=famous).set(title="top 5 genres by popularity")


# In[ ]:




