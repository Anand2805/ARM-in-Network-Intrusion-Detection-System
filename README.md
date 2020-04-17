<p style="font-size:36px;text-align:center"> <b>Association Rule feature Mining (ARM) in Network Intrusion Detection System</b> </p>

# 1. Business Problem

## 1.1. Description
<p>Source: <a>https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets</a></p>
<p> Data: CISCO Networking Dataset - The UNSW-NB15 Dataset </p>
<p> Download UNSW-NB15 - csv file.</p> 

<h6> Problem statement : </h6>
<p> Classify the given network is intrusion or normal based on evidence from The raw network packets of the UNSW-NB 15 dataset. it was created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). </p>

# 2. Machine Learning Problem Formulation

## 2.1. Data

### 2.1.1. Data Overview

- Source: https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys
- We have multiple data files: download the UNSW-NB15 - csv file this file contains a following structure

        a part of training and testing set - folder contains train and test data csv files
        NUSW-NB15_features.csv - Feature description
        NUSW-NB15_GT.csv
        The UNSW-NB15 description.pdf
        UNSW-NB15_1.csv
        UNSW-NB15_2.csv
        UNSW-NB15_3.csv
        UNSW-NB15_4.csv
        UNSW-NB15_LIST_EVENTS.csv
        

- These features are described in **UNSW-NB15_features.csv** file.

- The total number of records is two million and 540,044 which are stored in the four CSV files, namely, UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv and UNSW-NB15_4.csv.

- The ground truth table is named **UNSW-NB15_GT.csv** and the list of event file is called UNSW-NB15_LIST_EVENTS.csv.

- A partition from this dataset is configured as a training set and testing set, namely, **UNSW_NB15_training-set.csv** and **UNSW_NB15_testing-set.csv** respectively.

- The number of records in the training set is 82,332 records and the testing set is 175,341 records from the different types, attack and normal.Figure 1 and 2 show the testbed configuration dataset and the method of the feature creation of the UNSW-NB15, respectively. 

- <p> 
    Data file's information:
    <ul>
        <li> <p>both train and test files contains 45 columns</p> <p>['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
       'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
       'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
       'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
       'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
       'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
            'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']</p> </li>     
        <li>
        Categorical (4) columns
        </li>
        <li>
        Numerical (41) columns
        </li>
    </ul>
</p>

## 2.2. Mapping the real-world problem to an ML problem

### 2.2.1. Type of Machine Learning Problem

<p>
    
            There are two different class normal or attack. Find the network is normal or intrusion. => Binary class classification problem
   
      
    
</p>

### 2.2.2. Performance Metric

metric used to identify the performance of the model

Metric(s): 
* AUC and f1-score 
* Confusion matrix
* FAR - false alarm rate should be as minimum as possible


### 2.2.3. Machine Learing Objectives and Constraints

<p>   
    
            Objective: Predict the probability of each data-point whether the network is normal or attack.
    
    
</p>


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
```


```python
train_data =pd.read_csv("data/UNSW_NB15_training-set.csv")
print(train_data.shape)
train_data.head()
```

    (82332, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000011</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>496</td>
      <td>0</td>
      <td>90909.0902</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000008</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1762</td>
      <td>0</td>
      <td>125000.0003</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000005</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>200000.0051</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000006</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>900</td>
      <td>0</td>
      <td>166666.6608</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000010</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>2126</td>
      <td>0</td>
      <td>100000.0025</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>



# 3. Exploratory Data Analysis 

## 3.1 visualizing class label


```python
total = len(train_data)*1.
ax=sns.countplot(x="label", data=train_data)
for p in ax.patches:
    print(p)
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.3, p.get_height()+5))

#put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
ax.yaxis.set_ticks(np.linspace(0, total, 11))
print(ax.yaxis.get_majorticklocs())
#adjust the ticklabel to the desired format, without changing the position of the ticks. 
ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
plt.savefig('class_label.png')
plt.show()
```

    Rectangle(xy=(-0.4, 0), width=0.8, height=37000, angle=0)
    Rectangle(xy=(0.6, 0), width=0.8, height=45332, angle=0)
    [    0.   8233.2 16466.4 24699.6 32932.8 41166.  49399.2 57632.4 65865.6
     74098.8 82332. ]
    


![png](output_18_1.png)


Above plot show that the dataset is not an imbalanced dataset

## 3.2 visualizing categorical data


```python
cat_feature = train_data.select_dtypes(include=['category', object]).columns
cat_feature
```




    Index(['proto', 'service', 'state', 'attack_cat'], dtype='object')




```python
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(hspace = 0.4)
for col, subplot in zip(cat_feature, ax.flatten()):
    sns.countplot(train_data[col], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.savefig('cate_f.png')
plt.show()
```


![png](output_22_0.png)


From the categorical data we can see the data imbalance. Also “proto” category has more than 200 categories. Other columns have less than or equal 13 columns

## 3.2 visualizing numerical data and its distribution


```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_features = train_data.drop(['id','label'], axis=1).select_dtypes(include=numerics).columns
num_features
```




    Index(['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
           'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
           'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
           'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
           'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
           'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
           'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'],
          dtype='object')




```python
fig, ax = plt.subplots(8, 5, figsize=(30, 20))
plt.subplots_adjust(hspace = 0.4)
for col, splot in zip(num_features, ax.flatten()):
    sns.distplot(train_data[col], ax=splot)
plt.savefig('num_f.png')
plt.show()
```


![png](output_26_0.png)


from the distribution plot each feature have one value which is occurring more number of times than other values
As we can see from numerical data distribution there few features which are highly correlated with each other.

## 3.3 Correlation of data


```python
df_corr = train_data.corr()
```


```python
plt.figure(figsize=(30,20))
sns.heatmap(df_corr, annot=True, cmap=plt.cm.viridis)
plt.savefig('corr_mat.png')
plt.show()
```


![png](output_30_0.png)


- From above visualization we can clearly see that there are few columns which are having high correlation with one another.
- we will find correlation and distribution of those columns and eliminate the necessary one


```python
high_corr_var=np.where(df_corr>0.95)
```


```python
high_corr_var
```




    (array([ 0,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5,  6,  7,  8,
             9, 10, 11, 11, 11, 12, 12, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30, 30, 31, 31, 32, 33,
            34, 34, 35, 35, 36, 37, 38, 38, 39, 40], dtype=int64),
     array([ 0,  1,  2,  4, 11,  3,  5, 12,  2,  4, 11,  3,  5, 12,  6,  7,  8,
             9, 10,  2,  4, 11,  3,  5, 12, 13, 14, 15, 16, 17, 20, 18, 19, 17,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 38, 29, 30, 31, 30, 31, 32, 33,
            34, 35, 34, 35, 36, 37, 28, 38, 39, 40], dtype=int64))




```python
#ref: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
high_corr_var=[(df_corr.columns[x],df_corr.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
```


```python
high_corr_var
```




    [('spkts', 'sbytes'),
     ('spkts', 'sloss'),
     ('dpkts', 'dbytes'),
     ('dpkts', 'dloss'),
     ('sbytes', 'sloss'),
     ('dbytes', 'dloss'),
     ('swin', 'dwin'),
     ('ct_srv_src', 'ct_srv_dst'),
     ('ct_dst_ltm', 'ct_src_dport_ltm'),
     ('is_ftp_login', 'ct_ftp_cmd')]



## 3.4 Feature Description


```python
data_features =pd.read_csv("data/NUSW-NB15_features.csv", engine='python')
print(data_features.shape)
```

    (49, 4)
    


```python
data_features.head(49)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>srcip</td>
      <td>nominal</td>
      <td>Source IP address</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>sport</td>
      <td>integer</td>
      <td>Source port number</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>dstip</td>
      <td>nominal</td>
      <td>Destination IP address</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>dsport</td>
      <td>integer</td>
      <td>Destination port number</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>proto</td>
      <td>nominal</td>
      <td>Transaction protocol</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>state</td>
      <td>nominal</td>
      <td>Indicates to the state and its dependent proto...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>dur</td>
      <td>Float</td>
      <td>Record total duration</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>sbytes</td>
      <td>Integer</td>
      <td>Source to destination transaction bytes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>dbytes</td>
      <td>Integer</td>
      <td>Destination to source transaction bytes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>sttl</td>
      <td>Integer</td>
      <td>Source to destination time to live value</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>dttl</td>
      <td>Integer</td>
      <td>Destination to source time to live value</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>sloss</td>
      <td>Integer</td>
      <td>Source packets retransmitted or dropped</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>dloss</td>
      <td>Integer</td>
      <td>Destination packets retransmitted or dropped</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>service</td>
      <td>nominal</td>
      <td>http, ftp, smtp, ssh, dns, ftp-data ,irc  and ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Sload</td>
      <td>Float</td>
      <td>Source bits per second</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Dload</td>
      <td>Float</td>
      <td>Destination bits per second</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Spkts</td>
      <td>integer</td>
      <td>Source to destination packet count</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Dpkts</td>
      <td>integer</td>
      <td>Destination to source packet count</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>swin</td>
      <td>integer</td>
      <td>Source TCP window advertisement value</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>dwin</td>
      <td>integer</td>
      <td>Destination TCP window advertisement value</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>stcpb</td>
      <td>integer</td>
      <td>Source TCP base sequence number</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>dtcpb</td>
      <td>integer</td>
      <td>Destination TCP base sequence number</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>smeansz</td>
      <td>integer</td>
      <td>Mean of the ?ow packet size transmitted by the...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>dmeansz</td>
      <td>integer</td>
      <td>Mean of the ?ow packet size transmitted by the...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>trans_depth</td>
      <td>integer</td>
      <td>Represents the pipelined depth into the connec...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>res_bdy_len</td>
      <td>integer</td>
      <td>Actual uncompressed content size of the data t...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>Sjit</td>
      <td>Float</td>
      <td>Source jitter (mSec)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>Djit</td>
      <td>Float</td>
      <td>Destination jitter (mSec)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>Stime</td>
      <td>Timestamp</td>
      <td>record start time</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>Ltime</td>
      <td>Timestamp</td>
      <td>record last time</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>Sintpkt</td>
      <td>Float</td>
      <td>Source interpacket arrival time (mSec)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>Dintpkt</td>
      <td>Float</td>
      <td>Destination interpacket arrival time (mSec)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>tcprtt</td>
      <td>Float</td>
      <td>TCP connection setup round-trip time, the sum ...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>synack</td>
      <td>Float</td>
      <td>TCP connection setup time, the time between th...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>ackdat</td>
      <td>Float</td>
      <td>TCP connection setup time, the time between th...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>is_sm_ips_ports</td>
      <td>Binary</td>
      <td>If source (1) and destination (3)IP addresses ...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>ct_state_ttl</td>
      <td>Integer</td>
      <td>No. for each state (6) according to specific r...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>ct_flw_http_mthd</td>
      <td>Integer</td>
      <td>No. of flows that has methods such as Get and ...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>is_ftp_login</td>
      <td>Binary</td>
      <td>If the ftp session is accessed by user and pas...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>ct_ftp_cmd</td>
      <td>integer</td>
      <td>No of flows that has a command in ftp session.</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>ct_srv_src</td>
      <td>integer</td>
      <td>No. of connections that contain the same servi...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>ct_srv_dst</td>
      <td>integer</td>
      <td>No. of connections that contain the same servi...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>ct_dst_ltm</td>
      <td>integer</td>
      <td>No. of connections of the same destination add...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44</td>
      <td>ct_src_ ltm</td>
      <td>integer</td>
      <td>No. of connections of the same source address ...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45</td>
      <td>ct_src_dport_ltm</td>
      <td>integer</td>
      <td>No of connections of the same source address (...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>ct_dst_sport_ltm</td>
      <td>integer</td>
      <td>No of connections of the same destination addr...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>ct_dst_src_ltm</td>
      <td>integer</td>
      <td>No of connections of the same source (1) and t...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>attack_cat</td>
      <td>nominal</td>
      <td>The name of each attack category. In this data...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>Label</td>
      <td>binary</td>
      <td>0 for normal and 1 for attack records</td>
    </tr>
  </tbody>
</table>
</div>



### 3.4.1 dur


```python
data_features[6:7]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>dur</td>
      <td>Float</td>
      <td>Record total duration</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(train_data[train_data['label']==0]['dur'], label='normal', hist=False, rug=True)
sns.distplot(train_data[train_data['label']==1]['dur'], label='anomaly', hist=False, rug=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x20677228a90>




![png](output_41_1.png)


dur is the total rocrd duration for both anomaly and normal.

### 3.4.2 sbytes - dbytes


```python
data_features[7:9]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>sbytes</td>
      <td>Integer</td>
      <td>Source to destination transaction bytes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>dbytes</td>
      <td>Integer</td>
      <td>Destination to source transaction bytes</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(train_data[train_data['label']==0]['sbytes'].apply(np.log1p), label='normal', hist=False, rug=True)
sns.distplot(train_data[train_data['label']==1]['sbytes'].apply(np.log1p), label='anomaly', hist=False, rug=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x17db1ccd5f8>




![png](output_45_1.png)


- applied logarithm of x log(1+x) for some of the features to visualize properly 

### 3.4.3 sloss - dloss


```python
data_features[11:13]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>sloss</td>
      <td>Integer</td>
      <td>Source packets retransmitted or dropped</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>dloss</td>
      <td>Integer</td>
      <td>Destination packets retransmitted or dropped</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(train_data[train_data['label']==0]['sloss'].apply(np.log1p), label='normal', hist=False, rug=True)
sns.distplot(train_data[train_data['label']==1]['sloss'].apply(np.log1p), label='anomaly', hist=False, rug=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x17da4084080>




![png](output_49_1.png)


- sloss and dloss have same disruption as we can see in overall numerical distribution plot.
-

### 3.4.4 spkts - dpkts


```python
data_features[16:18]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Spkts</td>
      <td>integer</td>
      <td>Source to destination packet count</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Dpkts</td>
      <td>integer</td>
      <td>Destination to source packet count</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(train_data[train_data['label']==0]['spkts'].apply(np.log1p), label='normal', hist=False, rug=True)
sns.distplot(train_data[train_data['label']==1]['spkts'].apply(np.log1p), label='anomaly', hist=False, rug=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x2059acbcc18>




![png](output_53_1.png)


- these sbytes and dbytes have same distribution. anomaly we can see it peaks to 0.8 and above. normal is below 0.4


```python
print(train_data[['sbytes','sloss','spkts']].corr())
print(train_data[['dbytes','dloss','dpkts']].corr())
```

              sbytes     sloss     spkts
    sbytes  1.000000  0.995027  0.965750
    sloss   0.995027  1.000000  0.973644
    spkts   0.965750  0.973644  1.000000
              dbytes     dloss     dpkts
    dbytes  1.000000  0.997109  0.976419
    dloss   0.997109  1.000000  0.981506
    dpkts   0.976419  0.981506  1.000000
    


```python
sns.heatmap(train_data[['sbytes','sloss','spkts']].corr(), annot=True, cmap=plt.cm.viridis)
plt.show()
sns.heatmap(train_data[['dbytes','dloss','dpkts']].corr(), annot=True, cmap=plt.cm.viridis)
plt.show()
```


![png](output_56_0.png)



![png](output_56_1.png)


- so we can drop column sbyte and dbytes from above representation.
- From this visualization we can see that both have high correlation and same distribution with other columns

### 3.4.5 sttl - dttl


```python
data_features[9:11]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>sttl</td>
      <td>Integer</td>
      <td>Source to destination time to live value</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>dttl</td>
      <td>Integer</td>
      <td>Destination to source time to live value</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(train_data[train_data['label']==0]['sttl'], label='normal', hist=False, rug=True)
sns.distplot(train_data[train_data['label']==1]['sttl'], label='anomaly', hist=False, rug=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x2058e5659b0>




![png](output_60_1.png)


- fewer amount of data have anomaly of higher rate. >0.2 are anomaly


```python
sns.distplot(train_data[train_data['label']==0]['dttl'], label='normal', hist=False, rug=True)
sns.distplot(train_data[train_data['label']==1]['dttl'], label='anomaly', hist=False, rug=True)
plt.legend()
```




    <matplotlib.legend.Legend at 0x21feb08c6d8>




![png](output_62_1.png)


- sttl and dttl have different distribution as we can see in overall numerical distribution plot. cant ignore column

### 3.4.6 swin - dwin


```python
data_features[18:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>swin</td>
      <td>integer</td>
      <td>Source TCP window advertisement value</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>dwin</td>
      <td>integer</td>
      <td>Destination TCP window advertisement value</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))
for i, col in enumerate(['swin', 'dwin']):
    plt.subplot(1,2,i+1)
    sns.distplot(train_data[train_data['label']==0][col], label='normal', hist=False, rug=True)
    sns.distplot(train_data[train_data['label']==1][col], label='anomaly', hist=False, rug=True)
    plt.legend()
plt.show()
```


![png](output_66_0.png)


- From above distribution plot we can distinguish between normal and anomaly.
- window rate of -50 to 50 less than 0.0075 is normal and 200 to 300 less than 0.0075 is anomaly

### 3.4.7 ct_dst_src_ltm, ct_srv_src, ct_srv_dst


```python
data_features[40:42]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>ct_srv_src</td>
      <td>integer</td>
      <td>No. of connections that contain the same servi...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>ct_srv_dst</td>
      <td>integer</td>
      <td>No. of connections that contain the same servi...</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(data_features[40:42]['Description'])
```




    ['No. of connections that contain the same service (14) and source address (1) in 100 connections according to the last time (26).',
     'No. of connections that contain the same service (14) and destination address (3) in 100 connections according to the last time (26).']




```python
plt.figure(figsize=(10,6))
for i, col in enumerate(['ct_dst_src_ltm', 'ct_srv_src', 'ct_srv_dst']):
    plt.subplot(2,2,i+1)
    sns.distplot(train_data[train_data['label']==0][col], label='normal', hist=False, rug=True)
    sns.distplot(train_data[train_data['label']==1][col], label='anomaly', hist=False, rug=True)
    plt.legend()
plt.show()
```


![png](output_71_0.png)


- As we can see the distribution are similar


```python
sns.heatmap(train_data[['ct_dst_src_ltm','ct_srv_src','ct_srv_dst']].corr(), annot=True, cmap=plt.cm.viridis)
plt.show()
```


![png](output_73_0.png)


- we can drop column ct_srv_dst
- From this visualization we can see that three columns are having high correlation and same distribution

#### 3.4.8 'is_ftp_login', 'ct_ftp_cmd'


```python
data_features[38:40]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No.</th>
      <th>Name</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>is_ftp_login</td>
      <td>Binary</td>
      <td>If the ftp session is accessed by user and pas...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>ct_ftp_cmd</td>
      <td>integer</td>
      <td>No of flows that has a command in ftp session.</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(data_features[38:40]['Description'])
```




    ['If the ftp session is accessed by user and password then 1 else 0. ',
     'No of flows that has a command in ftp session.']




```python
plt.figure(figsize=(10,6))
for i, col in enumerate(['is_ftp_login', 'ct_ftp_cmd']):
    plt.subplot(1,2,i+1)
    sns.distplot(train_data[train_data['label']==0][col], label='normal', hist=False, rug=True)
    sns.distplot(train_data[train_data['label']==1][col], label='anomaly', hist=False, rug=True)
    plt.legend()
plt.show()
```


![png](output_78_0.png)


- As we can see the distribution are same and both the columns related to ftp session


```python
sns.heatmap(train_data[['is_ftp_login', 'ct_ftp_cmd']].corr(), annot=True, cmap=plt.cm.viridis)
plt.show()
```


![png](output_80_0.png)


- we can drop ct_ftp_cmd. 
- From this visualization we can see that both have high correlation and same distribution

## 3.5 EDA conclusion

#### we can drop the 5 columns mentioned from above analysis which are highly correlated
- sbyte and dbytes
- ct_srv_dst
- ct_ftp_cmd
- dwin


```python
train_data.drop(['sbytes', 'dbytes','ct_srv_dst', 'ct_ftp_cmd', 'dwin'], axis=1, inplace=True)
train_data.shape
```




    (82332, 40)



# 4. ARM - Feature selection

## 4.1 Association Rule Mining

<hr>

Association rule mining is a frequent pattern mining it is to determine how frequent an item is in total transaction.
here Frequent item is a set of items which satisfies the minimum threshold value. threshold or metric for ARM. it is the support and confidence.

from our data set consider sbytes & dbytes two items. below is the rule.

**sbytes** **=>** **dbytes** **[support=3%, confidence- 70%]**

The set of items sbytes & dbytes are called antecedent and consequent.

above state means that there is 3% that sbytes and dbytes are frequent together in total transaction. and there are 70% confidence level that sbytes and dbytes are occurred together.

### 4.1.1 Implementation Steps

|**Steps**|
-----------
<center>Set the minimum threshold values</center>
               <center>⇩</center>
<center>find all the subsets on the transaction using **apriori algorithm** having support of 30% or more.</center>
                <center>⇩</center>
<center>Find all the item sets or rule of these subsets from step 2 which are having a higher confidence than minimum confidence and maximum rule length of 2.</center>
                <center>⇩</center>
<center>get the columns from the rules using a set {} to eliminate the repeated columns.</center>
                <center>⇩</center>
<center>use these columns as feature for machine learning models.</center>

<hr>

## 4.2 Data preprocessing
- For better understanding of dataset 
- Identify the catogarical and numerical data and perform following encoding
    - Catogarical data (Label encoding)
    - Numerical data (StandardScalar)
- convert the data into numerical values so that it will input to machine learning models.

### 4.2.1 Catagorical Data


```python
cat_feature = train_data.select_dtypes(include=['category', object]).columns
cat_feature
```




    Index(['proto', 'service', 'state', 'attack_cat'], dtype='object')




```python
from sklearn.preprocessing import LabelEncoder
train_data[cat_feature] = train_data[cat_feature].apply(LabelEncoder().fit_transform)
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>rate</th>
      <th>sttl</th>
      <th>dttl</th>
      <th>...</th>
      <th>ct_dst_ltm</th>
      <th>ct_src_dport_ltm</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000011</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>90909.0902</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000008</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>125000.0003</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000005</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>200000.0051</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000006</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>166666.6608</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000010</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>100000.0025</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
train_data.shape
```




    (82332, 40)



### 4.2.2 Split data into equal parts

- To reduce the time complexity, the data set are divided into equal parts
     ### Number 𝑜𝑓 dataset = Number 𝑜𝑓 𝑟𝑒𝑐𝑜𝑟𝑑𝑠 / Number 𝑜𝑓 𝑎𝑡𝑡𝑟𝑖𝑏𝑢𝑡𝑒𝑠


```python
shuffled = train_data.sample(frac=1)
```


```python
shuffled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>rate</th>
      <th>sttl</th>
      <th>dttl</th>
      <th>...</th>
      <th>ct_dst_ltm</th>
      <th>ct_src_dport_ltm</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74784</th>
      <td>74785</td>
      <td>0.000004</td>
      <td>111</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>250000.000600</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31904</th>
      <td>31905</td>
      <td>0.001027</td>
      <td>117</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2921.129446</td>
      <td>31</td>
      <td>29</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11153</th>
      <td>11154</td>
      <td>0.623818</td>
      <td>111</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>8</td>
      <td>27.251539</td>
      <td>254</td>
      <td>252</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48077</th>
      <td>48078</td>
      <td>0.000012</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>83333.330390</td>
      <td>254</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>75900</th>
      <td>75901</td>
      <td>0.169748</td>
      <td>111</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>41.237601</td>
      <td>62</td>
      <td>252</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
data_42 = np.array_split(shuffled, 42)
```


```python
len(data_42)
```




    42



### 4.2.3 Find Mode of the attribute

- Lets compute the mode for each attribute. it is the most frequent values of the attribute.
- For each data set attribute frequent values are identified, and most frequent value is set to true and remaining are false. task will be performed in both numerical and categorical data like below. 
- It will accomplish the reliability of model output adopting to relevant attributes.
### Example: 
    1. Numeric

    X = {1,2,3,1,3,1} => {1} => {1,0,0,1,0,1}

    2. Categorical

    X = {'INT', 'FIN', 'REQ', 'ACC', ‘INT’, ‘REQ’, ‘INT’} => {‘INT’}  => {1, 0, 0, 0, 1, 0, 1}


```python
def create_arm_data(data):
    """ Create the binary mode for the data
    Find the most frequent data point in an attribute"""
    columns = data.columns
    for col in columns:
        #find mode of a attribute and make the model value 1 and others 0
        data[col] = np.where(data[col] == data[col].mode().values[0], 1, 0)
    return data
```

### 4.2.4 Create ARM rule based on Apriori Algorithm


```python
def create_arm_rule(result):
    """Create association rule for the given apriori data set """
    rules = association_rules(result, metric ="confidence", min_threshold = 1)
    # sort in order of confidence and lift
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
    # find the length of antecedents & consequents
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))
    rules_list_sorted = []
    # iterate each row to add the both antecedents & consequents in single column set
    for x,y in rules.iterrows():
        rules_list_sorted.append(sorted(set(y.antecedents) | set(y.consequents)))
    rules['rules_set_sorted'] = rules_list_sorted
    rules["rules_len"] = rules["rules_set_sorted"].apply(lambda x: len(x))
    #sort the set and make it list
    rules['rules_sorted'] = rules.rules_set_sorted.apply(lambda x: ','.join(map(str, x)))
    return rules
```

## 4.2.5 Feature selction

- After the creation of rules in each set the antecedents, consequents are combined to generate the frequent itemsets on each dataset
- Each data set frequent items are again combined to make the final attributes for all dataset.
- Maximum rule length sets to two to eliminate the attribute which are not a frequent item. Least one item/attribute comes together with other attribute
- considering the items/attributes which are present more than 30% of total dataset/transaction.
- min_support = 0.3 that is 30% of items present in total transaction.


```python
col_ruled_sets = []
i=1
for part in data_42:
    """find columns of frequent transaction for all the dataset"""
    print("===Started dataset "+ str(i) +"====")
    #drop id and label
    part = part.drop(['id', 'label'], axis=1)
    print(part.shape)
    #create the binary mode data
    part_binary = create_arm_data(part)
    #Use apriori algorithm to find the sunsets of frequent item
    result = apriori(part_binary, min_support=0.3, use_colnames=True, max_len=2)
    #Create the rule from subsets
    arm_rules = create_arm_rule(result)
    final_columns = arm_rules['rules_sorted'].unique()
    col_final = set()
    #add each frequent columns to set
    for row in final_columns:
        for col in row.split(","):
            col_final.add(col)
    print(col_final)
    col_ruled_sets.append(col_final)
    print("===Completed dataset "+ str(i) +"====")
    i+=1
```

    ===Started dataset 1====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 1====
    ===Started dataset 2====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 2====
    ===Started dataset 3====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 3====
    ===Started dataset 4====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 4====
    ===Started dataset 5====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 5====
    ===Started dataset 6====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 6====
    ===Started dataset 7====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 7====
    ===Started dataset 8====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 8====
    ===Started dataset 9====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 9====
    ===Started dataset 10====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 10====
    ===Started dataset 11====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 11====
    ===Started dataset 12====
    (1961, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 12====
    ===Started dataset 13====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 13====
    ===Started dataset 14====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 14====
    ===Started dataset 15====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 15====
    ===Started dataset 16====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'swin', 'dmean'}
    ===Completed dataset 16====
    ===Started dataset 17====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 17====
    ===Started dataset 18====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 18====
    ===Started dataset 19====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 19====
    ===Started dataset 20====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 20====
    ===Started dataset 21====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 21====
    ===Started dataset 22====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 22====
    ===Started dataset 23====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 23====
    ===Started dataset 24====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 24====
    ===Started dataset 25====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 25====
    ===Started dataset 26====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 26====
    ===Started dataset 27====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 27====
    ===Started dataset 28====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'sjit', 'dloss', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'swin', 'dmean'}
    ===Completed dataset 28====
    ===Started dataset 29====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 29====
    ===Started dataset 30====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 30====
    ===Started dataset 31====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 31====
    ===Started dataset 32====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 32====
    ===Started dataset 33====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 33====
    ===Started dataset 34====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 34====
    ===Started dataset 35====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 35====
    ===Started dataset 36====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 36====
    ===Started dataset 37====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 37====
    ===Started dataset 38====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 38====
    ===Started dataset 39====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 39====
    ===Started dataset 40====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 40====
    ===Started dataset 41====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'swin', 'dmean'}
    ===Completed dataset 41====
    ===Started dataset 42====
    (1960, 38)
    {'synack', 'is_sm_ips_ports', 'sloss', 'dloss', 'sjit', 'dtcpb', 'response_body_len', 'sttl', 'stcpb', 'dpkts', 'ct_flw_http_mthd', 'service', 'is_ftp_login', 'spkts', 'state', 'dload', 'ct_src_dport_ltm', 'dinpkt', 'djit', 'tcprtt', 'ct_state_ttl', 'ct_dst_ltm', 'ackdat', 'proto', 'trans_depth', 'dttl', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'swin', 'dmean'}
    ===Completed dataset 42====
    

iterate over all the 42 data set to find all possible columns 


```python
#iterate over all the 42 data set to find all possibel columns 
col_set = set()
for set_i in col_ruled_sets:
    for col in set_i:
        col_set.add(col)
print(len(col_set))
```

    30
    


```python
col_set
```




    {'ackdat',
     'ct_dst_ltm',
     'ct_dst_sport_ltm',
     'ct_dst_src_ltm',
     'ct_flw_http_mthd',
     'ct_src_dport_ltm',
     'ct_state_ttl',
     'dinpkt',
     'djit',
     'dload',
     'dloss',
     'dmean',
     'dpkts',
     'dtcpb',
     'dttl',
     'is_ftp_login',
     'is_sm_ips_ports',
     'proto',
     'response_body_len',
     'service',
     'sjit',
     'sloss',
     'spkts',
     'state',
     'stcpb',
     'sttl',
     'swin',
     'synack',
     'tcprtt',
     'trans_depth'}




```python
with open("feature_selected.txt", "w") as file:
    file.write(str(list(col_set)))
```

# 5. Machine Learning Models (Response Coding)


```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')
```

## Response Coding:


```python
def create_prime_df(x_data, y):
    d = {'state' : pd.Series(x_data), 'class' : pd.Series(y)}
    return pd.DataFrame(d)
```


```python
#generating response table
def get_response_df(s_u,p_df):
    data = []
    #iterate over unique values in state columns
    for u in tqdm(range(len(s_u))):
        class_0=0
        class_1=0
        #iterate over primary table
        for i in range(len(p_df)): 
            s =p_df.loc[i, "state"]
            c =p_df.loc[i, "class"]
            #if state = unique value and class = 0 add 1 to class_0
            #else add 1 to class_1
            #summing all the values in primary table
            if s == s_u[u] and c == 0:
                class_0 += 1
            elif s == s_u[u] and c == 1:
                class_1 += 1
        #append [state,class0,class1] and return as dataframe
        data.append([s_u[u],class_0,class_1])
    return pd.DataFrame(data, columns=['state', 'class_0', 'class_1'])
```


```python
def encoded_data(input_df,res_df):
    data_e = []
    #iterate over response table
    #if state is present in input table the get the row
    #else 1/2 for class 0 and class 1
    for i in tqdm(range(len(input_df))):
        if input_df.loc[i, "state"] in res_df['state'].values:
            select_r = res_df.loc[res_df['state'] == input_df.loc[i, "state"]]
            c0 = select_r['class_0'].values[0]
            c1 = select_r['class_1'].values[0]
            #append the column in row as encoding rule
            data_e.append([int(c0)/int(c0+c1), int(c1)/int(c0+c1)])
        else:
            #append the column in row as encoding rule
            data_e.append([1/2, 1/2])
    return data_e
```

## 5.1 Reading Train and Test data


```python
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
print(train_data.shape)
train_data.head()
```

    (82332, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000011</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>496</td>
      <td>0</td>
      <td>90909.0902</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000008</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1762</td>
      <td>0</td>
      <td>125000.0003</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000005</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>200000.0051</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000006</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>900</td>
      <td>0</td>
      <td>166666.6608</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000010</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>2126</td>
      <td>0</td>
      <td>100000.0025</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_train = train_data[list(col_set)]
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>synack</th>
      <th>is_sm_ips_ports</th>
      <th>sloss</th>
      <th>dloss</th>
      <th>sjit</th>
      <th>dtcpb</th>
      <th>response_body_len</th>
      <th>sttl</th>
      <th>stcpb</th>
      <th>dpkts</th>
      <th>...</th>
      <th>ct_state_ttl</th>
      <th>ct_dst_ltm</th>
      <th>ackdat</th>
      <th>proto</th>
      <th>trans_depth</th>
      <th>dttl</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>swin</th>
      <th>dmean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>udp</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>udp</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>udp</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>udp</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>udp</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
cat_features = df_train.select_dtypes(include=['category', object]).columns
cat_features
```




    Index(['service', 'state', 'proto'], dtype='object')




```python
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")
print(test_data.shape)
test_data.head()
```

    (175341, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.121478</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>6</td>
      <td>4</td>
      <td>258</td>
      <td>172</td>
      <td>74.087490</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.649902</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>14</td>
      <td>38</td>
      <td>734</td>
      <td>42014</td>
      <td>78.473372</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.623129</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>8</td>
      <td>16</td>
      <td>364</td>
      <td>13186</td>
      <td>14.170161</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.681642</td>
      <td>tcp</td>
      <td>ftp</td>
      <td>FIN</td>
      <td>12</td>
      <td>12</td>
      <td>628</td>
      <td>770</td>
      <td>13.677108</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.449454</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>10</td>
      <td>6</td>
      <td>534</td>
      <td>268</td>
      <td>33.373826</td>
      <td>...</td>
      <td>1</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>39</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_test = test_data[list(col_set)]
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>synack</th>
      <th>is_sm_ips_ports</th>
      <th>sloss</th>
      <th>dloss</th>
      <th>sjit</th>
      <th>dtcpb</th>
      <th>response_body_len</th>
      <th>sttl</th>
      <th>stcpb</th>
      <th>dpkts</th>
      <th>...</th>
      <th>ct_state_ttl</th>
      <th>ct_dst_ltm</th>
      <th>ackdat</th>
      <th>proto</th>
      <th>trans_depth</th>
      <th>dttl</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>swin</th>
      <th>dmean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.177547</td>
      <td>2202533631</td>
      <td>0</td>
      <td>252</td>
      <td>621772692</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>tcp</td>
      <td>0</td>
      <td>254</td>
      <td>1</td>
      <td>1</td>
      <td>255</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
      <td>61.426934</td>
      <td>3077387971</td>
      <td>0</td>
      <td>62</td>
      <td>1417884146</td>
      <td>38</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0.000000</td>
      <td>tcp</td>
      <td>0</td>
      <td>252</td>
      <td>1</td>
      <td>2</td>
      <td>255</td>
      <td>1106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.061458</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>17179.586860</td>
      <td>2963114973</td>
      <td>0</td>
      <td>62</td>
      <td>2116150707</td>
      <td>16</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0.050439</td>
      <td>tcp</td>
      <td>0</td>
      <td>252</td>
      <td>1</td>
      <td>3</td>
      <td>255</td>
      <td>824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>259.080172</td>
      <td>1047442890</td>
      <td>0</td>
      <td>62</td>
      <td>1107119177</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0.000000</td>
      <td>tcp</td>
      <td>0</td>
      <td>252</td>
      <td>1</td>
      <td>3</td>
      <td>255</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.071147</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2415.837634</td>
      <td>1977154190</td>
      <td>0</td>
      <td>254</td>
      <td>2436137549</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0.057234</td>
      <td>tcp</td>
      <td>0</td>
      <td>252</td>
      <td>1</td>
      <td>40</td>
      <td>255</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
cat_feature = df_test.select_dtypes(include=['category', object]).columns
cat_feature
```




    Index(['service', 'state', 'proto'], dtype='object')



### Proto


```python
prime_train_s_df = create_prime_df(df_train['proto'].values, train_data['label'])
prime_test_s_df = create_prime_df(df_test['proto'].values, test_data['label'])
response_df = get_response_df(df_train['proto'].unique(), prime_train_s_df)
response_df.head()
```

    100%|████████████████████████████████████████████████████████████████████████████████| 131/131 [05:17<00:00,  2.43s/it]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>class_0</th>
      <th>class_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>udp</td>
      <td>8097</td>
      <td>21321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arp</td>
      <td>987</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tcp</td>
      <td>27848</td>
      <td>15247</td>
    </tr>
    <tr>
      <th>3</th>
      <td>igmp</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ospf</td>
      <td>38</td>
      <td>638</td>
    </tr>
  </tbody>
</table>
</div>




```python
response_df.to_csv("proto_response.csv")
```


```python
x_train_proto = pd.DataFrame(encoded_data(prime_train_s_df,response_df), columns=['state_0', 'state_1'])
x_test_proto = pd.DataFrame(encoded_data(prime_test_s_df,response_df), columns=['state_0', 'state_1'])
```

    100%|███████████████████████████████████████████████████████████████████████████| 82332/82332 [02:10<00:00, 633.01it/s]
    100%|█████████████████████████████████████████████████████████████████████████| 175341/175341 [04:06<00:00, 711.27it/s]
    

### service


```python
prime_train_ser_df = create_prime_df(df_train['service'].values, train_data['label'])
prime_test_ser_df = create_prime_df(df_test['service'].values, test_data['label'])
response_df_ser = get_response_df(df_train['service'].unique(), prime_train_ser_df)
response_df_ser.head()
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 13/13 [00:27<00:00,  2.13s/it]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>class_0</th>
      <th>class_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-</td>
      <td>27375</td>
      <td>19778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http</td>
      <td>4013</td>
      <td>4274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ftp</td>
      <td>758</td>
      <td>794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ftp-data</td>
      <td>949</td>
      <td>447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>smtp</td>
      <td>635</td>
      <td>1216</td>
    </tr>
  </tbody>
</table>
</div>




```python
response_df_ser.to_csv("service_response.csv")
```


```python
x_train_ser = pd.DataFrame(encoded_data(prime_train_ser_df,response_df_ser), columns=['state_0', 'state_1'])
x_test_ser = pd.DataFrame(encoded_data(prime_test_ser_df,response_df_ser), columns=['state_0', 'state_1'])
```

    100%|███████████████████████████████████████████████████████████████████████████| 82332/82332 [02:39<00:00, 516.76it/s]
    100%|█████████████████████████████████████████████████████████████████████████| 175341/175341 [05:25<00:00, 539.25it/s]
    

### state


```python
prime_train_st_df = create_prime_df(df_train['state'].values, train_data['label'])
prime_test_st_df = create_prime_df(df_test['state'].values, test_data['label'])
response_df_st = get_response_df(df_train['state'].unique(), prime_train_st_df)
response_df_st.head()
```

    100%|████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:21<00:00,  3.08s/it]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>class_0</th>
      <th>class_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INT</td>
      <td>4485</td>
      <td>29678</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FIN</td>
      <td>24172</td>
      <td>15167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>REQ</td>
      <td>1707</td>
      <td>135</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACC</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CON</td>
      <td>6633</td>
      <td>349</td>
    </tr>
  </tbody>
</table>
</div>




```python
response_df_st.to_csv("state_response.csv")
```


```python
x_train_state = pd.DataFrame(encoded_data(prime_train_st_df,response_df_st), columns=['state_0', 'state_1'])
x_test_state = pd.DataFrame(encoded_data(prime_test_st_df,response_df_st), columns=['state_0', 'state_1'])
```

    100%|███████████████████████████████████████████████████████████████████████████| 82332/82332 [02:54<00:00, 471.75it/s]
    100%|█████████████████████████████████████████████████████████████████████████| 175341/175341 [05:32<00:00, 527.85it/s]
    


```python
cat_df_train = pd.concat([x_train_proto, x_train_ser,x_train_state], axis=1, sort=False)
cat_df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train = df_train.drop(cat_features, axis=1)
df_train.shape
```




    (82332, 27)




```python
df_train = df_train.join(cat_df_train)
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>synack</th>
      <th>is_sm_ips_ports</th>
      <th>sloss</th>
      <th>dloss</th>
      <th>sjit</th>
      <th>dtcpb</th>
      <th>response_body_len</th>
      <th>sttl</th>
      <th>stcpb</th>
      <th>dpkts</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>swin</th>
      <th>dmean</th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
cat_df_test = pd.concat([x_test_proto, x_test_ser,x_test_state], axis=1, sort=False)
cat_df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.488402</td>
      <td>0.511598</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test = df_test.drop(cat_feature, axis=1)
df_test.shape
```




    (175341, 27)




```python
df_test = df_test.join(cat_df_test)
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>synack</th>
      <th>is_sm_ips_ports</th>
      <th>sloss</th>
      <th>dloss</th>
      <th>sjit</th>
      <th>dtcpb</th>
      <th>response_body_len</th>
      <th>sttl</th>
      <th>stcpb</th>
      <th>dpkts</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>swin</th>
      <th>dmean</th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
      <th>state_0</th>
      <th>state_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.177547</td>
      <td>2202533631</td>
      <td>0</td>
      <td>252</td>
      <td>621772692</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>255</td>
      <td>43</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
      <td>61.426934</td>
      <td>3077387971</td>
      <td>0</td>
      <td>62</td>
      <td>1417884146</td>
      <td>38</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>255</td>
      <td>1106</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.061458</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>17179.586860</td>
      <td>2963114973</td>
      <td>0</td>
      <td>62</td>
      <td>2116150707</td>
      <td>16</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>255</td>
      <td>824</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>259.080172</td>
      <td>1047442890</td>
      <td>0</td>
      <td>62</td>
      <td>1107119177</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>255</td>
      <td>64</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.488402</td>
      <td>0.511598</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.071147</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2415.837634</td>
      <td>1977154190</td>
      <td>0</td>
      <td>254</td>
      <td>2436137549</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>40</td>
      <td>255</td>
      <td>45</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.580557</td>
      <td>0.419443</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



### 5.1.1 Standardize the data


```python
x = df_train.values
x_test = df_test.values
std_scaler = preprocessing.MinMaxScaler()
std_scaler.fit(x)
```




    MinMaxScaler(copy=True, feature_range=(0, 1))




```python
with open("model_scaler.pkl", 'wb') as file:
    pickle.dump(std_scaler, file)
```


```python
# Load from pickle file
with open("model_scaler.pkl", 'rb') as file:
    minmax_scaler = pickle.load(file)
```


```python
x_scaled = std_scaler.transform(x)
df_train = pd.DataFrame(x_scaled)
x_scaled_test = std_scaler.transform(x_test)
df_test = pd.DataFrame(x_scaled_test)
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.016129</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.016129</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.27524</td>
      <td>0.72476</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.131282</td>
      <td>0.868718</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000020</td>
      <td>0.512828</td>
      <td>0.0</td>
      <td>0.988235</td>
      <td>0.144768</td>
      <td>0.000363</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.028667</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000376</td>
      <td>0.003087</td>
      <td>0.000041</td>
      <td>0.716525</td>
      <td>0.0</td>
      <td>0.243137</td>
      <td>0.330128</td>
      <td>0.003449</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.016129</td>
      <td>1.0</td>
      <td>0.737333</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.019046</td>
      <td>0.0</td>
      <td>0.000188</td>
      <td>0.001090</td>
      <td>0.011578</td>
      <td>0.689918</td>
      <td>0.0</td>
      <td>0.243137</td>
      <td>0.492707</td>
      <td>0.001452</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>1.0</td>
      <td>0.549333</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000188</td>
      <td>0.000545</td>
      <td>0.000175</td>
      <td>0.243882</td>
      <td>0.0</td>
      <td>0.243137</td>
      <td>0.257772</td>
      <td>0.001089</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>1.0</td>
      <td>0.042667</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.498170</td>
      <td>0.501830</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.022049</td>
      <td>0.0</td>
      <td>0.000376</td>
      <td>0.000182</td>
      <td>0.001628</td>
      <td>0.460351</td>
      <td>0.0</td>
      <td>0.996078</td>
      <td>0.567210</td>
      <td>0.000545</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.629032</td>
      <td>1.0</td>
      <td>0.030000</td>
      <td>0.6462</td>
      <td>0.3538</td>
      <td>0.592168</td>
      <td>0.407832</td>
      <td>0.614454</td>
      <td>0.385546</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
y_train = train_data['label']
y_test = test_data['label']
print("train data shape", df_train.shape, y_train.shape)
print("test data shape", df_test.shape, y_test.shape)
```

    train data shape (82332, 33) (82332,)
    test data shape (175341, 33) (175341,)
    

## 5.2 Logistic Regression Model


```python
prams={
    'alpha':[10 ** x for x in range(-4, 1)],
     'max_iter':[5, 10, 20, 50, 100],
    'eta0': [10 ** x for x in range(-4, 1)]
}
lr_cfl=GridSearchCV(SGDClassifier(penalty='l2', loss='log', n_jobs = -1), param_grid=prams,verbose=10,n_jobs=-1)
lr_cfl.fit(df_train,y_train)
```

    Fitting 5 folds for each of 125 candidates, totalling 625 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    5.7s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    6.7s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    7.6s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    8.4s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    9.4s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:   10.3s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   11.6s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   12.8s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   14.2s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:   15.9s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   17.4s
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:   19.1s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   20.8s
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:   22.5s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   24.4s
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:   26.2s
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:   27.9s
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:   30.1s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   31.9s
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:   33.9s
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:   36.1s
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:   38.5s
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:   40.6s
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:   43.2s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   45.2s
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:   47.6s
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:   50.1s
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:   53.1s
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:   55.9s
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed:   59.6s
    [Parallel(n_jobs=-1)]: Done 625 out of 625 | elapsed:  1.0min finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SGDClassifier(alpha=0.0001, average=False,
                                         class_weight=None, early_stopping=False,
                                         epsilon=0.1, eta0=0.0, fit_intercept=True,
                                         l1_ratio=0.15, learning_rate='optimal',
                                         loss='log', max_iter=1000,
                                         n_iter_no_change=5, n_jobs=-1,
                                         penalty='l2', power_t=0.5,
                                         random_state=None, shuffle=True, tol=0.001,
                                         validation_fraction=0.1, verbose=0,
                                         warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                             'eta0': [0.0001, 0.001, 0.01, 0.1, 1],
                             'max_iter': [5, 10, 20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=10)




```python
results = pd.DataFrame.from_dict(lr_cfl.cv_results_)
results = results.sort_values(['rank_test_score'])
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_alpha</th>
      <th>param_eta0</th>
      <th>param_max_iter</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>0.733896</td>
      <td>0.125073</td>
      <td>0.012087</td>
      <td>0.001493</td>
      <td>0.0001</td>
      <td>0.01</td>
      <td>50</td>
      <td>{'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 50}</td>
      <td>0.939698</td>
      <td>0.979474</td>
      <td>0.858253</td>
      <td>0.828191</td>
      <td>0.785437</td>
      <td>0.878211</td>
      <td>0.071473</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.771722</td>
      <td>0.084044</td>
      <td>0.009397</td>
      <td>0.001724</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>20</td>
      <td>{'alpha': 0.0001, 'eta0': 0.0001, 'max_iter': 20}</td>
      <td>0.901561</td>
      <td>0.972308</td>
      <td>0.869489</td>
      <td>0.814345</td>
      <td>0.800255</td>
      <td>0.871591</td>
      <td>0.062310</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.738520</td>
      <td>0.055430</td>
      <td>0.011809</td>
      <td>0.003169</td>
      <td>0.0001</td>
      <td>0.001</td>
      <td>10</td>
      <td>{'alpha': 0.0001, 'eta0': 0.001, 'max_iter': 10}</td>
      <td>0.897431</td>
      <td>0.963928</td>
      <td>0.868213</td>
      <td>0.819568</td>
      <td>0.799648</td>
      <td>0.869758</td>
      <td>0.058431</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.712953</td>
      <td>0.036958</td>
      <td>0.016193</td>
      <td>0.010511</td>
      <td>0.0001</td>
      <td>0.01</td>
      <td>20</td>
      <td>{'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 20}</td>
      <td>0.918807</td>
      <td>0.962956</td>
      <td>0.859225</td>
      <td>0.796793</td>
      <td>0.806450</td>
      <td>0.868846</td>
      <td>0.064079</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.495005</td>
      <td>0.021378</td>
      <td>0.013800</td>
      <td>0.004314</td>
      <td>0.0001</td>
      <td>0.1</td>
      <td>5</td>
      <td>{'alpha': 0.0001, 'eta0': 0.1, 'max_iter': 5}</td>
      <td>0.912613</td>
      <td>0.944738</td>
      <td>0.870339</td>
      <td>0.809790</td>
      <td>0.789445</td>
      <td>0.865385</td>
      <td>0.059008</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(lr_cfl.best_params_)
```

    {'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 50}
    


```python
logisticR=SGDClassifier(alpha=lr_cfl.best_params_['alpha'],eta0=lr_cfl.best_params_['eta0'], penalty='l2', loss='log', n_jobs = -1, max_iter=lr_cfl.best_params_['max_iter'])
logisticR.fit(df_train,y_train)
sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
sig_clf.fit(df_train, y_train)
predict_y_tr_lr = sig_clf.predict(df_train)
predict_y_te_lr = sig_clf.predict(df_test)
lr_f1 = f1_score(y_test, predict_y_te_lr)
print(lr_f1)
```

    0.9423032118488934
    


```python
cm_lr = confusion_matrix(y_test, predict_y_te_lr)
```


```python
tn, fp, fn, tp = cm_lr.ravel()
```


```python
fpr_lr = (fp/(fp+tn))*100
fnr_lr = (fn/(fn+tp))*100
far_lr = (fpr_lr+fnr_lr)/2
print("FAR:",far_lr)
```

    FAR: 9.493785462605953
    


```python
def plot_cm(cm):
    sns.heatmap(cm, annot=True, cmap=sns.light_palette("blue"), fmt="g")
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    plt.show()
```


```python
plot_cm(cm_lr)
```


![png](output_162_0.png)



```python
from sklearn.metrics import roc_curve, auc
def plot_roc_curve(fpr_tr, tpr_tr,fpr_te, tpr_te):
    '''
    plot the ROC curve for the FPR and TPR value
    '''
    plt.plot(fpr_te, tpr_te, 'k.-', color='orange', label='ROC_test AUC:%.3f'% auc(fpr_te, tpr_te))
    plt.plot(fpr_tr, tpr_tr, 'k.-', color='green', label='ROC_train AUC:%.3f'% auc(fpr_tr, tpr_tr))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
```


```python
#finding the FPR and TPR for logistic reg model set
fpr_te_lr, tpr_te_lr, t_te_lr = roc_curve(y_test, predict_y_te_lr)
fpr_tr_lr, tpr_tr_lr, t_tr_lr = roc_curve(y_train, predict_y_tr_lr)
auc_te_lr = auc(fpr_te_lr, tpr_te_lr)
print("AUC_LR: ",auc_te_lr)
plot_roc_curve(fpr_tr_lr,tpr_tr_lr,fpr_te_lr, tpr_te_lr)
```

    AUC_LR:  0.9050621453739406
    


![png](output_164_1.png)


## 5.3 Support Vector Machine Model


```python
prams={
    'alpha':[10 ** x for x in range(-4, 1)],
     'max_iter':[5, 10, 20, 50, 100],
    'eta0': [10 ** x for x in range(-4, 1)]
}
svm_cfl=GridSearchCV(SGDClassifier(penalty='l2', loss='hinge', n_jobs = -1), param_grid=prams,verbose=10,n_jobs=-1)
svm_cfl.fit(df_train,y_train)
```

    Fitting 5 folds for each of 125 candidates, totalling 625 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    1.0s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    1.9s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    2.7s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    3.5s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:    4.8s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    5.8s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:    7.1s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:    8.3s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:   10.0s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   11.4s
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:   13.0s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   14.4s
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:   15.7s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   17.1s
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:   18.7s
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:   20.0s
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:   22.1s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   23.9s
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:   26.2s
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:   28.1s
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:   30.1s
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:   31.9s
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:   33.8s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   35.7s
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:   37.8s
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:   39.9s
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:   41.9s
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:   44.3s
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed:   46.7s
    [Parallel(n_jobs=-1)]: Done 625 out of 625 | elapsed:   48.5s finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SGDClassifier(alpha=0.0001, average=False,
                                         class_weight=None, early_stopping=False,
                                         epsilon=0.1, eta0=0.0, fit_intercept=True,
                                         l1_ratio=0.15, learning_rate='optimal',
                                         loss='hinge', max_iter=1000,
                                         n_iter_no_change=5, n_jobs=-1,
                                         penalty='l2', power_t=0.5,
                                         random_state=None, shuffle=True, tol=0.001,
                                         validation_fraction=0.1, verbose=0,
                                         warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                             'eta0': [0.0001, 0.001, 0.01, 0.1, 1],
                             'max_iter': [5, 10, 20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=10)




```python
print(svm_cfl.best_params_)
```

    {'alpha': 0.0001, 'eta0': 1, 'max_iter': 10}
    


```python
svm=SGDClassifier(alpha=svm_cfl.best_params_['alpha'],eta0=svm_cfl.best_params_['eta0'], penalty='l2', loss='hinge', n_jobs = -1, max_iter=svm_cfl.best_params_['max_iter'])
svm.fit(df_train,y_train)
sig_clf_svm = CalibratedClassifierCV(svm, method="sigmoid")
sig_clf_svm.fit(df_train, y_train)
predict_y_tr_svm = sig_clf.predict(df_train)
predict_y_te_svm = sig_clf_svm.predict(df_test)
svm_f1 = f1_score(y_test, predict_y_te_svm)
print("F1-Score", svm_f1)
```

    F1-Score 0.9197150144204811
    


```python
cm_svm = confusion_matrix(y_test, predict_y_te_svm)
```


```python
tn, fp, fn, tp = cm_svm.ravel()
```


```python
fpr_svm = fp/(fp+tn)*100
fnr_svm = fn/(fn+tp)*100
far_svm = (fpr_svm+fnr_svm)/2
print("FAR:", far_svm)
```

    FAR: 9.417783793619005
    


```python
plot_cm(cm_svm)
```


![png](output_172_0.png)



```python
#finding the FPR and TPR for SVM set
fpr_te_svm, tpr_te_svm, t_te_svm = roc_curve(y_test, predict_y_te_svm)
fpr_tr_svm, tpr_tr_svm, t_tr_svm = roc_curve(y_train, predict_y_tr_svm)
auc_te_svm = auc(fpr_te_svm, tpr_te_svm)
print("AUC_SVM: ",auc_te_svm)
plot_roc_curve(fpr_tr_svm,tpr_tr_svm,fpr_te_svm, tpr_te_svm)
```

    AUC_SVM:  0.9058221620638099
    


![png](output_173_1.png)


## 5.4 Random Forest Model


```python
param_grid = {"n_estimators": [10,100,500,1000, 2000],
    "min_samples_split": [50, 80, 120, 200],
              "max_depth": [3, 5, 10, 50, 100]}
rfc = RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1)
gridCV_rfc = GridSearchCV(rfc, param_grid, cv=3, verbose=10, n_jobs=-1)
gridCV_rfc.fit(df_train, y_train)
#grid Search cv results are stored in result for future use
results_rfc = pd.DataFrame.from_dict(gridCV_rfc.cv_results_)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.8s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   22.6s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   49.9s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  2.6min
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:  4.7min
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:  6.0min
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:  7.9min
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:  9.5min
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed: 11.3min
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed: 13.4min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed: 16.1min
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed: 18.6min
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 22.2min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed: 26.4min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed: 29.4min
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed: 34.2min
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed: 37.3min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 43.5min finished
    


```python
results_rfc = results_rfc.sort_values(['rank_test_score'])
results_rfc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_depth</th>
      <th>param_min_samples_split</th>
      <th>param_n_estimators</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>63.188302</td>
      <td>5.296170</td>
      <td>8.351116</td>
      <td>0.673394</td>
      <td>50</td>
      <td>50</td>
      <td>500</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.950590</td>
      <td>0.945453</td>
      <td>0.902128</td>
      <td>0.932724</td>
      <td>0.021736</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>57.129693</td>
      <td>3.058769</td>
      <td>6.948288</td>
      <td>0.296046</td>
      <td>100</td>
      <td>50</td>
      <td>500</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.950590</td>
      <td>0.945453</td>
      <td>0.902128</td>
      <td>0.932724</td>
      <td>0.021736</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>118.540384</td>
      <td>13.901412</td>
      <td>9.046493</td>
      <td>0.692907</td>
      <td>50</td>
      <td>50</td>
      <td>1000</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.950736</td>
      <td>0.944979</td>
      <td>0.902310</td>
      <td>0.932675</td>
      <td>0.021599</td>
      <td>3</td>
    </tr>
    <tr>
      <th>83</th>
      <td>108.926263</td>
      <td>12.328838</td>
      <td>8.117628</td>
      <td>0.368971</td>
      <td>100</td>
      <td>50</td>
      <td>1000</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.950736</td>
      <td>0.944979</td>
      <td>0.902310</td>
      <td>0.932675</td>
      <td>0.021599</td>
      <td>3</td>
    </tr>
    <tr>
      <th>64</th>
      <td>224.453358</td>
      <td>22.106221</td>
      <td>11.170527</td>
      <td>1.758222</td>
      <td>50</td>
      <td>50</td>
      <td>2000</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.950772</td>
      <td>0.945088</td>
      <td>0.901399</td>
      <td>0.932420</td>
      <td>0.022057</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(gridCV_rfc.best_params_)
```

    {'max_depth': 50, 'min_samples_split': 50, 'n_estimators': 500}
    


```python
rfc= RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1, max_depth=gridCV_rfc.best_params_['max_depth'],min_samples_split=gridCV_rfc.best_params_['min_samples_split'], n_estimators=gridCV_rfc.best_params_['n_estimators'])
rfc.fit(df_train,y_train)
sig_clf_rfc = CalibratedClassifierCV(rfc, method="sigmoid")
sig_clf_rfc.fit(df_train, y_train)
predict_y_tr_rfc = sig_clf_rfc.predict(df_train)
predict_y_te_rfc = sig_clf_rfc.predict(df_test)
rfc_f1 = f1_score(y_test, predict_y_te_rfc)
print(rfc_f1)
```

    0.929126145048273
    


```python
cm_rfc = confusion_matrix(y_test, predict_y_te_rfc)
```


```python
tn, fp, fn, tp = cm_rfc.ravel()
```


```python
fpr_rfc = fp/(fp+tn)*100
fnr_rfc = fn/(fn+tp)*100
far_rfc = (fpr_rfc+fnr_rfc)/2
print("far:",far_rfc)
```

    far: 7.679130780105509
    


```python
plot_cm(cm_rfc)
```


![png](output_182_0.png)



```python
#finding the FPR and TPR for RFC set
fpr_te_rfc, tpr_te_rfc, t_te_rfc = roc_curve(y_test, predict_y_te_rfc)
fpr_tr_rfc, tpr_tr_rfc, t_tr_rfc = roc_curve(y_train, predict_y_tr_rfc)
auc_te_rfc = auc(fpr_te_rfc, tpr_te_rfc)
print("AUC_RFC: ",auc_te_rfc)
plot_roc_curve(fpr_tr_rfc,tpr_tr_rfc,fpr_te_rfc, tpr_te_rfc)
```

    AUC_RFC:  0.9232086921989449
    


![png](output_183_1.png)


## 5.5 Stacking classifier


```python
clf1 = SGDClassifier(alpha=0.0001,eta0=1, penalty='l2', loss='log', n_jobs = -1, max_iter=10)
clf1.fit(df_train, y_train)
sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

clf2 = SGDClassifier(alpha=0.0001,eta0=0.0001, penalty='l2', loss='hinge', n_jobs = -1, max_iter=5)
clf2.fit(df_train, y_train)
sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")


clf3 = RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1, max_depth=50,min_samples_split=50, n_estimators=10)
clf3.fit(df_train, y_train)
sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")
```


```python
alpha = [0.0001,0.001,0.01,0.1,1,10] 
best_alpha = 999
for i in alpha:
    lr = LogisticRegression(C=i)
    sclf = StackingClassifier(estimators=[("lr",sig_clf1), ("svm", sig_clf2),("RF", sig_clf3)], final_estimator=lr, n_jobs=-1)
    sclf.fit(df_train, y_train)
    print("Stacking Classifer : for the value of alpha: %f Log loss: %0.3f F1-score: %0.3f" % (i, log_loss(y_test, sclf.predict_proba(df_test)),f1_score(y_test, sclf.predict(df_test))))
```

    Stacking Classifer : for the value of alpha: 0.000100 Log loss: 0.422 F1-score: 0.946
    Stacking Classifer : for the value of alpha: 0.001000 Log loss: 0.252 F1-score: 0.931
    Stacking Classifer : for the value of alpha: 0.010000 Log loss: 0.221 F1-score: 0.930
    Stacking Classifer : for the value of alpha: 0.100000 Log loss: 0.216 F1-score: 0.931
    Stacking Classifer : for the value of alpha: 1.000000 Log loss: 0.215 F1-score: 0.931
    Stacking Classifer : for the value of alpha: 10.000000 Log loss: 0.218 F1-score: 0.930
    


```python
lr = LogisticRegression(C=0.0001)
sig_clf_sc = StackingClassifier(estimators=[("lr",sig_clf1), ("svm", sig_clf2),("RF", sig_clf3)], final_estimator=lr, n_jobs=-1)
sig_clf_sc.fit(df_train, y_train)
predict_y_tr_sc= sig_clf_sc.predict(df_train)
predict_y_te_sc = sig_clf_sc.predict(df_test)
sc_f1 = f1_score(y_test, predict_y_te_sc)
print(sc_f1)
```

    0.9453929718455836
    


```python
import pickle

file_pkl = "model_rf.pkl"
with open(file_pkl, 'wb') as file:
    pickle.dump(sig_clf_rfc, file)
```


```python
cm_sc = confusion_matrix(y_test, predict_y_te_sc)
```


```python
tn, fp, fn, tp = cm_sc.ravel()
```


```python
fpr_sc = fp/(fp+tn)*100
fnr_sc = fn/(fn+tp)*100
far_sc = (fpr_sc+fnr_sc)/2
print("far:",far_sc)
```

    far: 7.396553550929091
    


```python
plot_cm(cm_sc)
```


![png](output_192_0.png)



```python
#finding the FPR and TPR for RFC set
fpr_te_sc, tpr_te_sc, t_te_sc = roc_curve(y_test, predict_y_te_sc)
fpr_tr_sc, tpr_tr_sc, t_tr_sc = roc_curve(y_train, predict_y_tr_sc)
auc_te_sc = auc(fpr_te_sc, tpr_te_sc)
print("AUC_SC: ",auc_te_sc)
plot_roc_curve(fpr_tr_sc,tpr_tr_sc,fpr_te_sc, tpr_te_sc)
```

    AUC_SC:  0.9262184317717418
    


![png](output_193_1.png)


## 5.6. Model Evaluation

Measures | Equations
----------|------------
FPR | FP/(TN + FP)
FNR | FN/(FN + TP)
FAR | (FPR + FNR)/2


```python
from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Model", "F1 Score", "AUC","FPR %","FNR %","FAR %"]
x.add_row(["Logistic Regression", "{0:.4}".format(lr_f1), "{0:.4}".format(auc_te_lr),"%.2f" % float(fpr_lr),"%.2f" % float(fnr_lr),"%.2f" % float(far_lr)])
x.add_row(["Linear SVM", "{0:.4}".format(svm_f1), "{0:.4}".format(auc_te_svm),"%.2f" % float(fpr_svm),"%.2f" % float(fnr_svm),"%.2f" % float(far_svm)])
x.add_row(["Random Forest", "{0:.4}".format(rfc_f1), "{0:.4}".format(auc_te_rfc),"%.2f" % float(fpr_rfc),"%.2f" % float(fnr_rfc),"%.2f" % float(far_rfc)])
x.add_row(["Stacking Classifier", "{0:.4}".format(sc_f1), "{0:.4}".format(auc_te_sc),"%.2f" % float(fpr_sc),"%.2f" % float(fnr_sc),"%.2f" % float(far_sc)])
print(x)
```

    +---------------------+----------+--------+-------+-------+-------+
    |        Model        | F1 Score |  AUC   | FPR % | FNR % | FAR % |
    +---------------------+----------+--------+-------+-------+-------+
    | Logistic Regression |  0.9423  | 0.9051 | 13.88 |  5.11 |  9.49 |
    |      Linear SVM     |  0.9197  | 0.9058 |  6.61 | 12.22 |  9.42 |
    |    Random Forest    |  0.9291  | 0.9232 |  3.58 | 11.78 |  7.68 |
    | Stacking Classifier |  0.9453  | 0.9262 |  7.91 |  6.85 |  7.39 |
    +---------------------+----------+--------+-------+-------+-------+
    

# 6. Machine Learning Models (Label Encoder)

## 6.1 Reading Train and Test data


```python
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
print(train_data.shape)
train_data.head()
```

    (82332, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000011</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>496</td>
      <td>0</td>
      <td>90909.0902</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000008</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1762</td>
      <td>0</td>
      <td>125000.0003</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000005</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>200000.0051</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000006</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>900</td>
      <td>0</td>
      <td>166666.6608</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000010</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>2126</td>
      <td>0</td>
      <td>100000.0025</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
cat_feature = train_data.select_dtypes(include=['category', object]).columns
```


```python
train_data[cat_feature] = train_data[cat_feature].apply(LabelEncoder().fit_transform)
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000011</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>496</td>
      <td>0</td>
      <td>90909.0902</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000008</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1762</td>
      <td>0</td>
      <td>125000.0003</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000005</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>200000.0051</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000006</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>900</td>
      <td>0</td>
      <td>166666.6608</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000010</td>
      <td>117</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>2126</td>
      <td>0</td>
      <td>100000.0025</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_train = train_data[list(col_set)]
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ackdat</th>
      <th>dmean</th>
      <th>proto</th>
      <th>ct_state_ttl</th>
      <th>ct_dst_ltm</th>
      <th>trans_depth</th>
      <th>service</th>
      <th>ct_dst_src_ltm</th>
      <th>response_body_len</th>
      <th>dload</th>
      <th>...</th>
      <th>djit</th>
      <th>synack</th>
      <th>sjit</th>
      <th>dloss</th>
      <th>dtcpb</th>
      <th>swin</th>
      <th>stcpb</th>
      <th>tcprtt</th>
      <th>is_sm_ips_ports</th>
      <th>sttl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0</td>
      <td>117</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0</td>
      <td>117</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0</td>
      <td>117</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0</td>
      <td>117</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0</td>
      <td>117</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>254</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")
print(test_data.shape)
test_data.head()
```

    (175341, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.121478</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>6</td>
      <td>4</td>
      <td>258</td>
      <td>172</td>
      <td>74.087490</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.649902</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>14</td>
      <td>38</td>
      <td>734</td>
      <td>42014</td>
      <td>78.473372</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.623129</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>8</td>
      <td>16</td>
      <td>364</td>
      <td>13186</td>
      <td>14.170161</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.681642</td>
      <td>tcp</td>
      <td>ftp</td>
      <td>FIN</td>
      <td>12</td>
      <td>12</td>
      <td>628</td>
      <td>770</td>
      <td>13.677108</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.449454</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>10</td>
      <td>6</td>
      <td>534</td>
      <td>268</td>
      <td>33.373826</td>
      <td>...</td>
      <td>1</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>39</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
cat_feature_test = test_data.select_dtypes(include=['category', object]).columns
```


```python
test_data[cat_feature_test] = test_data[cat_feature_test].apply(LabelEncoder().fit_transform)
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.121478</td>
      <td>113</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>258</td>
      <td>172</td>
      <td>74.087490</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.649902</td>
      <td>113</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>38</td>
      <td>734</td>
      <td>42014</td>
      <td>78.473372</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.623129</td>
      <td>113</td>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>16</td>
      <td>364</td>
      <td>13186</td>
      <td>14.170161</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.681642</td>
      <td>113</td>
      <td>3</td>
      <td>2</td>
      <td>12</td>
      <td>12</td>
      <td>628</td>
      <td>770</td>
      <td>13.677108</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.449454</td>
      <td>113</td>
      <td>0</td>
      <td>2</td>
      <td>10</td>
      <td>6</td>
      <td>534</td>
      <td>268</td>
      <td>33.373826</td>
      <td>...</td>
      <td>1</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>39</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_test = test_data[list(col_set)]
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ackdat</th>
      <th>dmean</th>
      <th>proto</th>
      <th>ct_state_ttl</th>
      <th>ct_dst_ltm</th>
      <th>trans_depth</th>
      <th>service</th>
      <th>ct_dst_src_ltm</th>
      <th>response_body_len</th>
      <th>dload</th>
      <th>...</th>
      <th>djit</th>
      <th>synack</th>
      <th>sjit</th>
      <th>dloss</th>
      <th>dtcpb</th>
      <th>swin</th>
      <th>stcpb</th>
      <th>tcprtt</th>
      <th>is_sm_ips_ports</th>
      <th>sttl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>43</td>
      <td>113</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8495.365234</td>
      <td>...</td>
      <td>11.830604</td>
      <td>0.000000</td>
      <td>30.177547</td>
      <td>0</td>
      <td>2202533631</td>
      <td>255</td>
      <td>621772692</td>
      <td>0.000000</td>
      <td>0</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>1106</td>
      <td>113</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>503571.312500</td>
      <td>...</td>
      <td>1387.778330</td>
      <td>0.000000</td>
      <td>61.426934</td>
      <td>17</td>
      <td>3077387971</td>
      <td>255</td>
      <td>1417884146</td>
      <td>0.000000</td>
      <td>0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.050439</td>
      <td>824</td>
      <td>113</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>60929.230470</td>
      <td>...</td>
      <td>11420.926230</td>
      <td>0.061458</td>
      <td>17179.586860</td>
      <td>6</td>
      <td>2963114973</td>
      <td>255</td>
      <td>2116150707</td>
      <td>0.111897</td>
      <td>0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>64</td>
      <td>113</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>3358.622070</td>
      <td>...</td>
      <td>4991.784669</td>
      <td>0.000000</td>
      <td>259.080172</td>
      <td>3</td>
      <td>1047442890</td>
      <td>255</td>
      <td>1107119177</td>
      <td>0.000000</td>
      <td>0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.057234</td>
      <td>45</td>
      <td>113</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>3987.059814</td>
      <td>...</td>
      <td>115.807000</td>
      <td>0.071147</td>
      <td>2415.837634</td>
      <td>1</td>
      <td>1977154190</td>
      <td>255</td>
      <td>2436137549</td>
      <td>0.128381</td>
      <td>0</td>
      <td>254</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



### 6.1.1 Standardize the data


```python
x = df_train.values
x_test = df_test.values
std_scaler = preprocessing.MinMaxScaler()
std_scaler.fit(x)
x_scaled = std_scaler.transform(x)
df_train = pd.DataFrame(x_scaled)
x_scaled_test = std_scaler.transform(x_test)
df_test = pd.DataFrame(x_scaled_test)
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.016129</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.016129</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>0.333333</td>
      <td>0.017241</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>0.333333</td>
      <td>0.017241</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.996078</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
y_train = train_data['label']
y_test = test_data['label']
print("train data shape", df_train.shape, y_train.shape)
print("test data shape", df_test.shape, y_test.shape)
```

    train data shape (82332, 30) (82332,)
    test data shape (175341, 30) (175341,)
    


```python
[10 ** x for x in range(-5, 2)]
```




    [1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]



## 6.2 Logistic Regression Model


```python
prams={
    'alpha':[10 ** x for x in range(-4, 1)],
     'max_iter':[5, 10, 20, 50, 100],
    'eta0': [10 ** x for x in range(-4, 1)]
}
lr_cfl=GridSearchCV(SGDClassifier(penalty='l2', loss='log', n_jobs = -1), param_grid=prams,verbose=10,n_jobs=-1)
lr_cfl.fit(df_train,y_train)
```

    Fitting 5 folds for each of 125 candidates, totalling 625 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.6s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    1.3s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    1.9s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    3.4s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:    4.4s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    5.4s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:    6.5s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:    7.5s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:    8.8s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:    9.9s
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:   11.5s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   13.1s
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:   14.5s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   16.0s
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:   17.7s
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:   19.2s
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:   21.1s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   22.9s
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:   24.8s
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:   26.8s
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:   28.9s
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:   30.9s
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:   33.4s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   35.4s
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:   37.9s
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:   40.4s
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:   42.7s
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:   45.2s
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed:   48.2s
    [Parallel(n_jobs=-1)]: Done 625 out of 625 | elapsed:   50.0s finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SGDClassifier(alpha=0.0001, average=False,
                                         class_weight=None, early_stopping=False,
                                         epsilon=0.1, eta0=0.0, fit_intercept=True,
                                         l1_ratio=0.15, learning_rate='optimal',
                                         loss='log', max_iter=1000,
                                         n_iter_no_change=5, n_jobs=-1,
                                         penalty='l2', power_t=0.5,
                                         random_state=None, shuffle=True, tol=0.001,
                                         validation_fraction=0.1, verbose=0,
                                         warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                             'eta0': [0.0001, 0.001, 0.01, 0.1, 1],
                             'max_iter': [5, 10, 20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=10)




```python
results = pd.DataFrame.from_dict(lr_cfl.cv_results_)
results = results.sort_values(['rank_test_score'])
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_alpha</th>
      <th>param_eta0</th>
      <th>param_max_iter</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>0.659036</td>
      <td>0.014011</td>
      <td>0.007781</td>
      <td>0.000399</td>
      <td>0.0001</td>
      <td>0.001</td>
      <td>100</td>
      <td>{'alpha': 0.0001, 'eta0': 0.001, 'max_iter': 100}</td>
      <td>0.913463</td>
      <td>0.961134</td>
      <td>0.858071</td>
      <td>0.795943</td>
      <td>0.772015</td>
      <td>0.860125</td>
      <td>0.070618</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.619651</td>
      <td>0.027652</td>
      <td>0.008979</td>
      <td>0.001263</td>
      <td>0.0001</td>
      <td>0.001</td>
      <td>20</td>
      <td>{'alpha': 0.0001, 'eta0': 0.001, 'max_iter': 20}</td>
      <td>0.905629</td>
      <td>0.978928</td>
      <td>0.845621</td>
      <td>0.793878</td>
      <td>0.765395</td>
      <td>0.857890</td>
      <td>0.077113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.601999</td>
      <td>0.037194</td>
      <td>0.010476</td>
      <td>0.001782</td>
      <td>0.0001</td>
      <td>0.001</td>
      <td>10</td>
      <td>{'alpha': 0.0001, 'eta0': 0.001, 'max_iter': 10}</td>
      <td>0.905872</td>
      <td>0.944131</td>
      <td>0.859104</td>
      <td>0.801348</td>
      <td>0.765820</td>
      <td>0.855255</td>
      <td>0.065392</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.626726</td>
      <td>0.041205</td>
      <td>0.011367</td>
      <td>0.001019</td>
      <td>0.0001</td>
      <td>1</td>
      <td>10</td>
      <td>{'alpha': 0.0001, 'eta0': 1, 'max_iter': 10}</td>
      <td>0.904415</td>
      <td>0.966721</td>
      <td>0.839184</td>
      <td>0.797279</td>
      <td>0.760962</td>
      <td>0.853712</td>
      <td>0.073946</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.582242</td>
      <td>0.020178</td>
      <td>0.011183</td>
      <td>0.005554</td>
      <td>0.0001</td>
      <td>0.1</td>
      <td>10</td>
      <td>{'alpha': 0.0001, 'eta0': 0.1, 'max_iter': 10}</td>
      <td>0.899435</td>
      <td>0.978381</td>
      <td>0.838759</td>
      <td>0.776388</td>
      <td>0.760962</td>
      <td>0.850785</td>
      <td>0.080493</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(lr_cfl.best_params_)
```

    {'alpha': 0.0001, 'eta0': 0.001, 'max_iter': 100}
    


```python
logisticR=SGDClassifier(alpha=lr_cfl.best_params_['alpha'],eta0=lr_cfl.best_params_['eta0'], penalty='l2', loss='log', n_jobs = -1, max_iter=lr_cfl.best_params_['max_iter'])
logisticR.fit(df_train,y_train)
sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
sig_clf.fit(df_train, y_train)
predict_y_tr_lr = sig_clf.predict(df_train)
predict_y_te_lr = sig_clf.predict(df_test)
lr_f1 = f1_score(y_test, predict_y_te_lr)
print(lr_f1)
```

    0.8029353446987435
    


```python
cm_lr = confusion_matrix(y_test, predict_y_te_lr)
```


```python
tn, fp, fn, tp = cm_lr.ravel()
```


```python
fpr_lr = fp/(fp+tn)*100
fnr_lr = fn/(fn+tp)*100
far_lr = (fpr_lr+fnr_lr)/2
print("FAR: %0.2f" %far_lr)
```

    FAR: 16.95
    


```python
def plot_cm(cm):
    sns.heatmap(cm, annot=True, cmap=sns.light_palette("blue"), fmt="g")
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    plt.show()
```


```python
plot_cm(cm_lr)
```


![png](output_221_0.png)



```python
from sklearn.metrics import roc_curve, auc
def plot_roc_curve(fpr_tr, tpr_tr,fpr_te, tpr_te):
    '''
    plot the ROC curve for the FPR and TPR value
    '''
    plt.plot(fpr_te, tpr_te, 'k.-', color='orange', label='ROC_test AUC:%.3f'% auc(fpr_te, tpr_te))
    plt.plot(fpr_tr, tpr_tr, 'k.-', color='green', label='ROC_train AUC:%.3f'% auc(fpr_tr, tpr_tr))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
```


```python
#finding the FPR and TPR for logistic reg model set
fpr_te_lr, tpr_te_lr, t_te_lr = roc_curve(y_test, predict_y_te_lr)
fpr_tr_lr, tpr_tr_lr, t_tr_lr = roc_curve(y_train, predict_y_tr_lr)
auc_te_lr = auc(fpr_te_lr, tpr_te_lr)
print("AUC_LR: ",auc_te_lr)
plot_roc_curve(fpr_tr_lr,tpr_tr_lr,fpr_te_lr, tpr_te_lr)
```

    AUC_LR:  0.8304820999129745
    


![png](output_223_1.png)


## 6.3 Support Vector Machine Model


```python
prams={
    'alpha':[10 ** x for x in range(-4, 1)],
     'max_iter':[5, 10, 20, 50, 100],
    'eta0': [10 ** x for x in range(-4, 1)]
}
svm_cfl=GridSearchCV(SGDClassifier(penalty='l1', loss='hinge', n_jobs = -1), param_grid=prams,verbose=10,n_jobs=-1)
svm_cfl.fit(df_train,y_train)
```

    Fitting 5 folds for each of 125 candidates, totalling 625 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    0.7s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    1.5s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    2.3s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    3.3s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    4.1s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:    5.4s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    6.5s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:    7.9s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:    9.2s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:   11.0s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   12.4s
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:   14.1s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   15.6s
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:   17.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   19.0s
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:   20.8s
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:   22.6s
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:   24.7s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   26.5s
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:   28.5s
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:   30.5s
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:   32.9s
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:   34.9s
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:   37.3s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   39.6s
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:   42.3s
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:   45.1s
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:   48.0s
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:   51.0s
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed:   54.3s
    [Parallel(n_jobs=-1)]: Done 625 out of 625 | elapsed:   56.9s finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SGDClassifier(alpha=0.0001, average=False,
                                         class_weight=None, early_stopping=False,
                                         epsilon=0.1, eta0=0.0, fit_intercept=True,
                                         l1_ratio=0.15, learning_rate='optimal',
                                         loss='hinge', max_iter=1000,
                                         n_iter_no_change=5, n_jobs=-1,
                                         penalty='l1', power_t=0.5,
                                         random_state=None, shuffle=True, tol=0.001,
                                         validation_fraction=0.1, verbose=0,
                                         warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                             'eta0': [0.0001, 0.001, 0.01, 0.1, 1],
                             'max_iter': [5, 10, 20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=10)




```python
print(svm_cfl.best_params_)
```

    {'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 100}
    


```python
svm=SGDClassifier(alpha=svm_cfl.best_params_['alpha'],eta0=svm_cfl.best_params_['eta0'], penalty='l2', loss='hinge', n_jobs = -1, max_iter=svm_cfl.best_params_['max_iter'])
svm.fit(df_train,y_train)
sig_clf_svm = CalibratedClassifierCV(svm, method="sigmoid")
sig_clf_svm.fit(df_train, y_train)
predict_y_tr_svm = sig_clf.predict(df_train)
predict_y_te_svm = sig_clf_svm.predict(df_test)
svm_f1 = f1_score(y_test, predict_y_te_svm)
print("F1-Score", svm_f1)
```

    F1-Score 0.8570189661023341
    


```python
cm_svm = confusion_matrix(y_test, predict_y_te_svm)
```


```python
tn, fp, fn, tp = cm_svm.ravel()
```


```python
fpr_svm = fp/(fp+tn)*100
fnr_svm = fn/(fn+tp)*100
far_svm = (fpr_svm+fnr_svm)/2
print("FAR: %0.2f" % far_svm)
```

    FAR: 13.62
    


```python
plot_cm(cm_svm)
```


![png](output_231_0.png)



```python
#finding the FPR and TPR for SVM set
fpr_te_svm, tpr_te_svm, t_te_svm = roc_curve(y_test, predict_y_te_svm)
fpr_tr_svm, tpr_tr_svm, t_tr_svm = roc_curve(y_train, predict_y_tr_svm)
auc_te_svm = auc(fpr_te_svm, tpr_te_svm)
print("AUC_SVM: ",auc_te_svm)
plot_roc_curve(fpr_tr_svm,tpr_tr_svm,fpr_te_svm, tpr_te_svm)
```

    AUC_SVM:  0.8638459891194141
    


![png](output_232_1.png)


## 6.4 Random Forest Model


```python
param_grid = {"n_estimators": [10,100,500,1000, 2000],
    "min_samples_split": [50, 80, 120, 200],
              "max_depth": [3, 5, 10, 50, 100]}
rfc = RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1)
gridCV_rfc = GridSearchCV(rfc, param_grid, cv=3, return_train_score=True, verbose=10, n_jobs=-1)
gridCV_rfc.fit(df_train, y_train)
#grid Search cv results are stored in result for future use
results_rfc = pd.DataFrame.from_dict(gridCV_rfc.cv_results_)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    4.4s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   32.8s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   58.6s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  2.9min
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:  3.8min
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:  5.2min
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:  6.8min
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:  8.5min
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed: 10.1min
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed: 11.7min
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed: 13.7min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed: 16.4min
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed: 19.1min
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 22.8min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed: 27.2min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed: 30.5min
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed: 35.6min
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed: 39.2min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 45.8min finished
    


```python
results_rfc = results_rfc.sort_values(['rank_test_score'])
results_rfc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_depth</th>
      <th>param_min_samples_split</th>
      <th>param_n_estimators</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>207.673127</td>
      <td>17.483878</td>
      <td>14.140464</td>
      <td>1.567208</td>
      <td>50</td>
      <td>50</td>
      <td>2000</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.874071</td>
      <td>0.944651</td>
      <td>0.898921</td>
      <td>0.905881</td>
      <td>0.029231</td>
      <td>1</td>
      <td>0.969374</td>
      <td>0.967461</td>
      <td>0.974767</td>
      <td>0.970534</td>
      <td>0.003093</td>
    </tr>
    <tr>
      <th>84</th>
      <td>208.767452</td>
      <td>17.792458</td>
      <td>14.420839</td>
      <td>1.246119</td>
      <td>100</td>
      <td>50</td>
      <td>2000</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.874071</td>
      <td>0.944651</td>
      <td>0.898921</td>
      <td>0.905881</td>
      <td>0.029231</td>
      <td>1</td>
      <td>0.969374</td>
      <td>0.967461</td>
      <td>0.974767</td>
      <td>0.970534</td>
      <td>0.003093</td>
    </tr>
    <tr>
      <th>63</th>
      <td>109.719585</td>
      <td>14.179158</td>
      <td>7.163603</td>
      <td>0.395994</td>
      <td>50</td>
      <td>50</td>
      <td>1000</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.871192</td>
      <td>0.944760</td>
      <td>0.899177</td>
      <td>0.905043</td>
      <td>0.030319</td>
      <td>3</td>
      <td>0.969210</td>
      <td>0.967643</td>
      <td>0.974730</td>
      <td>0.970528</td>
      <td>0.003040</td>
    </tr>
    <tr>
      <th>83</th>
      <td>108.777701</td>
      <td>11.380208</td>
      <td>10.336323</td>
      <td>0.203533</td>
      <td>100</td>
      <td>50</td>
      <td>1000</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.871192</td>
      <td>0.944760</td>
      <td>0.899177</td>
      <td>0.905043</td>
      <td>0.030319</td>
      <td>3</td>
      <td>0.969210</td>
      <td>0.967643</td>
      <td>0.974730</td>
      <td>0.970528</td>
      <td>0.003040</td>
    </tr>
    <tr>
      <th>82</th>
      <td>55.538241</td>
      <td>2.770431</td>
      <td>6.988311</td>
      <td>0.162329</td>
      <td>100</td>
      <td>50</td>
      <td>500</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.871302</td>
      <td>0.944724</td>
      <td>0.898849</td>
      <td>0.904958</td>
      <td>0.030284</td>
      <td>5</td>
      <td>0.969101</td>
      <td>0.967607</td>
      <td>0.974785</td>
      <td>0.970497</td>
      <td>0.003092</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(gridCV_rfc.best_params_)
```

    {'max_depth': 50, 'min_samples_split': 50, 'n_estimators': 2000}
    


```python
rfc= RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1, max_depth=gridCV_rfc.best_params_['max_depth'],min_samples_split=gridCV_rfc.best_params_['min_samples_split'], n_estimators=gridCV_rfc.best_params_['n_estimators'])
rfc.fit(df_train,y_train)
sig_clf_rfc = CalibratedClassifierCV(rfc, method="sigmoid")
sig_clf_rfc.fit(df_train, y_train)
predict_y_tr_rfc = sig_clf_rfc.predict(df_train)
predict_y_te_rfc = sig_clf_rfc.predict(df_test)
rfc_f1 = f1_score(y_test, predict_y_te_rfc)
print(rfc_f1)
```

    0.9405912523518647
    


```python
cm_rfc = confusion_matrix(y_test, predict_y_te_rfc)
```


```python
tn, fp, fn, tp = cm_rfc.ravel()
```


```python
fpr_rfc = fp/(fp+tn)*100
fnr_rfc = fn/(fn+tp)*100
far_rfc = (fpr_rfc+fnr_rfc)/2
print("far:",far_rfc)
```

    far: 7.676136262295199
    


```python
plot_cm(cm_rfc)
```


![png](output_241_0.png)



```python
#finding the FPR and TPR for RFC set
fpr_te_rfc, tpr_te_rfc, t_te_rfc = roc_curve(y_test, predict_y_te_rfc)
fpr_tr_rfc, tpr_tr_rfc, t_tr_rfc = roc_curve(y_train, predict_y_tr_rfc)
auc_te_rfc = auc(fpr_te_rfc, tpr_te_rfc)
print("AUC_RFC: ",auc_te_rfc)
plot_roc_curve(fpr_tr_rfc,tpr_tr_rfc,fpr_te_rfc, tpr_te_rfc)
```

    AUC_RFC:  0.923238637377048
    


![png](output_242_1.png)


## 6.5 Stacking classifier


```python
clf1 = SGDClassifier(alpha=0.0001,eta0=0.001, penalty='l2', loss='log', n_jobs = -1, max_iter=100)
clf1.fit(df_train, y_train)
sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

clf2 = SGDClassifier(alpha=0.0001,eta0=0.01, penalty='l2', loss='hinge', n_jobs = -1, max_iter=100)
clf2.fit(df_train, y_train)
sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")


clf3 = RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1, max_depth=50,min_samples_split=50, n_estimators=2000)
clf3.fit(df_train, y_train)
sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")
```


```python
alpha = [0.0001,0.001,0.01,0.1,1,10] 
best_alpha = 999
for i in alpha:
    lr = LogisticRegression(C=i)
    sclf = StackingClassifier(estimators=[("lr",sig_clf1), ("svm", sig_clf2),("RF", sig_clf3)], final_estimator=lr, n_jobs=-1)
    sclf.fit(df_train, y_train)
    print("Stacking Classifer : for the value of alpha: %f Log loss: %0.3f F1-score: %0.3f" % (i, log_loss(y_test, sclf.predict_proba(df_test)),f1_score(y_test, sclf.predict(df_test))))
    log_error =log_loss(y_test, sclf.predict_proba(df_test))
    if best_alpha > log_error:
        best_alpha = log_error
```

    Stacking Classifer : for the value of alpha: 0.000100 Log loss: 0.431 F1-score: 0.918
    Stacking Classifer : for the value of alpha: 0.001000 Log loss: 0.283 F1-score: 0.924
    Stacking Classifer : for the value of alpha: 0.010000 Log loss: 0.245 F1-score: 0.930
    Stacking Classifer : for the value of alpha: 0.100000 Log loss: 0.243 F1-score: 0.931
    Stacking Classifer : for the value of alpha: 1.000000 Log loss: 0.238 F1-score: 0.932
    Stacking Classifer : for the value of alpha: 10.000000 Log loss: 0.241 F1-score: 0.932
    


```python
lr = LogisticRegression(C=10)
sig_clf_sc = StackingClassifier(estimators=[("lr",sig_clf1), ("svm", sig_clf2),("RF", sig_clf3)], final_estimator=lr, n_jobs=-1)
sig_clf_sc.fit(df_train, y_train)
predict_y_tr_sc= sig_clf_sc.predict(df_train)
predict_y_te_sc = sig_clf_sc.predict(df_test)
sc_f1 = f1_score(y_test, predict_y_te_sc)
print(sc_f1)
```

    0.9318461928445934
    


```python
cm_sc = confusion_matrix(y_test, predict_y_te_sc)
```


```python
tn, fp, fn, tp = cm_sc.ravel()
```


```python
fpr_sc = fp/(fp+tn)*100
fnr_sc = fn/(fn+tp)*100
far_sc = (fpr_sc+fnr_sc)/2
print("far:",far_sc)
```

    far: 7.88822963937672
    


```python
plot_cm(cm_sc)
```


![png](output_250_0.png)



```python
#finding the FPR and TPR for RFC set
fpr_te_sc, tpr_te_sc, t_te_sc = roc_curve(y_test, predict_y_te_sc)
fpr_tr_sc, tpr_tr_sc, t_tr_sc = roc_curve(y_train, predict_y_tr_sc)
auc_te_sc = auc(fpr_te_sc, tpr_te_sc)
print("AUC_SC: ",auc_te_sc)
plot_roc_curve(fpr_tr_sc,tpr_tr_sc,fpr_te_sc, tpr_te_sc)
```

    AUC_SC:  0.9211177036062328
    


![png](output_251_1.png)


## 6.6 Model Evaluation


```python
x = PrettyTable()
x.field_names = ["Model", "F1 Score", "AUC","FPR %","FNR %","FAR %"]
x.add_row(["Logistic Regression", "{0:.4}".format(lr_f1), "{0:.4}".format(auc_te_lr),"%.2f" % float(fpr_lr),"%.2f" % float(fnr_lr),"%.2f" % float(far_lr)])
x.add_row(["Linear SVM", "{0:.4}".format(svm_f1), "{0:.4}".format(auc_te_svm),"%.2f" % float(fpr_svm),"%.2f" % float(fnr_svm),"%.2f" % float(far_svm)])
x.add_row(["Random Forest", "{0:.4}".format(rfc_f1), "{0:.4}".format(auc_te_rfc),"%.2f" % float(fpr_rfc),"%.2f" % float(fnr_rfc),"%.2f" % float(far_rfc)])
x.add_row(["Stacking Classifier", "{0:.4}".format(sc_f1), "{0:.4}".format(auc_te_sc),"%.2f" % float(fpr_sc),"%.2f" % float(fnr_sc),"%.2f" % float(far_sc)])
print(x)
```

    +---------------------+----------+--------+-------+-------+-------+
    |        Model        | F1 Score |  AUC   | FPR % | FNR % | FAR % |
    +---------------------+----------+--------+-------+-------+-------+
    | Logistic Regression |  0.8029  | 0.8305 |  1.43 | 32.48 | 16.95 |
    |      Linear SVM     |  0.857   | 0.8638 |  3.41 | 23.82 | 13.62 |
    |    Random Forest    |  0.9406  | 0.9232 |  7.09 |  8.26 |  7.68 |
    | Stacking Classifier |  0.9318  | 0.9211 |  5.11 | 10.67 |  7.89 |
    +---------------------+----------+--------+-------+-------+-------+
    

# 7. Machine Learning Models (One Hot Encoding)

## 7.1 Reading Train and Test data


```python
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
print(train_data.shape)
train_data.head()
```

    (82332, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000011</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>496</td>
      <td>0</td>
      <td>90909.0902</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000008</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1762</td>
      <td>0</td>
      <td>125000.0003</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000005</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>1068</td>
      <td>0</td>
      <td>200000.0051</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.000006</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>900</td>
      <td>0</td>
      <td>166666.6608</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000010</td>
      <td>udp</td>
      <td>-</td>
      <td>INT</td>
      <td>2</td>
      <td>0</td>
      <td>2126</td>
      <td>0</td>
      <td>100000.0025</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_train = train_data[list(col_set)]
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sttl</th>
      <th>ct_state_ttl</th>
      <th>stcpb</th>
      <th>trans_depth</th>
      <th>dttl</th>
      <th>proto</th>
      <th>dmean</th>
      <th>ct_dst_sport_ltm</th>
      <th>dinpkt</th>
      <th>ackdat</th>
      <th>...</th>
      <th>response_body_len</th>
      <th>djit</th>
      <th>swin</th>
      <th>dload</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_dport_ltm</th>
      <th>ct_dst_ltm</th>
      <th>is_ftp_login</th>
      <th>dloss</th>
      <th>ct_dst_src_ltm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>udp</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>udp</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>udp</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>udp</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>udp</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
cat_features = df_train.select_dtypes(include=['category', object]).columns
cat_features
```




    Index(['proto', 'state', 'service'], dtype='object')




```python
ohe = OneHotEncoder()
cat_f = pd.DataFrame(ohe.fit_transform(train_data[cat_features]).toarray())
```


```python
cat_f.shape
```




    (82332, 151)




```python
df_train = df_train.drop(cat_features, axis=1)
df_train.shape
```




    (82332, 27)




```python
df_train = df_train.join(cat_f)
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sttl</th>
      <th>ct_state_ttl</th>
      <th>stcpb</th>
      <th>trans_depth</th>
      <th>dttl</th>
      <th>dmean</th>
      <th>ct_dst_sport_ltm</th>
      <th>dinpkt</th>
      <th>ackdat</th>
      <th>sloss</th>
      <th>...</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 178 columns</p>
</div>




```python
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=150).fit_transform(df_train, train_data['label'])
```


```python
X_new.shape
```




    (82332, 150)




```python
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")
print(test_data.shape)
test_data.head()
```

    (175341, 45)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>dur</th>
      <th>proto</th>
      <th>service</th>
      <th>state</th>
      <th>spkts</th>
      <th>dpkts</th>
      <th>sbytes</th>
      <th>dbytes</th>
      <th>rate</th>
      <th>...</th>
      <th>ct_dst_sport_ltm</th>
      <th>ct_dst_src_ltm</th>
      <th>is_ftp_login</th>
      <th>ct_ftp_cmd</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_ltm</th>
      <th>ct_srv_dst</th>
      <th>is_sm_ips_ports</th>
      <th>attack_cat</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.121478</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>6</td>
      <td>4</td>
      <td>258</td>
      <td>172</td>
      <td>74.087490</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.649902</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>14</td>
      <td>38</td>
      <td>734</td>
      <td>42014</td>
      <td>78.473372</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.623129</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>8</td>
      <td>16</td>
      <td>364</td>
      <td>13186</td>
      <td>14.170161</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.681642</td>
      <td>tcp</td>
      <td>ftp</td>
      <td>FIN</td>
      <td>12</td>
      <td>12</td>
      <td>628</td>
      <td>770</td>
      <td>13.677108</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.449454</td>
      <td>tcp</td>
      <td>-</td>
      <td>FIN</td>
      <td>10</td>
      <td>6</td>
      <td>534</td>
      <td>268</td>
      <td>33.373826</td>
      <td>...</td>
      <td>1</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>39</td>
      <td>0</td>
      <td>Normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_test = test_data[list(col_set)]
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sttl</th>
      <th>ct_state_ttl</th>
      <th>stcpb</th>
      <th>trans_depth</th>
      <th>dttl</th>
      <th>proto</th>
      <th>dmean</th>
      <th>ct_dst_sport_ltm</th>
      <th>dinpkt</th>
      <th>ackdat</th>
      <th>...</th>
      <th>response_body_len</th>
      <th>djit</th>
      <th>swin</th>
      <th>dload</th>
      <th>ct_flw_http_mthd</th>
      <th>ct_src_dport_ltm</th>
      <th>ct_dst_ltm</th>
      <th>is_ftp_login</th>
      <th>dloss</th>
      <th>ct_dst_src_ltm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>252</td>
      <td>0</td>
      <td>621772692</td>
      <td>0</td>
      <td>254</td>
      <td>tcp</td>
      <td>43</td>
      <td>1</td>
      <td>8.375000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>11.830604</td>
      <td>255</td>
      <td>8495.365234</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62</td>
      <td>1</td>
      <td>1417884146</td>
      <td>0</td>
      <td>252</td>
      <td>tcp</td>
      <td>1106</td>
      <td>1</td>
      <td>15.432865</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>1387.778330</td>
      <td>255</td>
      <td>503571.312500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62</td>
      <td>1</td>
      <td>2116150707</td>
      <td>0</td>
      <td>252</td>
      <td>tcp</td>
      <td>824</td>
      <td>1</td>
      <td>102.737203</td>
      <td>0.050439</td>
      <td>...</td>
      <td>0</td>
      <td>11420.926230</td>
      <td>255</td>
      <td>60929.230470</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62</td>
      <td>1</td>
      <td>1107119177</td>
      <td>0</td>
      <td>252</td>
      <td>tcp</td>
      <td>64</td>
      <td>1</td>
      <td>90.235726</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>4991.784669</td>
      <td>255</td>
      <td>3358.622070</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>254</td>
      <td>1</td>
      <td>2436137549</td>
      <td>0</td>
      <td>252</td>
      <td>tcp</td>
      <td>45</td>
      <td>1</td>
      <td>75.659602</td>
      <td>0.057234</td>
      <td>...</td>
      <td>0</td>
      <td>115.807000</td>
      <td>255</td>
      <td>3987.059814</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
cat_feature = df_test.select_dtypes(include=['category', object]).columns
cat_feature
```




    Index(['proto', 'state', 'service'], dtype='object')




```python
ohe = OneHotEncoder()
cat_f_t = pd.DataFrame(ohe.fit_transform(df_test[cat_feature]).toarray())
```


```python
cat_f_t.shape
```




    (175341, 155)




```python
df_test = df_test.drop(cat_features, axis=1)
df_test.shape
```




    (175341, 27)




```python
df_test = df_test.join(cat_f_t)
```


```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sttl</th>
      <th>ct_state_ttl</th>
      <th>stcpb</th>
      <th>trans_depth</th>
      <th>dttl</th>
      <th>dmean</th>
      <th>ct_dst_sport_ltm</th>
      <th>dinpkt</th>
      <th>ackdat</th>
      <th>sloss</th>
      <th>...</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>150</th>
      <th>151</th>
      <th>152</th>
      <th>153</th>
      <th>154</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>252</td>
      <td>0</td>
      <td>621772692</td>
      <td>0</td>
      <td>254</td>
      <td>43</td>
      <td>1</td>
      <td>8.375000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62</td>
      <td>1</td>
      <td>1417884146</td>
      <td>0</td>
      <td>252</td>
      <td>1106</td>
      <td>1</td>
      <td>15.432865</td>
      <td>0.000000</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62</td>
      <td>1</td>
      <td>2116150707</td>
      <td>0</td>
      <td>252</td>
      <td>824</td>
      <td>1</td>
      <td>102.737203</td>
      <td>0.050439</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62</td>
      <td>1</td>
      <td>1107119177</td>
      <td>0</td>
      <td>252</td>
      <td>64</td>
      <td>1</td>
      <td>90.235726</td>
      <td>0.000000</td>
      <td>1</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>254</td>
      <td>1</td>
      <td>2436137549</td>
      <td>0</td>
      <td>252</td>
      <td>45</td>
      <td>1</td>
      <td>75.659602</td>
      <td>0.057234</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 182 columns</p>
</div>




```python
from sklearn.feature_selection import SelectKBest, chi2
X_new_test = SelectKBest(chi2, k=150).fit_transform(df_test, test_data['label'])
```


```python
X_new_test.shape
```




    (175341, 150)



### 7.1.1 Standardize the data


```python
std_scaler = preprocessing.MinMaxScaler()
std_scaler.fit(X_new)
x_scaled = std_scaler.transform(X_new)
df_train = pd.DataFrame(x_scaled)
x_scaled_test = std_scaler.transform(X_new_test)
df_test = pd.DataFrame(x_scaled_test)
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.996078</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.996078</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.996078</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.996078</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.996078</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 150 columns</p>
</div>




```python
y_train = train_data['label']
y_test = test_data['label']
print("train data shape", df_train.shape, y_train.shape)
print("test data shape", df_test.shape, y_test.shape)
```

    train data shape (82332, 150) (82332,)
    test data shape (175341, 150) (175341,)
    

## 7.2 Logistic Regression Model


```python
prams={
    'alpha':[10 ** x for x in range(-4, 1)],
     'max_iter':[5, 10, 20, 50, 100],
    'eta0': [10 ** x for x in range(-4, 1)]
}
lr_cfl=GridSearchCV(SGDClassifier(penalty='l2', loss='log', n_jobs = -1), param_grid=prams,verbose=10,n_jobs=-1)
lr_cfl.fit(df_train,y_train)
```

    Fitting 5 folds for each of 125 candidates, totalling 625 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    1.2s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    2.5s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    3.3s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    5.1s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    6.7s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:    8.6s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   11.0s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   12.9s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   15.7s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:   18.4s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   20.6s
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:   23.6s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   26.5s
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:   29.5s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   32.2s
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:   35.6s
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:   38.9s
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:   43.4s
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:   47.8s
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:   52.0s
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:   56.2s
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-1)]: Done 625 out of 625 | elapsed:  1.8min finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SGDClassifier(alpha=0.0001, average=False,
                                         class_weight=None, early_stopping=False,
                                         epsilon=0.1, eta0=0.0, fit_intercept=True,
                                         l1_ratio=0.15, learning_rate='optimal',
                                         loss='log', max_iter=1000,
                                         n_iter_no_change=5, n_jobs=-1,
                                         penalty='l2', power_t=0.5,
                                         random_state=None, shuffle=True, tol=0.001,
                                         validation_fraction=0.1, verbose=0,
                                         warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                             'eta0': [0.0001, 0.001, 0.01, 0.1, 1],
                             'max_iter': [5, 10, 20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=10)




```python
results = pd.DataFrame.from_dict(lr_cfl.cv_results_)
results = results.sort_values(['rank_test_score'])
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_alpha</th>
      <th>param_eta0</th>
      <th>param_max_iter</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>1.402684</td>
      <td>0.131904</td>
      <td>0.020544</td>
      <td>0.008756</td>
      <td>0.0001</td>
      <td>0.01</td>
      <td>50</td>
      <td>{'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 50}</td>
      <td>0.922390</td>
      <td>0.986458</td>
      <td>0.870643</td>
      <td>0.812037</td>
      <td>0.802745</td>
      <td>0.878855</td>
      <td>0.068991</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.436356</td>
      <td>0.037683</td>
      <td>0.018948</td>
      <td>0.002602</td>
      <td>0.0001</td>
      <td>0.1</td>
      <td>10</td>
      <td>{'alpha': 0.0001, 'eta0': 0.1, 'max_iter': 10}</td>
      <td>0.894941</td>
      <td>0.968057</td>
      <td>0.888498</td>
      <td>0.819750</td>
      <td>0.804992</td>
      <td>0.875248</td>
      <td>0.058639</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.397862</td>
      <td>0.072990</td>
      <td>0.018155</td>
      <td>0.001464</td>
      <td>0.0001</td>
      <td>0.01</td>
      <td>10</td>
      <td>{'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 10}</td>
      <td>0.927552</td>
      <td>0.971276</td>
      <td>0.859893</td>
      <td>0.819446</td>
      <td>0.792785</td>
      <td>0.874190</td>
      <td>0.066485</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.544993</td>
      <td>0.075344</td>
      <td>0.016157</td>
      <td>0.004106</td>
      <td>0.0001</td>
      <td>0.01</td>
      <td>20</td>
      <td>{'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 20}</td>
      <td>0.924698</td>
      <td>0.933928</td>
      <td>0.871918</td>
      <td>0.818414</td>
      <td>0.821693</td>
      <td>0.874130</td>
      <td>0.048973</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.420799</td>
      <td>0.126149</td>
      <td>0.015758</td>
      <td>0.001716</td>
      <td>0.0001</td>
      <td>0.1</td>
      <td>20</td>
      <td>{'alpha': 0.0001, 'eta0': 0.1, 'max_iter': 20}</td>
      <td>0.934111</td>
      <td>0.945102</td>
      <td>0.862201</td>
      <td>0.813069</td>
      <td>0.795822</td>
      <td>0.870061</td>
      <td>0.060918</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(lr_cfl.best_params_)
```

    {'alpha': 0.0001, 'eta0': 0.01, 'max_iter': 50}
    


```python
logisticR=SGDClassifier(alpha=lr_cfl.best_params_['alpha'],eta0=lr_cfl.best_params_['eta0'], penalty='l2', loss='log', n_jobs = -1, max_iter=lr_cfl.best_params_['max_iter'])
logisticR.fit(df_train,y_train)
sig_clf = CalibratedClassifierCV(logisticR, method="sigmoid")
sig_clf.fit(df_train, y_train)
predict_y_tr_lr = sig_clf.predict(df_train)
predict_y_te_lr = sig_clf.predict(df_test)
lr_f1 = f1_score(y_test, predict_y_te_lr)
print(lr_f1)
```

    0.9306771659712835
    


```python
cm_lr = confusion_matrix(y_test, predict_y_te_lr)
```


```python
tn, fp, fn, tp = cm_lr.ravel()
```


```python
fpr_lr = (fp/(fp+tn))*100
fnr_lr = (fn/(fn+tp))*100
far_lr = (fpr_lr+fnr_lr)/2
print("FAR:",far_lr)
```

    FAR: 15.533255051251693
    


```python
def plot_cm(cm):
    sns.heatmap(cm, annot=True, cmap=sns.light_palette("blue"), fmt="g")
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    plt.show()
```


```python
plot_cm(cm_lr)
```


![png](output_289_0.png)



```python
from sklearn.metrics import roc_curve, auc
def plot_roc_curve(fpr_tr, tpr_tr,fpr_te, tpr_te):
    '''
    plot the ROC curve for the FPR and TPR value
    '''
    plt.plot(fpr_te, tpr_te, 'k.-', color='orange', label='ROC_test AUC:%.3f'% auc(fpr_te, tpr_te))
    plt.plot(fpr_tr, tpr_tr, 'k.-', color='green', label='ROC_train AUC:%.3f'% auc(fpr_tr, tpr_tr))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
```


```python
#finding the FPR and TPR for logistic reg model set
fpr_te_lr, tpr_te_lr, t_te_lr = roc_curve(y_test, predict_y_te_lr)
fpr_tr_lr, tpr_tr_lr, t_tr_lr = roc_curve(y_train, predict_y_tr_lr)
auc_te_lr = auc(fpr_te_lr, tpr_te_lr)
print("AUC_LR: ",auc_te_lr)
plot_roc_curve(fpr_tr_lr,tpr_tr_lr,fpr_te_lr, tpr_te_lr)
```

    AUC_LR:  0.8446674494874831
    


![png](output_291_1.png)


## 7.3 Support Vector Machine Model


```python
prams={
    'alpha':[10 ** x for x in range(-5, 1)],
     'max_iter':[5, 10, 20, 50, 100],
    'eta0': [10 ** x for x in range(-5, 1)]
}
svm_cfl=GridSearchCV(SGDClassifier(penalty='l2', loss='hinge', n_jobs = -1), param_grid=prams,verbose=10,n_jobs=-1)
svm_cfl.fit(df_train,y_train)
```

    Fitting 5 folds for each of 180 candidates, totalling 900 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    7.8s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   11.9s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   17.5s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:   23.6s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   26.7s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:   35.4s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   38.6s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   47.8s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   51.2s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:  2.3min
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:  2.4min
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:  2.5min
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:  2.6min
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  2.9min
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:  3.2min
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:  3.3min
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:  3.4min
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed:  3.6min
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed:  3.8min
    [Parallel(n_jobs=-1)]: Done 669 tasks      | elapsed:  3.9min
    [Parallel(n_jobs=-1)]: Done 706 tasks      | elapsed:  4.1min
    [Parallel(n_jobs=-1)]: Done 745 tasks      | elapsed:  4.2min
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  4.4min
    [Parallel(n_jobs=-1)]: Done 825 tasks      | elapsed:  4.5min
    [Parallel(n_jobs=-1)]: Done 866 tasks      | elapsed:  4.7min
    [Parallel(n_jobs=-1)]: Done 900 out of 900 | elapsed:  4.8min finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SGDClassifier(alpha=0.0001, average=False,
                                         class_weight=None, early_stopping=False,
                                         epsilon=0.1, eta0=0.0, fit_intercept=True,
                                         l1_ratio=0.15, learning_rate='optimal',
                                         loss='hinge', max_iter=1000,
                                         n_iter_no_change=5, n_jobs=-1,
                                         penalty='l2', power_t=0.5,
                                         random_state=None, shuffle=True, tol=0.001,
                                         validation_fraction=0.1, verbose=0,
                                         warm_start=False),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1, 1],
                             'eta0': [1e-05, 0.0001, 0.001, 0.01, 0.1, 1],
                             'max_iter': [5, 10, 20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=10)




```python
print(svm_cfl.best_params_)
```

    {'alpha': 1e-05, 'eta0': 1e-05, 'max_iter': 100}
    


```python
svm=SGDClassifier(alpha=svm_cfl.best_params_['alpha'],eta0=svm_cfl.best_params_['eta0'], penalty='l2', loss='hinge', n_jobs = -1, max_iter=svm_cfl.best_params_['max_iter'])
svm.fit(df_train,y_train)
sig_clf_svm = CalibratedClassifierCV(svm, method="sigmoid")
sig_clf_svm.fit(df_train, y_train)
predict_y_tr_svm = sig_clf.predict(df_train)
predict_y_te_svm = sig_clf_svm.predict(df_test)
svm_f1 = f1_score(y_test, predict_y_te_svm)
print("F1-Score", svm_f1)
```

    F1-Score 0.8934157791398737
    


```python
cm_svm = confusion_matrix(y_test, predict_y_te_svm)
```


```python
tn, fp, fn, tp = cm_svm.ravel()
```


```python
fpr_svm = fp/(fp+tn)*100
fnr_svm = fn/(fn+tp)*100
far_svm = (fpr_svm+fnr_svm)/2
print("FAR:", far_svm)
```

    FAR: 22.99486192477259
    


```python
plot_cm(cm_svm)
```


![png](output_299_0.png)



```python
#finding the FPR and TPR for SVM set
fpr_te_svm, tpr_te_svm, t_te_svm = roc_curve(y_test, predict_y_te_svm)
fpr_tr_svm, tpr_tr_svm, t_tr_svm = roc_curve(y_train, predict_y_tr_svm)
auc_te_svm = auc(fpr_te_svm, tpr_te_svm)
print("AUC_SVM: ",auc_te_svm)
plot_roc_curve(fpr_tr_svm,tpr_tr_svm,fpr_te_svm, tpr_te_svm)
```

    AUC_SVM:  0.7700513807522741
    


![png](output_300_1.png)


## 7.4 Random Forest Model


```python
param_grid = {"n_estimators": [10,100,500,1000, 2000],
    "min_samples_split": [50, 80, 120, 200],
              "max_depth": [3, 5, 10, 50, 100]}
rfc = RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1)
gridCV_rfc = GridSearchCV(rfc, param_grid, cv=3, verbose=10, n_jobs=-1)
gridCV_rfc.fit(df_train, y_train)
#grid Search cv results are stored in result for future use
results_rfc = pd.DataFrame.from_dict(gridCV_rfc.cv_results_)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    7.4s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   31.7s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   59.4s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  2.7min
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:  3.4min
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:  4.5min
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:  5.9min
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:  7.8min
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:  9.6min
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed: 11.4min
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed: 13.3min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed: 16.4min
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed: 19.4min
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 23.6min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed: 29.2min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed: 33.0min
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed: 39.2min
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed: 43.4min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 51.2min finished
    


```python
results_rfc = results_rfc.sort_values(['rank_test_score'])
results_rfc.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_depth</th>
      <th>param_min_samples_split</th>
      <th>param_n_estimators</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>142.432115</td>
      <td>18.089001</td>
      <td>9.515215</td>
      <td>0.412303</td>
      <td>100</td>
      <td>50</td>
      <td>1000</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.926541</td>
      <td>0.942355</td>
      <td>0.896626</td>
      <td>0.921841</td>
      <td>0.018963</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>152.819562</td>
      <td>14.167800</td>
      <td>13.583502</td>
      <td>1.334664</td>
      <td>50</td>
      <td>50</td>
      <td>1000</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.926541</td>
      <td>0.942282</td>
      <td>0.896444</td>
      <td>0.921756</td>
      <td>0.019017</td>
      <td>2</td>
    </tr>
    <tr>
      <th>64</th>
      <td>302.903203</td>
      <td>23.717455</td>
      <td>18.424916</td>
      <td>2.786622</td>
      <td>50</td>
      <td>50</td>
      <td>2000</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.926468</td>
      <td>0.942355</td>
      <td>0.896189</td>
      <td>0.921671</td>
      <td>0.019150</td>
      <td>3</td>
    </tr>
    <tr>
      <th>84</th>
      <td>270.264782</td>
      <td>25.018238</td>
      <td>16.924602</td>
      <td>1.252264</td>
      <td>100</td>
      <td>50</td>
      <td>2000</td>
      <td>{'max_depth': 100, 'min_samples_split': 50, 'n...</td>
      <td>0.926541</td>
      <td>0.942282</td>
      <td>0.896116</td>
      <td>0.921647</td>
      <td>0.019163</td>
      <td>4</td>
    </tr>
    <tr>
      <th>62</th>
      <td>81.389524</td>
      <td>6.615336</td>
      <td>9.358697</td>
      <td>1.878683</td>
      <td>50</td>
      <td>50</td>
      <td>500</td>
      <td>{'max_depth': 50, 'min_samples_split': 50, 'n_...</td>
      <td>0.926396</td>
      <td>0.942501</td>
      <td>0.895934</td>
      <td>0.921610</td>
      <td>0.019310</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(gridCV_rfc.best_params_)
```

    {'max_depth': 100, 'min_samples_split': 50, 'n_estimators': 1000}
    


```python
rfc= RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1, max_depth=gridCV_rfc.best_params_['max_depth'],min_samples_split=gridCV_rfc.best_params_['min_samples_split'], n_estimators=gridCV_rfc.best_params_['n_estimators'])
rfc.fit(df_train,y_train)
sig_clf_rfc = CalibratedClassifierCV(rfc, method="sigmoid")
sig_clf_rfc.fit(df_train, y_train)
predict_y_tr_rfc = sig_clf_rfc.predict(df_train)
predict_y_te_rfc = sig_clf_rfc.predict(df_test)
rfc_f1 = f1_score(y_test, predict_y_te_rfc)
print(rfc_f1)
```

    0.9523705581088892
    


```python
cm_rfc = confusion_matrix(y_test, predict_y_te_rfc)
```


```python
tn, fp, fn, tp = cm_rfc.ravel()
```


```python
fpr_rfc = fp/(fp+tn)*100
fnr_rfc = fn/(fn+tp)*100
far_rfc = (fpr_rfc+fnr_rfc)/2
print("far:",far_rfc)
```

    far: 9.791693438190922
    


```python
plot_cm(cm_rfc)
```


![png](output_309_0.png)



```python
#finding the FPR and TPR for RFC set
fpr_te_rfc, tpr_te_rfc, t_te_rfc = roc_curve(y_test, predict_y_te_rfc)
fpr_tr_rfc, tpr_tr_rfc, t_tr_rfc = roc_curve(y_train, predict_y_tr_rfc)
auc_te_rfc = auc(fpr_te_rfc, tpr_te_rfc)
print("AUC_RFC: ",auc_te_rfc)
plot_roc_curve(fpr_tr_rfc,tpr_tr_rfc,fpr_te_rfc, tpr_te_rfc)
```

    AUC_RFC:  0.9020830656180907
    


![png](output_310_1.png)


## 7.5 Stacking classifier


```python
clf1 = SGDClassifier(alpha=0.0001,eta0=0.01, penalty='l2', loss='log', n_jobs = -1, max_iter=50)
clf1.fit(df_train, y_train)
sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

clf2 = SGDClassifier(alpha=1e-05,eta0=1e-05, penalty='l2', loss='hinge', n_jobs = -1, max_iter=100)
clf2.fit(df_train, y_train)
sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")


clf3 = RandomForestClassifier(criterion='gini', random_state=42, n_jobs=-1, max_depth=100,min_samples_split=50, n_estimators=1000)
clf3.fit(df_train, y_train)
sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")
```


```python
alpha = [0.0001,0.001,0.01,0.1,1,10] 
best_alpha = 999
for i in alpha:
    lr = LogisticRegression(C=i)
    sclf = StackingClassifier(estimators=[("lr",sig_clf1), ("svm", sig_clf2),("RF", sig_clf3)], final_estimator=lr, n_jobs=-1)
    sclf.fit(df_train, y_train)
    print("Stacking Classifer : for the value of alpha: %f Log loss: %0.3f F1-score: %0.3f" % (i, log_loss(y_test, sclf.predict_proba(df_test)),f1_score(y_test, sclf.predict(df_test))))
```

    Stacking Classifer : for the value of alpha: 0.000100 Log loss: 0.402 F1-score: 0.933
    Stacking Classifer : for the value of alpha: 0.001000 Log loss: 0.258 F1-score: 0.938
    Stacking Classifer : for the value of alpha: 0.010000 Log loss: 0.233 F1-score: 0.942
    Stacking Classifer : for the value of alpha: 0.100000 Log loss: 0.222 F1-score: 0.943
    Stacking Classifer : for the value of alpha: 1.000000 Log loss: 0.225 F1-score: 0.943
    Stacking Classifer : for the value of alpha: 10.000000 Log loss: 0.225 F1-score: 0.943
    


```python
lr = LogisticRegression(C=10)
sig_clf_sc = StackingClassifier(estimators=[("lr",sig_clf1), ("svm", sig_clf2),("RF", sig_clf3)], final_estimator=lr, n_jobs=-1)
sig_clf_sc.fit(df_train, y_train)
predict_y_tr_sc= sig_clf_sc.predict(df_train)
predict_y_te_sc = sig_clf_sc.predict(df_test)
sc_f1 = f1_score(y_test, predict_y_te_sc)
print(sc_f1)
```

    0.94252901026609
    


```python
cm_sc = confusion_matrix(y_test, predict_y_te_sc)
```


```python
tn, fp, fn, tp = cm_sc.ravel()
```


```python
fpr_sc = fp/(fp+tn)*100
fnr_sc = fn/(fn+tp)*100
far_sc = (fpr_sc+fnr_sc)/2
print("far:",far_sc)
```

    far: 12.412772418651475
    


```python
plot_cm(cm_sc)
```


![png](output_318_0.png)



```python
#finding the FPR and TPR for RFC set
fpr_te_sc, tpr_te_sc, t_te_sc = roc_curve(y_test, predict_y_te_sc)
fpr_tr_sc, tpr_tr_sc, t_tr_sc = roc_curve(y_train, predict_y_tr_sc)
auc_te_sc = auc(fpr_te_sc, tpr_te_sc)
print("AUC_SC: ",auc_te_sc)
plot_roc_curve(fpr_tr_sc,tpr_tr_sc,fpr_te_sc, tpr_te_sc)
```

    AUC_SC:  0.8758722758134853
    


![png](output_319_1.png)


# 7.6. Model Evaluation


```python
x = PrettyTable()
x.field_names = ["Model", "F1 Score", "AUC","FPR %","FNR %","FAR %"]
x.add_row(["Logistic Regression", "{0:.4}".format(lr_f1), "{0:.4}".format(auc_te_lr),"%.2f" % float(fpr_lr),"%.2f" % float(fnr_lr),"%.2f" % float(far_lr)])
x.add_row(["Linear SVM", "{0:.4}".format(svm_f1), "{0:.4}".format(auc_te_svm),"%.2f" % float(fpr_svm),"%.2f" % float(fnr_svm),"%.2f" % float(far_svm)])
x.add_row(["Random Forest", "{0:.4}".format(rfc_f1), "{0:.4}".format(auc_te_rfc),"%.2f" % float(fpr_rfc),"%.2f" % float(fnr_rfc),"%.2f" % float(far_rfc)])
x.add_row(["Stacking Classifier", "{0:.4}".format(sc_f1), "{0:.4}".format(auc_te_sc),"%.2f" % float(fpr_sc),"%.2f" % float(fnr_sc),"%.2f" % float(far_sc)])
print(x)
```

    +---------------------+----------+--------+-------+-------+-------+
    |        Model        | F1 Score |  AUC   | FPR % | FNR % | FAR % |
    +---------------------+----------+--------+-------+-------+-------+
    | Logistic Regression |  0.9307  | 0.8447 |  30.6 |  0.47 | 15.53 |
    |      Linear SVM     |  0.8934  | 0.7701 | 43.03 |  2.96 | 22.99 |
    |    Random Forest    |  0.9524  | 0.9021 | 18.29 |  1.29 |  9.79 |
    | Stacking Classifier |  0.9425  | 0.8759 | 23.99 |  0.84 | 12.41 |
    +---------------------+----------+--------+-------+-------+-------+
    

# 8. Conclusion

- In this work, i have implemented Association rule based feature mining technique for features selection. i have used mode() selection of point for each attribute this reduces the processing time for identifying frequent value and Association Rule Mining (ARM) customized to find the highest ranked features by removing irrelevant or noisy features. Final features are than input to the machine learning model. 
- To differentiate between normal and attack i have used Logistic regression, linear SVM Random forest and stacking are used.
- The experimental results show that, Stacking classifier model performed well compared to other model. it has **94% of f1 measure** and **7.3% of False Alarm Rate** which is significantly lower than other models.
- Also we can understand that Response encoding preformed well compared to other categorical data encoding techniques.
- **False negative rate** also very much low in this case the cost associated with False Negative should be very low. because an intrusion cannot be predicted as normal.
- F1-score is weighted average of precision and recall. Precision is the measure of the correctly identified intrusion from all the predicted intrusion. Recall is the measure of the correctly identified intrusion from all the actual labeled intrusion.
