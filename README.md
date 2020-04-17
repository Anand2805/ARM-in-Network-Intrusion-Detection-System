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

# Model Evaluation


    +---------------------+----------+--------+-------+-------+-------+
    |        Model        | F1 Score |  AUC   | FPR % | FNR % | FAR % |
    +---------------------+----------+--------+-------+-------+-------+
    | Logistic Regression |  0.9307  | 0.8447 |  30.6 |  0.47 | 15.53 |
    |      Linear SVM     |  0.8934  | 0.7701 | 43.03 |  2.96 | 22.99 |
    |    Random Forest    |  0.9524  | 0.9021 | 18.29 |  1.29 |  9.79 |
    | Stacking Classifier |  0.9425  | 0.8759 | 23.99 |  0.84 | 12.41 |
    +---------------------+----------+--------+-------+-------+-------+
    

# Conclusion

- In this work, i have implemented Association rule based feature mining technique for features selection. i have used mode() selection of point for each attribute this reduces the processing time for identifying frequent value and Association Rule Mining (ARM) customized to find the highest ranked features by removing irrelevant or noisy features. Final features are than input to the machine learning model. 
- To differentiate between normal and attack i have used Logistic regression, linear SVM Random forest and stacking are used.
- The experimental results show that, Stacking classifier model performed well compared to other model. it has **94% of f1 measure** and **7.3% of False Alarm Rate** which is significantly lower than other models.
- Also we can understand that Response encoding preformed well compared to other categorical data encoding techniques.
- **False negative rate** also very much low in this case the cost associated with False Negative should be very low. because an intrusion cannot be predicted as normal.
- F1-score is weighted average of precision and recall. Precision is the measure of the correctly identified intrusion from all the predicted intrusion. Recall is the measure of the correctly identified intrusion from all the actual labeled intrusion.

# References

1. The [Apriori algorithm](https://web.stanford.edu/class/cs345d-01/rl/ar-mining.pdf) was proposed by Agrawal and Srikant in 1994.
2. [Data Mining and Data Warehousing: Principles and Practical Techniques By Parteek Bhatia](https://books.google.co.in/books?id=bF6NDwAAQBAJ&amp;lpg=PA252&amp;ots=pMwClCpzFd&amp;dq=The%20join%20step%3A%20To%20find%20Lk%2C%20a%20set%20of%20candidates%20k-itemsets%20is%20generated%20by%20joining%20Lk-1%20with%20itself&amp;pg=PP1#v=onepage&amp;q=The%20join%20step:%20To%20find%20Lk,%20a%20set%20of%20candidates%20k-itemsets%20is%20generated%20by%20joining%20Lk-1%20with%20itself&amp;f=false), 2019 — Cambridge University Press
3. [Response coding for categorical data](https://medium.com/@thewingedwolf.winterfell/response-coding-for-categorical-data-7bb8916c6dc1)
4. IDS - [https://en.wikipedia.org/wiki/Intrusion\_detection\_system](https://en.wikipedia.org/wiki/Intrusion_detection_system)
5. [Network Intrusion Detection with a Hashing Based Apriori Algorithm Using Hadoop MapReduce](https://doi.org/10.3390/computers8040086) - by Nureni Ayofe Azeez,Tolulope Jide Ayemobola,Sanjay Misra,Rytis Maskeliūnas and Robertas Damaševičius
6. [Finding Frequent Itemsets using Apriori Algorihm to Detect Intrusions in Large Dataset](https://www.ijcait.com/IJCAIT/61/611.pdf) - Kamini Nalavade, B.B. Meshram
7. Applied AI course - [https://www.appliedaicourse.com/](https://www.appliedaicourse.com/)
