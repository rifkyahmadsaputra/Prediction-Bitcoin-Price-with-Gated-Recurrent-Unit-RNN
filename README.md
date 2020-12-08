# Prediction bitcoin price from 2014 to 2020 with Gated Recurrent Unit (RNN)

## Introduction
<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this project I created prediction model for predict bitcoin price with Gated Recurrent Unit Model. GRU is a gating mechanism in recurrent neural networks (RNN) similar to a long short-term memory (LSTM), GRU have more simple computation and faster than LSTM because have fewer number of gates.
 <br></br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dataset used in this model prediction is bitcoin price from 2014 to 2020 were recorded every day. I obtained dataset from kaggle, you can check dataset in this link : https://www.kaggle.com/khalilbrick/bitcoin-data-from-2014-to-2020. Prediction results from the prediction model will evaluate with RMSE and MAPE, to find out how optimal that the prediction models.
</p>

## Results
<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Results from prediction model for predict bitcoin price with Gated Recurrent Unit Model is quite good, the model can predict bitcoin price well. The following is a model loss from model that  I built :
</p>

<p align="center"> 
 <img src="images/model loss.png" /> 
 <br></br>
 Model loss
</p>

<p align = "justify"> 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Can be seen from the picture above, train and val loss are not fluctuating and overfitting, so it can be said the model prediction can predict bitcoin price quite well. After the model is compiled, that model will be implemented on test set which has been partitioned before.Then, evaluate the prediction model with RMSE and MAPE
</p>

<p align="center"> 
 <img src="images/rmse and mape.png" /> 
 <br></br>
 Root Mean Square Error and Mean Absolute Percentage Error
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Based on the results obtained, the model is quite good in making predictions of bitcoin, with result of RMSE is 1994.7004 and MAPE is 18.5992% (Accuracy : 81.4%). After that, here is a table and graph from comparing the results of prediction with actual test data
</p>

<p align="center"> 
 Table comparasion data actual and data prediction
  <br></br>
 <img src="images/table comparasion data actual and data prediction.png" />
  <br></br>
 <img src="images/graph comparasion data actual and data prediction.png" /> 
  <br></br>
 Graph comparasion data actual and data prediction
 <br></br>
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Can be seen from the graph, prediction results can can follow the movement from test set. But, there are some points of prediction result which is quite big different from the actual sets (for example at the first or initial data point).
</p>
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Summary
The following is a summary of what was done in this project:

#### - Data Preprocessing
<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; First, import some libraries that will be used and store data into dataframe
</p>
<p align="center"> 
 <img src="images/data.png" /> 
 <img src="images/information of data.png" /> 
 Informations of data
 <br></br>
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Based on picture above, we know dataset contains 2123 rows data and 6 columns. It can be seen that the dataset is fairly clean and also has no null values. After that, I do features selection, features I used in this model prediction is ‘Date’ and ‘Close’ which can represent bitcoin price movements every day.
</p>
<p align="center"> 
 <img src="images/bitcoin price graph.png" /> 
 Bitcoin price graph
 <br></br>
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Afterthat, I do min-max normalization and data partition. Dataset separated into 3 partitions that is training, validation and test sets with composision : 70% for training, 10% for validation and 20% test sets.
</p>
<p align="center"> 
 <img src="images/training data graph.png" /> 
 Training data graph
 <img src="images/validation data graph.png" /> 
 Validation data graph
 <img src="images/test data graph.png" /> 
 Test data graph
 <br></br>
</p>
<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Afterthat, I change format dataset from sequential into sliding windows format. In this prediction model, I used the sliding windows with two lag or length.
</p>
<br><br>

#### Predict Bitcoin Price with Gated Recurrent Unit
<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Prediction model architecture that implemented in this task is model prediction with three gru layer. Each layer GRU have dropout layer(the dropout rate is set to 20%) which aims to prevent overfitting on model. Then, do some evaluating for determine best hyperparameter on that model. There are best hyperparameter I used in this model prediction :
- learning_rate = 0.0001
  
- hidden_unit = 64

- batch_size=256

- epoch = 100

- optimizers = adam

</p>

<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Here is the model loss from the prediction model: :
</p>

<p align="center"> 
 <img src="images/model loss.png" /> 
 <br></br>
 Model loss
</p>

<p align = "justify"> 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Can be seen from the picture above, train and val loss are not fluctuating and overfitting, so it can be said the model prediction can predict bitcoin price quite well. After the model is compiled, that model will be implemented on test set which has been partitioned before.Then, evaluate the prediction model with RMSE and MAPE
</p>

<p align="center"> 
 <img src="images/rmse and mape.png" /> 
 <br></br>
 Root Mean Square Error and Mean Absolute Percentage Error
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Based on the results obtained, the model is quite good in making predictions of bitcoin, with result of RMSE is 1994.7004 and MAPE is 18.5992% (Accuracy : 81.4%). After that, here is a table and graph from comparing the results of prediction with actual test data
</p>

<p align="center"> 
 Table comparasion data actual and data prediction
 <img src="images/table comparasion data actual and data prediction.png" /> 
 <img src="images/graph comparasion data actual and data prediction.png" /> 
 Graph comparasion data actual and data prediction
 <br></br>
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Can be seen from the graph, prediction results can can follow the movement from test set. But, there are some points of prediction result which is quite big different from the actual sets (for example at the first or initial data point).
</p>

For details you can check my code : [Sentiment Analysis 'Online Lectures' in Indonesia.ipynb](https://github.com/rifkyahmadsaputra/Sentiment-Analysis-Online-Lectures-in-Indonesia/blob/main/Sentiment%20Analysis%20'Online%20Lectures'%20in%20Indonesia.ipynb)
