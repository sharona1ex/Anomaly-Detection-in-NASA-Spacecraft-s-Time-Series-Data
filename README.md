# Anomaly Detection in NASA Spacecraft Sensor Data
## Objective
Our main objective is to develop a system that can detect any unusual activity in the data collected from NASA satellites and spacecraft. The system is designed to identify any anomalies and notify technicians, thereby reducing the time and effort required to manually review all the data. Our aim is to minimize the number of false detections (both positive and negative) in order to achieve maximum accuracy.

## Project Scenario
NASA has developed an innovative approach called dynamic thresholding to detect anomalies in time series data using LSTM models. I have fine-tuned and improved the model's performance to achieve greater accuracy and minimize false detections. This system reduces the monitoring burden on operations engineers and minimizes operational risk, bringing us closer to unlocking the mysteries of the universe.

## Anomaly in Time Series
### What is an anomaly in time series data?
With regards to time series, any unusual value that seems to be out of normal is an anomaly. There are three types of anomalies with respect to time series as per the NASA engineers,
#### Point Anomalies:

* Point anomalies refer to individual data points that are considered abnormal when compared to the rest of the data.
* These anomalies are characterized by their deviation from the normal behavior of the time series.
* They typically occur as isolated instances within the time series and are identifiable as extreme values or outliers.
* Point anomalies can be caused by various factors such as measurement errors, sensor malfunctions, or rare events.
#### Collective Anomalies:

* Collective anomalies occur when a sequence of data points exhibits abnormal behavior as a whole, rather than any single point being anomalous by itself.
* Instead of isolated extreme values, collective anomalies are identified by patterns or trends in the time series data that deviate significantly from the expected or normal behavior.
* These anomalies may arise due to systemic issues, unexpected changes in underlying processes, or complex interactions among multiple variables over time.
#### Contextual Anomalies:

* Contextual anomalies are single data points that may not appear abnormal when viewed in isolation but are considered anomalous with respect to the local context or neighboring data points.
* Unlike point anomalies, contextual anomalies are identified by comparing a data point to its surrounding context or historical pattern within a specific time window.
* These anomalies often require contextual information or domain knowledge to distinguish them from normal variations in the data.
* Contextual anomalies can be caused by subtle changes in underlying conditions, shifts in behavior over time, or contextual dependencies within the time series.

## Anomaly Detection System Architecture
![image](https://github.com/StarRider/Anomaly-Detection-in-NASA-Spacecraft-s-Time-Series-Data/assets/30108439/33474489-c533-4e0d-8a3c-c24aa30ea403)

