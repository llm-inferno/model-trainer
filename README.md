# Train queueing model offline: Estimate parameters using a data set

## Sample data point

```json
        {
            "requestRate": 76.95,
            "inputTokens": 64,
            "outputTokens": 64,
            "avgWaitTime": 0.0,
            "avgPrefillTime": 20.47,
            "avgITLTime": 8.63,
            "maxBatchSize": 512
        }
```

## Sample output

```text
Optimization completed successfully!
-------------------------------
Name of data set: sample_dataset
Number of data points: 7
Initial parameters: {"alpha":8,"beta":0.05,"gamma":16,"delta":0.005}
Estimated parameters:
{"OptimizedParms":{"alpha":6.428916244727587,"beta":0.047848038501358894,"gamma":15.532604587678735,"delta":0.0018191239860987835},"MRSE":0.002387488865592934}
```

