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

## Sample output result

```text
Optimization completed successfully!
-------------------------------
Name of data set: sample_dataset
Number of data points: 7
Initial parameters: {"alpha":8,"beta":0.05,"gamma":16,"delta":0.005}
Estimated parameters:
{"OptimizedParms":{"alpha":6.428916244727587,"beta":0.047848038501358894,"gamma":15.532604587678735,"delta":0.0018191239860987835},"MSRE":0.002387488865592934}
```

## Usage

- Direct call
  
    Example [main.go](./demos/simple/main.go):

    ``` bash
    cd demos/simple
    go run main.go
    ```

- Docker

    Build and run the image

    ``` bash
    docker build -t model-trainer .
    docker run -d -p 8080:8080 --name model-trainer model-trainer
    ```

    Submit a train request with a data set

    ``` bash
    curl -X POST http://localhost:8080/train -d @./samples/data.json
    ```

    ```json
    {
    "OptimizedParms": {
        "alpha": 6.4287804608259735,
        "beta": 0.047850084084481076,
        "gamma": 15.5333533929581,
        "delta": 0.0018186959434353788
    },
    "MSRE": 0.002387488045214774
    }
    ```
