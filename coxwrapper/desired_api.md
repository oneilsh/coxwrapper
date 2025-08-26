# Desired API


For now, I'd like to do something like this:

```
!pip install git+https://<this repo>
```

I'd like to load the config from a dictionary read from a yaml, so that I can optionally define the config in pure python:

```
import yaml
from twinsight_model import CoxModelWrapper

config_dict = .... # load 'configuration_cox.yaml'

model = CoxModelWrapper(config = config_dict)
```

I'd like to be able to pretty simply load the data:

```
model.load_data()
model.train_data_summary() # info about # of patients, demographic breakdown, counts of outcomes and features...
```

Train it:

```
model.train(split = 0.8)
model.get_train_stats() # auroc, auprc, generate plots?
```

And use it:

```
pt_data = [{"age": 32, "bmi": 28.1, "heart_failure": True, ...}, {"age": 56, ...}]
predictions = model.get_prediction(pt_data) # 5 year survival probability, hazard ratio
```

Eventually I will want to export it to a .pkl file:

```
model.save_pickle('model.pkl')
```

And then later load it from pickle in another environment (assuming the deps are there), and use it to get predictions and other information:

```
model_rehydrated = .... # load from pickle

pt_data = [{"age": 32, "bmi": 28.1, "heart_failure": True, ...}, {"age": 56, ...}]
predictions = model_rehydrated.get_prediction(pt_data) # 5 year survival probability, hazard ratio
```

I also want to be able to get some of the stats as before:

```
model_rehydrated.train_data_summary()
model_rehydrated.get_train_stats()
```

HOWEVER, and this is IMPORTANT: no row-level data should be stored, and any count information that is stored should replace counts between 1 and 20 (inclusive) with "<20", since I am not allowed export any counts or 
other statistics based on patient groups of <20 (0 is ok).

Ideally, there would be some way for it to also tell me the required input types:

```
model_rehydrated.get_input_schema()
```

At this point, I don't really want to adjust the configuration schema that we've come up with, but if there's a way to use that info to inform the train_data_summary, train_stats, or input_schema, that could be good.



## suggested by AI:

# Simplified, production-ready API
from twinsight_model import CoxModelWrapper

# Initialize
model = CoxModelWrapper.from_pickle('copd_model.pkl')

# Basic prediction - validates input types and returns informative error
risk = model.predict_risk_batch([{"age": 65, "smoking": "current", "bmi": 28}])
# Returns: [{"1_year_risk": 0.05, "5_year_risk": 0.22, "hazard_ratio": 2.1}]

# Model information
info = model.get_model_info()
schema = model.get_input_schema()
importance = model.get_feature_importance()