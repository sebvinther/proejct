#!/bin/bash

set -e

cd ./proejct/SML

python news.py

python stocks.py

python news_preprocess.py

python stock_preprocess.py

python feature_pipeline.py

python feature_view.py

python training_pipeline1.py

python inference_pipeline.py