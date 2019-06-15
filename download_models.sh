#!/usr/bin/env bash

# Dowload the LAP model (age prediction)
mkdir models/lap
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel \
-O models/lap/dex_chalearn_iccv2015.caffemodel

wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt \
-O models/lap/age.prototxt

# Download the Gender prediction model
mkdir models/gender
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel \
-O models/gender/gender.caffemodel

wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt \
-O models/gender/gender.prototxt
