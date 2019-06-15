# Age and Gender prediction

Inference python script for several **age** and **gender** estimation
neural network models.

## Requirements

- python3.6
- numpy==1.13.0
- scikit_image==0.13.1
- caffe2==0.8.1
- skimage==0.0

## How To

### Download the pretrained models

```bash
	./download_models.sh
```

The script will download the [LAP](http://gesture.chalearn.org/)
**age** and **gender** prediction models in caffe format.

### Converting the original models

For example converting the LAP age model:
```bash
	cd models/lap
	python -m caffe2.python.caffe_translator \
		age.prototxt \
		dex_chalearn_iccv2015.caffemodel
```

This will create the needed `init_net.pb` and `predict_net.pb`
needed for inference.

Similarly for the gender model:
```bash
	cd models/gender
	python -m caffe2.python.caffe_translator \
		gender.prototxt \
		gender.caffemodel
```

### Run
```bash
	python run.py
```


## Resources

If you are using this codebase or the provided train models please cite
the authors:
```
@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}

@InProceedings{Rothe-ICCVW-2015,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {DEX: Deep EXpectation of apparent age from a single image},
  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year = {2015},
  month = {December},
}
```

### Papers:
* https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf
* https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf


### Datasets:

#### Appa-real
* https://github.com/yu4u/age-gender-estimation/tree/master/appa-real
* http://chalearnlap.cvc.uab.es/dataset/26/description/


#### IMDB - wiki 500:
* https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/


### Pre-trained models (caffe):
* https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/


### Caffe2 & Caffe2PyTorch conversion:
* https://caffe2.ai/docs/caffe-migration.html
* https://github.com/marvis/pytorch-caffe


### Misc:
* http://selfiecity.net/selfiexploratory/
