1. Build image `docker build -t gennari_fashion_captioning .`
2. Run `docker run -it --rm --gpus=all -v $PWD:/src -v /datasets/:/datasets gennari_fashion_captioning python ./test.py`

The script will take a random image from the Fashiongen validation dataset and it will generate a caption. The real caption is also printed. The image is saves as `img.png`