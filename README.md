# MGANet
PyTorch implementation of "MGANet: A Robust Model for Quality Enhancement of Compressed Video" 

[arXiv](http://arxiv.org/abs/2481027)

<p align="center">
    <img src="files/framework.png" width="800"> <br />
    <em> The proposed MGANet framework</em>
</p>

<p align="center">
    <img src="files/GUNet.png" width="800"> <br />
    <em> The guided encoder-decoder subnet </em>
</p>


## Installation
The code was developed using Python 3.6 & PyTorch 0.4 & CUDA 8.0. There may be a problem related to software versions. To fix the problem, you may look at the implementation in MGANet files and replace the syntax to match the new PyTorch environment. Install correlation package (if you haven't installed this package before):
## Code v1.0
Currently, we release our research code for testing and training. It should produce the same results as in the paper under LD and AI configurations and frame number 3.
## Test
* It would be very easy to understand the test function and test on your own data.
* An example of test usage is shown as follows:
```bash 
python MGANet_test_AI37.py --gpu_id 1 --is_training False
```
## Train and Data
* We will update the training database for the Tus' partition of HEVC for training and better reading after recent paper deadline.
## Video Results
* [Here](http://arxiv.org/abs/2481027) we provide quality enhancement results of compressed video for 18 widely used sequences for visual and quantitative comparisons.
## Citation

If you use any part of our code, or our data is useful for your research, please consider citing:

```
@article{MGANET2018,
  author={Xiandong, Meng and Xuan, Deng and Shuyuan, Zhu and Shuaicheng, Liu and Chuan, Wang and Chen, Chen and Bing, Zeng},
  title={MGANet: A Robust Model for Quality Enhancement of Compressed Video},
  journal={arXiv preprint arXiv:2481027},
  year={2018}
}
