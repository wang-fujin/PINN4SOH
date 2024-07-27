# PINN4SOH
This code is for our paper: [Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis](https://www.nature.com/articles/s41467-024-48779-z)

> [!IMPORTANT]
Summary of articles using the XJTU Battery Dataset:
https://github.com/wang-fujin/XJTU-Battery-Dataset-Papers-Summary

# 1. System requirements
python version: 3.7.10

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  1.7.1   |
|    sklearn     |  0.24.2  |
|     numpy      |  1.20.3  |
|     pandas     |  1.3.5   |
|   matplotlib   |  3.3.4   |
|  scienceplots  |          |



# 2. Installation guide
If you are not familiar with Python and Pytorch framework, 
you can install Anaconda first and use Anaconda to quickly configure the environment.
## 2.1 Create environment
```angular2html
conda create -n new_environment python=3.7.10
```



## 2.2 Activate environment
```angular2html
conda activate new_environment
```

## 2.3 Install dependencies
```angular2html
conda install pytorch=1.7.1
conda install scikit-learn=0.24.2 numpy=1.20.3 pandas=1.3.5 matplotlib=3.3.4
pip install scienceplots      # for beautiful plots
```

# 3. Demo
We provide a detailed demo of our code running on the XJTU dataset.
1. Run the `main_XJTU.py` file to train our model. The program will generate a folder named `results` and save the results in it.
2. Run the `main_comparison.py` file. You can change `setattr(args,'model','MLP')` to select the CNN or MLP model. It will generate a folder in the `results` to save the results of the corresponding model (CNN or MLP).
3. Run the `results analysis/XJTU results.py` file. It will process the results in Step one and generate the `XJTU_results.xlsx` file. At the same time, the results of each batch in the XJTU dataset will also be printed on the Command Console, corresponding to the results in Table 2 of our manuscript.
4. Run the `results analysis/Comparision results.py` file to generate the `XJTU-MLP_results.xlsx` file and save it in the `results` folder. The results of each batch in the XJTU data set will also be printed on the Command Console, corresponding to the results in Table 2 of our manuscript.

**Note: As we all know, the training process of neural network models is random, 
and the volatility of regression models is often greater than that of classification models. 
Therefore, the results obtained from the above process are not expected to be exactly identical to those mentioned in our manuscript. 
However, it is evident that the results obtained from our method are superior to those of MLP and CNN.**

In addition, we also provide the results of our training, 
which are saved in the `results` folder and `results analysis` folder. 
These results correspond exactly to the data in our manuscript.

What's more, we also provide the codes corresponding to the Figures in our manuscript, 
which are saved in the `plotter` folder.
You can use these codes to draw the Figures in the manuscript.


# 4.  Additional information
The data in the `data` folder is preprocessed data.
Raw data can be obtained from the following links:
1. XJTU dataset: [link](https://wang-fujin.github.io/)
2. TJU dataset: [link](https://zenodo.org/record/6405084)
3. HUST dataset: [link](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
4. MIT dataset: [link](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)

The code for **reading and preprocessing** the dataset is publicly available at [https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)

---

We generated a comprehensive dataset consisting of 55 lithium-nickel-cobalt-manganese-oxide (NCM) batteries. 

It is available at: [Link](https://wang-fujin.github.io/)

Zenodo link: [https://zenodo.org/records/10963339](https://zenodo.org/records/10963339).

![https://github.com/wang-fujin/PINN4SOH/blob/main/xjtu%20battery%20dataset.png](https://github.com/wang-fujin/PINN4SOH/blob/main/xjtu%20battery%20dataset.png)

![https://github.com/wang-fujin/PINN4SOH/blob/main/6%20batches.png](https://github.com/wang-fujin/PINN4SOH/blob/main/6%20batches.png)

# 5. Citation
If you find it useful, please cite our paper:
```bibtex
@article{wang2024physics,
  title={Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author={Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={4332},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
