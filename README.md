# Revisiting Graph-based Fraud Detection in Sight of Heterophily and Spectrum (AAAI2024)

## Abstract
Graph-based fraud detection (GFD) can be regarded as a challenging semi-supervised node binary classification task. In recent years, Graph Neural Networks (GNN) have been widely applied to GFD, characterizing the anomalous possibility of a node by aggregating neighbor information. However, fraud graphs are inherently heterophilic, thus most of GNNs perform poorly due to their assumption of homophily. In addition, due to the existence of heterophily and class imbalance problem, the existing models do not fully utilize the precious node label information. To address the above issues, this paper proposes a semi-supervised GNN-based fraud detector SEC-GFD. This detector includes a hybrid filtering module and a local environmental constraint module, the two modules are utilized to solve heterophily and label utilization problem respectively. The first module starts from the perspective of the spectral domain, and solves the heterophily problem to a certain extent. Specifically, it divides the spectrum into various mixed-frequency bands based on the correlation between spectrum energy distribution and heterophily. Then in order to make full use of the node label information, a local environmental constraint module is adaptively designed. The comprehensive experimental results on four real-world fraud detection datasets denote that SEC-GFD outperforms other competitive graph-based fraud detectors.


## Implementation
The relevant datasets developed in the paper are on [google drive](https://drive.google.com/drive/folders/1eqfWN0CIudj7e9KJvkmj5uzK-eWs_pSE?usp=sharing). Download and unzip all files in the `data` folder.


If you use this package and find it useful, please cite our paper using the following BibTeX:)

```
@inproceedings{xu2024revisiting,
  title={Revisiting graph-based fraud detection in sight of heterophily and spectrum},
  author={Xu, Fan and Wang, Nan and Wu, Hao and Wen, Xuezhi and Zhao, Xibin and Wan, Hai},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={8},
  pages={9214--9222},
  year={2024}
}
```
