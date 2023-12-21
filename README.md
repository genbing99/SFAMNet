# SFAMNet: A Scene Flow Attention-based Micro-expression Network

Neurocomputing | [Paper](https://www.sciencedirect.com/science/article/pii/S0925231223011219) | [Bibtex](#citation) |

(Released on April, 2023)

## Results
Performance comparison for <b>micro-expression spotting</b>. <br>
<img src='images/result_ME_spot.png' width=450 height=260>

Performance comparison for <b>micro-expression recognition</b>. <br>
<img src='images/result_ME_recog.png' width=700 height=300>

Performance comparison for <b>micro-expression analysis</b>. <br>
<img src='images/result_ME_analysis.png' width=600 height=210>

## How to run the code
<b>Step 1)</b> Download the processed_data from:

<!--
https://drive.google.com/drive/folders/1D5az-DAyzY1C1ZqoZb8Z_o3eTXesAIWT?usp=sharing
-->
<b>hidden at the moment</b>

The files are structured as follows:
>├─annotation <br>
>├─pretrained_weights <br>
>├─Utils <br>
>├─dataloader.py <br>
>├─load_data.py <br>
>├─main.py <br>
>├─network.py <br>
>├─prepare_data.py <br>
>├─requirements.txt <br>
>├─train.py <br>
>├─train_utils.py <br>
>├─<b>processed_data</b> <br>
>>├─<b>CASME_cube_recog_rgbd-flow.pkl</b> <br>
>>└─<b>CASME_cube_spot_rgbd-flow.pkl</b>

<b>Step 2)</b> Installation of packages using pip

``` pip install -r requirements.txt ```

<b>Step 3)</b> Network Training and Evaluation

``` python main.py ```

#### &nbsp; Note for parameter settings <br>
&nbsp;&nbsp; --train (True/False) <br>
&nbsp;&nbsp; --emotion (4/7)

## Citation
If you find this work useful for your research, please cite
```bibtex
@article{liong2024sfamnet,
  title={SFAMNet: A scene flow attention-based micro-expression network},
  author={Liong, Gen-Bing and Liong, Sze-Teng and Chan, Chee Seng and See, John},
  journal={Neurocomputing},
  volume={566},
  pages={126998},
  year={2024},
  publisher={Elsevier}
}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to
`genbing67@gmail.com` or `cs.chan at um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file). 

&#169;2023 Center of Image and Signal Processing, Faculty of Computer Science and Information Technology, Universiti Malaya.

