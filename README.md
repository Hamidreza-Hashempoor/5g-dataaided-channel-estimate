

# Deep Learning Based Data-Assisted Channel Estimation and Detection
This repository contains the code for the paper _Deep Learning Based Data-Assisted Channel Estimation and Detection_

## Framework

![image](./Fig/whole_frame.jpg)


## Requirements

Python 3.8 or later with all ```requirements.txt``` dependencies installed. To install run:
```bash
$ pip install -r requirements.txt
```

## Code
### Data Preparation
To generate channel.mat please use MATLAB. Check [here](https://www.mathworks.com/help/5g/ref/nrtdlchannel-system-object.html).
Details are explained there.

For simplicity, we are calling data generation modules in the ``main_script`` so can skip the data generation section.
Data generation `.py` files for single pendulum, double pendulum and irregular polygon experiments:

>   * project dir
>     * double pendulum image imputation
>       * `DoublePendulum.py`
>     * double pendulum state estimation
>       * `DoublePendulum.py`
>     * pendulum image imputation
>       * `PendulumData.py`
>     * pendulum state estimation
>       * `PendulumData.py`
>     * polybox image imputation
>       * PolyboxData.py
>       * PymunkData.py
>     * polybox state estimation
>       * PolyboxData.py
>       * PymunkData.py


### Experiments
If you just want run the experiments, you can directly run the ``main_script`` of each experiment as follow:
* double_pendulum image imputation
 ```
cd double pendulum image imputation
python double_pendulum_image_imputation.py --config config0.json
cd ..
```
After running the code, dataset will be generated in `double pendulum image imputation/data` folder and the results are saved at 
`double pendulum image imputation/results`

* double pendulum state estimation
 ```
cd double pendulum state estimation
python double_pendulum_state_estimation.py --config config0.json
cd ..
```
After running the code, dataset will be generated in `double pendulum state estimation/data` folder and the results are saved at 
`double pendulum state estimation/results`

* pendulum image imputation
 ```
cd pendulum image imputation
python pendulum_image_imputation.py --config config.json
cd ..
```
After running the code, dataset will be generated in `pendulum image imputation/data` folder and the results are saved at 
`pendulum image imputation/results`

* pendulum state estimation
 ```
cd pendulum state estimation
python pendulum_state_estimation.py --config config.json
cd ..
```
After running the code, dataset will be generated in `pendulum state estimation/data` folder and the results are saved at 
`pendulum state estimation/results`

* polybox image imputation
 ```
cd polybox image imputation
python polybox_image_imputation.py --config config.json
cd ..
```
After running the code, dataset will be generated in `polybox image imputation/data` folder and the results are saved at 
`polybox image imputation/results`

* polybox state estimation
 ```polybox_state_estimation.py
cd polybox state estimation
python polybox_state_estimation.py --config config.json
cd ..
```
After running the code, dataset will be generated in `polybox state estimation/data` folder and the results are saved at 
`polybox state estimation/results`
