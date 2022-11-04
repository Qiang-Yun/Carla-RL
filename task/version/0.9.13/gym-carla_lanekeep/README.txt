# 安装方法

## 1、下载carla0.9.6

## 2、安装环境

```shell
# 创建环境
$ conda create -n py36 python=3.6
# 激活环境
$ conda activate py36
# 进入gym-carla_acc文件夹下
$ pip install -r requirements.txt
$ pip install -e .
# 安装好上述包之后安装下面几个包
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install tianshou
$ pip install pyyaml
```



# 启动方法

## 1、启动环境

```shell
# 进入CARLA_0.9.6文件夹下
$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
```

## 2、启动代码
#######注意#####
启动代码前 需要将 TS_ppo.py 和 Carla_sac.py
sys.path.append('/home/yq/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
修改为本地对应的路径，才能import Carla

```shell
# 进入gym-carla_lanekeep文件夹下
$ conda activate py36
$ python TS_ppo.py
$ python Carla_sac.py
```

# 错误提示

1、如果无法使用pip安装requirements.txt文件夹下的环境，降低pip的setuptools的版本。
