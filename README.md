# AudioSeparation
# 1. 说明
这是基于开源的 FunASR 实现的说话人分离的 gradio webui 项目\
要求 python version >= 3.8 \
支持运行在 Windows、MacOS、Linux 系统 \
热词功能，在当前路径下的 hotwords.txt 中写入热词，每个热词一行
# 2. 安装
执行下面命令来安装依赖
```shell
pip install -U funasr modelscope ffmpeg-python pydub
```
此外还需要安装torch和ffmpeg
# 4. 功能
1. 支持对指定的单个或者多个音频中不同的说话人讲的话进行分离，分别归类到不同的目录中
2. 保存每个说话人对应的包含时间戳的文本内容
3. 支持视频切片，根据说话人声音进行视频切片 
4. 支持自定义热词

# 5. 模型下载
执行下面程序，会自动下载模型到当前用户 .cache/modelscope/hub/models/iic/ 目录中
```shell
python download_model.py
```
