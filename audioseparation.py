import os
import threading
import queue
from datetime import timedelta, datetime
import gradio as gr
import torch.cuda
from pydub import AudioSegment
import ffmpeg
import psutil
import shutil
import glob

spk_txt_queue = queue.Queue()
audio_concat_queue = queue.Queue()

# 初始化全局变量
home_directory = os.path.expanduser("~")
asr_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
asr_model_revision = "v2.0.4"
vad_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
vad_model_revision = "v2.0.4"
punc_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
punc_model_revision = "v2.0.4"
spk_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_campplus_sv_zh-cn_16k-common")
spk_model_revision = "v2.0.4"
hotword_file = "./hotwords.txt"

# 支持的音视频格式
support_audio_format = ['.mp3', '.m4a', '.aac', '.ogg', '.wav', '.flac', '.wma', '.aif']
support_video_format = ['.mp4', '.avi', '.mov', '.mkv']

# 加载模型（全局只加载一次）
device = "cuda" if torch.cuda.is_available() else "cpu"
ngpu = 1 if device == "cuda" else 0
ncpu = psutil.cpu_count(logical=False)  # 使用物理核心数

print(f"使用设备: {device}, GPU数量: {ngpu}, CPU核心数: {ncpu}")

# ASR 模型
model = None
def load_model():
    global model
    from funasr import AutoModel
    model = AutoModel(
        model=asr_model_path,
        model_revision=asr_model_revision,
        vad_model=vad_model_path,
        vad_model_revision=vad_model_revision,
        punc_model=punc_model_path,
        punc_model_revision=punc_model_revision,
        spk_model=spk_model_path,
        spk_model_revision=spk_model_revision,
        ngpu=ngpu,
        ncpu=ncpu,
        device=device,
        disable_pbar=True,
        disable_log=True,
        disable_update=True
    )
    print("模型加载完成")

# 在后台线程加载模型
threading.Thread(target=load_model).start()

# 热词处理
hotwords = ''
if os.path.exists(hotword_file):
    with open(hotword_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    hotwords = " ".join(lines)
    print(f"加载热词: {hotwords}")

def to_date(milliseconds):
    """将时间戳转换为SRT格式的时间"""
    time_obj = timedelta(milliseconds=milliseconds)
    return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"

def to_milliseconds(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    time_delta = time_obj - datetime(1900, 1, 1)
    milliseconds = int(time_delta.total_seconds() * 1000)
    return milliseconds

def find_audio_files(path):
    """查找指定路径下的所有音频/视频文件"""
    if not os.path.exists(path):
        return []
    
    files = []
    
    # 如果是文件，直接返回
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext.lower() in support_audio_format + support_video_format:
            return [path]
        return []
    
    # 如果是文件夹，递归查找
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in support_audio_format + support_video_format:
                files.append(os.path.join(root, filename))
    
    return files

def trans(file_paths, save_path, threshold=10, min_duration=60, progress=gr.Progress()):
    if model is None:
        return "错误：模型仍在加载中，请稍后再试"
    
    if not file_paths:
        return "请指定音频文件或文件夹路径"
    
    if not save_path:
        return "请指定保存路径"
    
    # 收集所有要处理的文件
    all_files = []
    for path in file_paths.split(';'):
        path = path.strip()
        if path:
            found_files = find_audio_files(path)
            all_files.extend(found_files)
    
    if not all_files:
        return f"未找到任何支持的音频/视频文件。支持的格式: {', '.join(support_audio_format + support_video_format)}"
    
    os.makedirs(save_path, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    final_save_path = os.path.join(save_path, date)
    os.makedirs(final_save_path, exist_ok=True)
    
    total_files = len(all_files)
    results = []
    min_duration_ms = min_duration * 1000  # 转换为毫秒
    
    for idx, audio_path in enumerate(all_files):
        try:
            # 处理进度更新
            progress(idx / total_files, f"处理文件中: {os.path.basename(audio_path)}")
            
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            _, audio_extension = os.path.splitext(audio_path)
            speaker_audios = {}  # 每个说话人作为 key，value 为列表，列表中为当前说话人对应的每个音频片段
            
            # 音频预处理
            try:
                audio_bytes, _ = (
                    ffmpeg.input(audio_path, threads=0, hwaccel='cuda' if device == "cuda" else None)
                    .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                    .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
                )
                
                # 使用锁确保模型安全调用
                with threading.Lock():
                    res = model.generate(
                        input=audio_bytes, 
                        batch_size_s=300, 
                        is_final=True, 
                        sentence_timestamp=True, 
                        hotword=hotwords
                    )
                
                rec_result = res[0]
                asr_result_text = rec_result['text']
                
                if asr_result_text == '':
                    results.append(f"{audio_name}: 未检测到语音内容")
                    continue
                
                sentences = []
                for sentence in rec_result["sentence_info"]:
                    start = to_date(sentence["start"])
                    end = to_date(sentence["end"])
                    
                    if sentences and sentence["spk"] == sentences[-1]["spk"] and len(sentences[-1]["text"]) < threshold:
                        sentences[-1]["text"] += " " + sentence["text"]
                        sentences[-1]["end"] = end
                    else:
                        sentences.append({
                            "text": sentence["text"], 
                            "start": start, 
                            "end": end, 
                            "spk": sentence["spk"]
                        })
                
                # 剪切音频或视频片段
                for i, stn in enumerate(sentences):
                    stn_txt = stn['text']
                    start = stn['start']
                    end = stn['end']
                    spk = stn['spk']
                    
                    # 根据文件名和 spk 创建目录
                    spk_save_path = os.path.join(final_save_path, audio_name, str(spk))
                    os.makedirs(spk_save_path, exist_ok=True)
                    
                    # 文本记录
                    spk_txt_file = os.path.join(final_save_path, audio_name, f'spk{spk}.txt')
                    with open(spk_txt_file, 'a', encoding='utf-8') as f:
                        f.write(f"{start} --> {end}\n{stn_txt}\n\n")
                    
                    # 处理音视频片段
                    final_save_file = os.path.join(spk_save_path, f"{i}{audio_extension}")
                    
                    try:
                        if audio_extension.lower() in support_audio_format:
                            (
                                ffmpeg.input(audio_path, threads=0, ss=start, to=end, hwaccel='cuda' if device == "cuda" else None)
                                .output(final_save_file)
                                .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True, capture_stderr=True)
                            )
                        elif audio_extension.lower() in support_video_format:
                            final_save_file = os.path.join(spk_save_path, f"{i}.mp4")
                            (
                                ffmpeg.input(audio_path, threads=0, ss=start, to=end, hwaccel='cuda' if device == "cuda" else None)
                                .output(final_save_file, vcodec='libx264', crf=23, acodec='aac', ab='128k')
                                .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True, capture_stderr=True)
                            )
                        else:
                            results.append(f"{audio_name}: 不支持的文件格式 {audio_extension}")
                    except ffmpeg.Error as e:
                        results.append(f"{audio_name}: 剪切错误 - {e.stderr.decode('utf-8')}")
                    
                    # 记录说话人和对应的音频片段
                    if spk not in speaker_audios:
                        speaker_audios[spk] = []
                    speaker_audios[spk].append({'file': final_save_file, 'audio_name': audio_name})
                
                # 合并每个说话人的音频片段（增加时长过滤）
                for spk, audio_segments in speaker_audios.items():
                    if not audio_segments:
                        continue
                    
                    # 计算说话人总时长
                    total_duration = 0
                    for seg in audio_segments:
                        try:
                            # 使用pydub获取音频时长（毫秒）
                            audio = AudioSegment.from_file(seg['file'])
                            total_duration += len(audio)
                        except:
                            # 如果是视频文件，使用ffmpeg获取时长
                            try:
                                probe = ffmpeg.probe(seg['file'])
                                video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                                audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
                                duration = float(audio_info['duration']) * 1000 if audio_info else float(video_info['duration']) * 1000
                                total_duration += duration
                            except:
                                # 如果无法获取时长，跳过这个文件
                                continue
                    
                    # 转换为秒
                    total_duration_sec = total_duration / 1000
                    
                    # 如果总时长不足阈值，跳过合并
                    if total_duration_sec < min_duration:
                        results.append(f"跳过说话人 {spk}，总时长 {total_duration_sec:.1f}秒不足{min_duration}秒")
                        continue
                    
                    output_file = os.path.join(final_save_path, audio_name, f"{spk}.mp3")
                    inputs = [seg['file'] for seg in audio_segments]
                    
                    try:
                        concat_audio = AudioSegment.from_file(inputs[0])
                        for audio_file_path in inputs[1:]:
                            concat_audio += AudioSegment.from_file(audio_file_path)
                        concat_audio.export(output_file, format="mp3")
                        results.append(f"✅ 合并完成: {os.path.basename(output_file)}，时长 {total_duration_sec:.1f}秒")
                    except Exception as e:
                        results.append(f"❌ 合并错误: {str(e)}")
                
                results.append(f"✅ {audio_name}: 处理完成，保存至 {os.path.join(final_save_path, audio_name)}")
                
            except Exception as e:
                results.append(f"❌ {audio_name}: 处理失败 - {str(e)}")
            
        except Exception as e:
            results.append(f"❌ 文件处理异常: {str(e)}")
    
    return "\n".join(results)

# Gradio UI
with gr.Blocks(title="说话人分离工具") as demo:
    gr.Markdown("## 🎙️ 音频说话人分离工具")
    gr.Markdown("直接指定实例上的音频文件或文件夹路径，系统将自动分离不同说话人的声音片段")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 输入设置")
            file_paths = gr.Textbox(
                label="文件或文件夹路径",
                placeholder="输入音频文件路径或包含音频的文件夹路径（多个用分号;分隔）",
                value="/root/autodl-tmp/input"
            )
            with gr.Row():
                example_files = gr.Examples(
                    examples=[
                        ["/root/autodl-tmp/audio1.mp3"],
                        ["/root/autodl-tmp/audio_folder"],
                        ["/root/autodl-tmp/audio1.wav;/root/autodl-tmp/audio2.mp4"]
                    ],
                    inputs=[file_paths],
                    label="示例路径"
                )
            
            save_path = gr.Textbox(
                label="保存路径",
                placeholder="输入保存结果的目录路径",
                value="/root/autodl-tmp/output"
            )
            
            with gr.Row():
                threshold = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=10,
                    step=1,
                    label="合并相同说话人阈值",
                    info="数值越小分割越细"
                )
                min_duration = gr.Slider(
                    minimum=5,
                    maximum=300,
                    value=60,
                    step=5,
                    label="最短合并时长（秒）",
                    info="总时长低于此值的说话人将被舍弃"
                )
            
            submit_btn = gr.Button("开始处理", variant="primary")
            
            with gr.Accordion("文件路径帮助", open=False):
                gr.Markdown("""
                - 支持直接输入文件路径或文件夹路径
                - 多个路径用分号(;)分隔
                - 支持的音频格式: .mp3, .wav, .aac, .flac, .ogg, .m4a, .wma, .aif
                - 支持的视频格式: .mp4, .avi, .mov, .mkv
                - 文件夹路径会自动搜索所有支持的音视频文件
                """)
        
        with gr.Column():
            output_result = gr.Textbox(label="处理结果", lines=20, interactive=False)
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            1. 在实例上准备音频文件或文件夹
            2. 指定文件路径或文件夹路径（默认为/root/autodl-tmp/input）
            3. 指定保存路径（默认为/root/autodl-tmp/output）
            4. 调整参数：
               - **合并阈值**：控制相邻相同说话人片段的合并程度
               - **最短时长**：舍弃总时长低于此值的说话人输出
            5. 点击开始处理按钮
            6. 结果将保存在指定目录下的日期文件夹中
            
            支持批量处理多个文件或整个文件夹的内容
            """)
    
    submit_btn.click(
        fn=trans,
        inputs=[file_paths, save_path, threshold, min_duration],
        outputs=output_result
    )

if __name__ == "__main__":
    # 在Autodl上建议使用share=True开启公网访问
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        show_error=True
    )