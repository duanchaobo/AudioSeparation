import os
import threading
import queue
from datetime import timedelta, datetime
import gradio as gr
import torch.cuda
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import ffmpeg
import psutil
import numpy as np
import json
import re
import wave
import contextlib

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

# 只支持视频格式
support_video_format = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.ts']
support_subtitle_format = ['.srt']

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
    """将时间字符串转换为毫秒"""
    try:
        # 标准化时间字符串：将逗号替换为点，并移除所有空格
        time_str = time_str.replace(',', '.').replace(' ', '')
        
        # 处理带毫秒的时间格式
        if '.' in time_str:
            # 分割小时、分钟、秒和毫秒
            parts = re.split(r'[:.]', time_str)
            
            if len(parts) >= 4:
                # HH:MM:SS.mmm 格式
                hours = int(parts[0]) if parts[0] else 0
                minutes =int(parts[1]) if parts[1] else 0
                seconds = int(parts[2]) if parts[2] else 0
                milliseconds = int(parts[3].ljust(3, '0')[:3])  # 确保毫秒部分有三位
                
                return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds
            elif len(parts) == 3:
                # MM:SS.mmm 格式
                minutes = int(parts[0]) if parts[0] else 0
                seconds = int(parts[1]) if parts[1] else 0
                milliseconds = int(parts[2].ljust(3, '0')[:3])
                return minutes * 60000 + seconds * 1000 + milliseconds
            elif len(parts) == 2:
                # SS.mmm 格式
                seconds = int(parts[0]) if parts[0] else 0
                milliseconds = int(parts[1].ljust(3, '0')[:3])
                return seconds * 1000 + milliseconds
            else:
                return 0
        else:
            # 处理不带毫秒的时间格式
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600000 + minutes * 60000 + seconds * 1000
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60000 + seconds * 1000
            elif len(parts) == 1:  # SS
                seconds = int(parts[0])
                return seconds * 1000
        
        return 0
    except Exception as e:
        print(f"时间转换错误: {time_str} - {str(e)}")
        return 0

def find_video_files(path):
    """查找指定路径下的所有视频文件"""
    if not os.path.exists(path):
        return []
    
    files = []
    
    # 如果是文件，直接返回
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext.lower() in support_video_format:
            return [path]
        return []
    
    # 如果是文件夹，递归查找
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in support_video_format:
                files.append(os.path.join(root, filename))
    
    return files

def parse_srt_content(content):
    """解析SRT字幕内容"""
    subtitles = []
    
    # 分割字幕块
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            # 第一行是序号，第二行是时间范围
            time_line = lines[1]
            text = '\n'.join(lines[2:]).strip()
            
            if '-->' in time_line:
                start_str, end_str = time_line.split('-->', 1)
                start_str = start_str.strip()
                end_str = end_str.strip()
                
                start_ms = to_milliseconds(start_str)
                end_ms = to_milliseconds(end_str)
                
                if start_ms > 0 or end_ms > 0:
                    subtitles.append({
                        'start': start_ms,
                        'end': end_ms,
                        'text': text
                    })
    
    return subtitles

def match_subtitles_to_media(media_path, subtitle_files):
    """为媒体文件匹配最合适的字幕文件"""
    media_name = os.path.splitext(os.path.basename(media_path))[0]
    best_match = None
    best_score = 0
    
    for sub_file in subtitle_files:
        sub_name = os.path.splitext(os.path.basename(sub_file.name))[0]
        
        # 计算文件名相似度
        score = 0
        min_length = min(len(media_name), len(sub_name))
        for i in range(min_length):
            if media_name[i] == sub_name[i]:
                score += 1
        
        # 如果当前匹配度更高，更新最佳匹配
        if score > best_score:
            best_match = sub_file
            best_score = score
    
    return best_match

def align_subtitles_with_asr(asr_sentences, subtitles):
    """将ASR结果与字幕进行对齐"""
    aligned_results = []
    
    # 如果没有字幕，直接返回ASR结果
    if not subtitles:
        return asr_sentences
    
    # 对齐逻辑
    for sub in subtitles:
        best_match = None
        best_overlap = 0
        
        for asr_sentence in asr_sentences:
            # 计算时间重叠
            overlap_start = max(sub['start'], asr_sentence['start'])
            overlap_end = min(sub['end'], asr_sentence['end'])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # 计算重叠比例
            sub_duration = sub['end'] - sub['start']
            asr_duration = asr_sentence['end'] - asr_sentence['start']
            
            # 更新最佳匹配
            if overlap_duration > best_overlap:
                best_match = asr_sentence
                best_overlap = overlap_duration
        
        # 如果找到匹配，使用字幕的时间戳
        if best_match:
            aligned_results.append({
                'start': sub['start'],
                'end': sub['end'],
                'text': sub['text'],
                'spk': best_match['spk']
            })
        else:
            # 否则保留ASR结果
            aligned_results.append(asr_sentence)
    
    return aligned_results

def detect_actual_speech(audio_path, asr_start, asr_end, silence_threshold=-40, min_silence_duration=200):
    """检测音频中实际语音的起止时间"""
    try:
        # 扩展检测范围（前后各加500毫秒）
        expand_ms = 500
        detection_start = max(0, asr_start - expand_ms)
        detection_end = asr_end + expand_ms
        
        # 从原始音频中提取检测片段
        temp_wav = "temp_detection.wav"
        (
            ffmpeg.input(audio_path)
            .filter('atrim', start=detection_start/1000, end=detection_end/1000)
            .output(temp_wav, ac=1, ar=16000, acodec='pcm_s16le')
            .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, quiet=True)
        )
        
        # 加载音频进行分析
        audio = AudioSegment.from_wav(temp_wav)
        
        # 检测非静音部分
        nonsilent_parts = detect_nonsilent(
            audio, 
            min_silence_len=min_silence_duration, 
            silence_thresh=silence_threshold
        )
        
        # 清理临时文件
        os.remove(temp_wav)
        
        if not nonsilent_parts:
            return asr_start, asr_end
        
        # 获取第一个和最后一个非静音段的起止时间
        first_speech_start = nonsilent_parts[0][0]
        last_speech_end = nonsilent_parts[-1][1]
        
        # 计算实际偏移量
        actual_start = detection_start + first_speech_start
        actual_end = detection_start + last_speech_end
        
        # 应用边界安全扩展
        start_buffer = 100  # 开始前增加100ms
        end_buffer = 200    # 结束后增加200ms
        
        actual_start = max(0, actual_start - start_buffer)
        actual_end = actual_end + end_buffer
        
        # 确保调整后的时间在原始范围内
        actual_start = max(asr_start, actual_start)
        actual_end = min(asr_end, actual_end)
        
        return actual_start, actual_end
        
    except Exception as e:
        print(f"语音边界检测错误: {str(e)}")
        return asr_start, asr_end

def trans(file_paths, save_path, subtitle_files, threshold=10, min_duration=60, max_duration=10, 
          silence_threshold=-35, vad_refinement=True, progress=gr.Progress()):
    if model is None:
        return "错误：模型仍在加载中，请稍后再试"
    
    if not file_paths:
        return "请指定视频文件或文件夹路径"
    
    if not save_path:
        return "请指定保存路径"
    
    # 收集所有要处理的视频文件
    all_files = []
    for path in file_paths.split(';'):
        path = path.strip()
        if path:
            found_files = find_video_files(path)
            all_files.extend(found_files)
    
    if not all_files:
        return f"未找到任何支持的视频文件。支持的格式: {', '.join(support_video_format)}"
    
    # 处理上传的字幕文件
    subtitle_contents = []
    if subtitle_files:
        for sub_file in subtitle_files:
            with open(sub_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                subtitle_contents.append(content)
    
    os.makedirs(save_path, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    final_save_path = os.path.join(save_path, date)
    os.makedirs(final_save_path, exist_ok=True)
    
    total_files = len(all_files)
    results = []
    min_duration_ms = min_duration * 1000  # 转换为毫秒
    
    for idx, video_path in enumerate(all_files):
        try:
            # 处理进度更新
            progress(idx / total_files, f"处理文件中: {os.path.basename(video_path)}")
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            _, video_extension = os.path.splitext(video_path)
            speaker_videos = {}  # 每个说话人作为 key，value 为列表，列表中为当前说话人对应的每个视频片段
            speaker_valid_segments = {}  # 记录有效片段数量
            speaker_discarded_segments = {}  # 记录舍弃片段数量
            timing_adjustments = []  # 记录时间调整信息
            
            # 视频预处理
            try:
                # 尝试为当前媒体文件匹配字幕
                matched_subtitles = []
                if subtitle_contents:
                    # 尝试查找匹配的字幕内容
                    for content in subtitle_contents:
                        subtitles = parse_srt_content(content)
                        # 简单的匹配：检查字幕中是否包含媒体文件名
                        if any(video_name.lower() in sub['text'].lower() for sub in subtitles):
                            matched_subtitles = subtitles
                            results.append(f"✅ 使用匹配的字幕内容 (基于文件名: {video_name})")
                            break
                    
                    # 如果没有精确匹配，使用第一个字幕文件
                    if not matched_subtitles:
                        matched_subtitles = parse_srt_content(subtitle_contents[0])
                        results.append(f"ℹℹ️ 使用第一个字幕文件内容")
                
                # 获取视频总时长
                try:
                    probe = ffmpeg.probe(video_path)
                    audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
                    if audio_info:
                        total_duration_ms = float(audio_info['duration']) * 1000
                    else:
                        # 如果没有音频流，可能是纯视频文件
                        total_duration_ms = 0
                        results.append(f"⚠️ {video_path} 未检测到音频流")
                except:
                    total_duration_ms = 0
                
                audio_bytes, _ = (
                    ffmpeg.input(video_path, threads=0, hwaccel='cuda' if device == "cuda" else None)
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
                    results.append(f"{video_name}: 未检测到语音内容")
                    continue
                
                # 处理ASR句子信息
                asr_sentences = []
                for sentence in rec_result["sentence_info"]:
                    asr_sentences.append({
                        "start": sentence["start"],
                        "end": sentence["end"],
                        "text": sentence["text"],
                        "spk": sentence["spk"]
                    })
                
                # 将ASR结果与字幕对齐
                aligned_sentences = align_subtitles_with_asr(asr_sentences, matched_subtitles)
                
                # 如果没有对齐结果，使用原始ASR结果
                if not aligned_sentences:
                    aligned_sentences = asr_sentences
                    results.append("⚠️ 未检测到字幕，使用原始ASR时间戳")
                
                # 处理每个说话人的片段
                for spk in set([s["spk"] for s in aligned_sentences]):
                    speaker_valid_segments[spk] = 0
                    speaker_discarded_segments[spk] = 0
                
                # 保存对齐后的时间戳信息
                timestamp_file = os.path.join(final_save_path, video_name, "timestamps.json")
                os.makedirs(os.path.dirname(timestamp_file), exist_ok=True)
                with open(timestamp_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "asr_sentences": asr_sentences,
                        "subtitles": matched_subtitles,
                        "aligned": aligned_sentences
                    }, f, ensure_ascii=False, indent=2)
                
                # 剪切视频片段
                for i, sentence in enumerate(aligned_sentences):
                    stn_txt = sentence['text']
                    start_ms = sentence['start']
                    end_ms = sentence['end']
                    spk = sentence['spk']
                    
                    # 根据ASR结果判断：如果文本为空，则舍弃该片段
                    if stn_txt.strip() == '':
                        speaker_discarded_segments[spk] += 1
                        continue
                    
                    # 应用VAD优化：检测实际语音边界
                    if vad_refinement:
                        original_start = start_ms
                        original_end = end_ms
                        start_ms, end_ms = detect_actual_speech(
                            video_path, start_ms, end_ms, 
                            silence_threshold=silence_threshold
                        )
                        
                        # 记录调整信息
                        timing_adjustments.append({
                            "segment": i,
                            "original_start": original_start,
                            "original_end": original_end,
                            "adjusted_start": start_ms,
                            "adjusted_end": end_ms,
                            "start_diff": start_ms - original_start,
                            "end_diff": end_ms - original_end
                        })
                    
                    # 格式化时间戳
                    start_str = to_date(start_ms)
                    end_str = to_date(end_ms)
                    
                    # 根据文件名和 spk 创建目录
                    spk_save_path = os.path.join(final_save_path, video_name, str(spk))
                    os.makedirs(spk_save_path, exist_ok=True)
                    
                    # 文本记录
                    spk_txt_file = os.path.join(final_save_path, video_name, f'spk{spk}.txt')
                    with open(spk_txt_file, 'a', encoding='utf-8') as f:
                        f.write(f"{start_str} --> {end_str}\n{stn_txt}\n\n")
                    
                    # 处理视频片段
                    final_save_file = os.path.join(spk_save_path, f"{i}.mp4")
                    
                    try:
                        (
                            ffmpeg.input(video_path, threads=0, ss=start_ms/1000, to=end_ms/1000, hwaccel='cuda' if device == "cuda" else None)
                            .output(final_save_file, vcodec='libx264', crf=23, acodec='aac', ab='128k')
                            .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True, capture_stderr=True)
                        )
                    except ffmpeg.Error as e:
                        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
                        results.append(f"{video_name}: 剪切错误 - {error_msg}")
                        continue
                    
                    # 记录该片段为有效片段
                    speaker_valid_segments[spk] += 1
                    
                    # 记录说话人和对应的视频片段
                    if spk not in speaker_videos:
                        speaker_videos[spk] = []
                    speaker_videos[spk].append({
                        'file': final_save_file, 
                        'video_name': video_name,
                        'start': start_ms,
                        'end': end_ms
                    })
                
                # 合并每个说话人的视频片段（增加时长过滤）
                for spk, video_segments in speaker_videos.items():
                    if not video_segments:
                        continue
                    
                    # 计算说话人总时长
                    total_duration = 0
                    for seg in video_segments:
                        total_duration += (seg['end'] - seg['start'])
                    
                    # 转换为秒
                    total_duration_sec = total_duration / 1000
                    
                    # 如果总时长不足阈值，跳过合并
                    if total_duration_sec < min_duration:
                        results.append(f"跳过说话人 {spk}，总时长 {total_duration_sec:.1f}秒不足{min_duration}秒")
                        continue
                    
                    # 输出合并视频文件（MP4格式）
                    output_file = os.path.join(final_save_path, video_name, f"{spk}.mp4")
                    
                    # 构建合并命令
                    try:
                        # 创建文件列表
                        concat_list = os.path.join(spk_save_path, "concat_list.txt")
                        with open(concat_list, 'w', encoding='utf-8') as f:
                            for seg in video_segments:
                                f.write(f"file '{seg['file']}'\n")
                        
                        # 使用ffmpeg合并视频
                        (
                            ffmpeg.input(concat_list, format='concat', safe=0)
                            .output(output_file, c='copy')
                            .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True)
                        )
                        
                        # 删除临时文件
                        os.remove(concat_list)
                        
                        # 添加舍弃信息
                        discarded_info = f" (舍弃{speaker_discarded_segments[spk]}个空文本片段)"
                        results.append(f"✅ 合并完成: {os.path.basename(output_file)}，时长 {total_duration_sec:.1f}秒{discarded_info}")
                    except Exception as e:
                        results.append(f"❌❌ 合并错误: {str(e)}")
                
                # 添加时间调整报告
                if vad_refinement and timing_adjustments:
                    results.append("\n🔧🔧 时间戳调整报告:")
                    total_start_diff = 0
                    total_end_diff = 0
                    
                    for adj in timing_adjustments:
                        start_str = to_date(adj['original_start'])
                        end_str = to_date(adj['original_end'])
                        
                        results.append(
                            f"  片段 {adj['segment']}: {start_str} --> {end_str} "
                            f"| 调整: 开始 {adj['start_diff']:+d}ms, 结束 {adj['end_diff']:+d}ms"
                        )
                        total_start_diff += adj['start_diff']
                        total_end_diff += adj['end_diff']
                    
                    avg_start_diff = total_start_diff / len(timing_adjustments)
                    avg_end_diff = total_end_diff / len(timing_adjustments)
                    results.append(
                        f"  平均调整: 开始 {avg_start_diff:.1f}ms, 结束 {avg_end_diff:.1f}ms"
                    )
                
                # 添加舍弃总结信息
                for spk in set([s["spk"] for s in aligned_sentences]):
                    if spk in speaker_videos:
                        summary = f"说话人 {spk}: 保留{speaker_valid_segments[spk]}片段，舍弃{speaker_discarded_segments[spk]}个空文本片段"
                        results.append(summary)
                
                results.append(f"✅ {video_name}: 处理完成，保存至 {os.path.join(final_save_path, video_name)}")
                
            except Exception as e:
                results.append(f"❌❌ {video_name}: 处理失败 - {str(e)}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            results.append(f"❌❌ 文件处理异常: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return "\n".join(results)

# Gradio UI
with gr.Blocks(title="视频说话人分离工具") as demo:
    gr.Markdown("## 🎬 视频说话人分离工具")
    gr.Markdown("处理视频文件并分离不同说话人的片段，输出分角色的合并视频")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 输入设置")
            file_paths = gr.Textbox(
                label="视频文件或文件夹路径",
                placeholder="输入视频文件路径或包含视频的文件夹路径（多个用分号;分隔）",
                value="/root/autodl-tmp/input"
            )
            
            subtitle_files = gr.File(
                label="上传字幕文件 (SRT格式，可上传多个)",
                file_count="multiple",
                file_types=[".srt"],
            )
            gr.Markdown("*系统会自动匹配对应的媒体文件*", elem_classes="caption")
            
            save_path = gr.Textbox(
                label="保存路径",
                placeholder="输入保存结果的目录路径",
                value="/root/autodl-tmp/output"
            )
            
            with gr.Row():
                example_files = gr.Examples(
                    examples=[
                        ["/root/autodl-tmp/video1.mp4"],
                        ["/root/autodl-tmp/video_folder"],
                        ["/root/autodl-tmp/video1.mp4;/root/autodl-tmp/video2.mkv"]
                    ],
                    inputs=[file_paths],
                    label="示例路径"
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
                max_duration = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=5,
                    label="最大片段时长（秒）",
                    info="超过此时长的片段会被自动分割"
                )
                silence_threshold = gr.Slider(
                    minimum=-60,
                    maximum=-20,
                    value=-35,
                    step=1,
                    label="VAD检测阈值(dB)",
                    info="值越低越容易检测到语音"
                )
            
            with gr.Row():
                vad_refinement = gr.Checkbox(
                    value=True,
                    label="启用精准时间戳优化",
                    info="校正ASR与音频实际结束点偏差"
                )
                submit_btn = gr.Button("开始处理", variant="primary")
            
            with gr.Accordion("文件路径帮助", open=False):
                gr.Markdown("""
                - 支持直接输入文件路径或文件夹路径
                - 多个路径用分号(;)分隔
                - 支持的视频格式: .mp4, .avi, .mov, .mkv, .flv, .ts
                - 支持的字幕格式: .srt
                - 文件夹路径会自动搜索所有支持的视频文件
                """)
        
        with gr.Column():
            output_result = gr.Textbox(label="处理结果", lines=20, interactive=False)
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            **视频说话人分离技术：**
            1. 用户上传SRT字幕文件（可选）
            2. 系统将ASR识别结果与字幕时间戳进行对齐
            3. 基于精准的时间戳提取各角色视频片段
            4. 使用VAD技术校正时间戳偏差
            
            **处理流程：**
            1. 在实例上准备视频文件或文件夹
            2. 指定文件路径或文件夹路径
            3. 上传字幕文件（可选）
            4. 指定保存路径
            5. 调整参数：
               - **合并阈值**：控制相邻相同说话人片段的合并程度
               - **最短时长**：舍弃总时长低于此值的说话人输出
               - **最大时长**：自动分割超过此时长的片段
               - **VAD阈值**：语音活动检测的灵敏度
               - **启用精准时间戳**：校正时间戳偏差（推荐）
            6. 点击开始处理按钮
            7. 结果将保存在指定目录下的日期文件夹中，包含：
               - 各角色的视频片段
               - 合并后的完整角色视频
               - 时间戳对齐信息文件
            """)
    
    submit_btn.click(
        fn=trans,
        inputs=[file_paths, save_path, subtitle_files, threshold, min_duration, max_duration, silence_threshold, vad_refinement],
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