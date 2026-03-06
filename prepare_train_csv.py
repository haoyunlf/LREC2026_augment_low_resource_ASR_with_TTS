import os
import pandas as pd
import soundfile as sf
import random

def process_validated_file(input_dir, min_duration=20, output_name="metadata_filtered_validated.csv"):
    """
    处理 validated.tsv 文件，筛选时长超过指定分钟数的说话人并保存元数据。
    
    Parameters:
        input_dir (str): 包含 validated.tsv、test.tsv、dev.tsv 和 clips 文件夹的目录路径。
        min_duration (int): 筛选时长的阈值（单位：分钟），默认为 20。
    """
    # 文件路径设置
    validated_file = os.path.join(input_dir, "validated.tsv") 
    test_file = os.path.join(input_dir, "test.tsv")
    dev_file = os.path.join(input_dir, "dev.tsv")
    audio_dir = os.path.join(input_dir, "clips")
    output_csv = os.path.join(input_dir, f"../../{output_name}")

    # 读取文件
    df_validated = pd.read_csv(validated_file, sep="\t")
    df_test = pd.read_csv(test_file, sep="\t")
    df_dev = pd.read_csv(dev_file, sep="\t")

    # 检查必要的列是否存在
    required_columns = ["client_id", "path", "sentence"]
    if not all(col in df_validated.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # 获取test和dev集中的说话人ID
    test_speakers = set(df_test['client_id'])
    dev_speakers = set(df_dev['client_id'])

    # 合并test和dev集中的所有说话人ID
    excluded_speakers = test_speakers.union(dev_speakers)

    # 定义获取音频时长的函数
    def get_audio_duration(file_name):
        audio_path = os.path.join(audio_dir, file_name)
        try:
            with sf.SoundFile(audio_path) as audio:
                return len(audio) / audio.samplerate
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return 0

    # 添加音频时长列
    df_validated['duration'] = df_validated['path'].apply(get_audio_duration)

    # 按说话人分组，计算总时长（分钟）
    speaker_duration = df_validated.groupby('client_id')['duration'].sum() / 60  # 转为分钟

    # 按照时长从长到短排序
    speaker_duration_sorted = speaker_duration.sort_values(ascending=False)

    # 打印所有说话人的总时长，按时长从长到短排序
    for speaker, duration in speaker_duration_sorted.items():
        print(f"Speaker: {speaker}, Total Duration: {duration:.2f} minutes")

    # 筛选出时长超过 min_duration 的说话人，且不在test或dev中
    filtered_speakers = speaker_duration_sorted[(speaker_duration_sorted > min_duration) & (~speaker_duration_sorted.index.isin(excluded_speakers))]

    # 输出总计统计信息
    total_speakers = len(filtered_speakers)
    total_duration = filtered_speakers.sum()
    print(f"Filtered speakers (duration > {min_duration} minutes, not in test/dev): {total_speakers}")
    print(f"Total duration (minutes) for filtered speakers: {total_duration:.2f}")

    # 获取筛选后说话人的数据
    filtered_data = df_validated[df_validated['client_id'].isin(filtered_speakers.index)]

    # 保存元数据到 CSV 文件
    metadata = filtered_data[['path', 'sentence', 'client_id']].rename(
        columns={'path': 'audio_file', 'sentence': 'text', 'client_id': 'speaker_name'}
    )
    metadata['audio_file'] = metadata['audio_file'].apply(lambda x: os.path.join(audio_dir, x))
    metadata.to_csv(output_csv, index=False, sep="|")

    print(f"Metadata saved to {output_csv}")

def generate_metadata_from_cv(cv_dir, split, output_name):
    tsv_file = os.path.join(cv_dir, split+".tsv")
    audio_dir = os.path.join(cv_dir, "clips")
    df = pd.read_csv(tsv_file, sep="\t")

    metadata = df[['path', 'sentence', 'client_id']].rename(
        columns={'path': 'audio_file', 'sentence': 'text', 'client_id': 'speaker_name'}
    )
    metadata['audio_file'] = metadata['audio_file'].apply(lambda x: os.path.join(audio_dir, x))
    metadata_file = os.path.join(cv_dir, f"../../metadata_{output_name}.csv")
    print(metadata_file)
    metadata.to_csv(metadata_file, index=False, sep="|")

    print(f"Metadata saved to {metadata_file}")


def generate_shuffled_metadata(cv_dir, split, output_name="shuffled"):
    tsv_file = os.path.join(cv_dir, split + ".tsv")
    audio_dir = os.path.join(cv_dir, "clips")
    df = pd.read_csv(tsv_file, sep="\t")

    # 创建初始 metadata DataFrame
    metadata = df[['path', 'sentence', 'client_id']].rename(
        columns={'path': 'audio_file', 'sentence': 'text', 'client_id': 'speaker_name'}
    )
    metadata['audio_file'] = metadata['audio_file'].apply(lambda x: os.path.join(audio_dir, x))

    # 构建 {speaker_name: [audio_files]} 映射
    speaker_to_audio = metadata.groupby('speaker_name')['audio_file'].apply(list).to_dict()

    # 获取所有说话人 ID
    all_speakers = list(speaker_to_audio.keys())

    # 存储新的 audio_file 和 speaker_name
    new_audio_files = []
    new_speaker_names = []

    for idx, row in metadata.iterrows():
        original_speaker = row['speaker_name']
        text = row['text']

        # 选择一个不同的 speaker_name
        possible_speakers = [s for s in all_speakers if s != original_speaker]
        if not possible_speakers:
            print('No possible speaker')
            new_speaker = original_speaker  # 如果只有一个说话人，就保持不变
        else:
            new_speaker = random.choice(possible_speakers)

        # 选择该说话人的音频文件
        new_audio_file = random.choice(speaker_to_audio[new_speaker])

        new_audio_files.append(new_audio_file)
        new_speaker_names.append(new_speaker)

    # 更新 metadata
    metadata['audio_file'] = new_audio_files
    metadata['speaker_name'] = new_speaker_names

    # 生成新的 metadata.csv
    metadata_file = os.path.join(cv_dir, f"../../metadata_{output_name}.csv")
    metadata.to_csv(metadata_file, index=False, sep="|")

    print(f"Metadata saved to {metadata_file}")


def main():
    """
    主函数：演示如何使用prepare_train_csv.py中的函数
    """
    # 配置参数
    base_path = "/data/data2/yun/XTTSv2-Finetuning-for-New-Languages"
    dataset_path = os.path.join(base_path, "commonvoice_dataset")
    
    # 示例1: 处理Frisian数据集
    lang = "frisian"
    code = "fy-NL"
    input_dir = os.path.join(dataset_path, lang, "cv-corpus-17.0-2024-03-15", code)
    
    print(f"🚀 开始处理 {lang} 数据集")
    print(f"数据路径: {input_dir}")
    
    # 1. 生成训练数据：筛选时长超过20分钟的说话人
    print("\n生成TTS训练数据...")
    process_validated_file(input_dir, min_duration=20, output_name="metadata_filtered_validated.csv")
    
    # 2. 生成基础元数据
    print("\n生成基础元数据...")
    generate_metadata_from_cv(input_dir, "dev", "dev")
    generate_metadata_from_cv(input_dir, "train", "standard_train")
    
    # 3. 生成错配数据用于语音合成推理
    print("\n生成shuffled推理数据...")
    generate_shuffled_metadata(input_dir, "train", output_name="shuffled")
    
    print("\n数据处理完成！")
    print("\n生成的文件:")
    print(f"  训练数据: {dataset_path}/{lang}/metadata_filtered_validated.csv")
    print(f"  Dev数据: {dataset_path}/{lang}/metadata_dev.csv") 
    print(f"  标准训练: {dataset_path}/{lang}/metadata_standard_train.csv")
    print(f"  推理数据: {dataset_path}/{lang}/metadata_shuffled.csv")


if __name__ == "__main__":
    main()
