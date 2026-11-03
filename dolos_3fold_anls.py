import pandas as pd
import io

# 假设您的三个文件在同一目录下
files = ['E:/dataset/ZDL/DOLOS/Training_Protocols_or/test_fold1.csv', 'E:/dataset/ZDL/DOLOS/Training_Protocols_or/test_fold2.csv', 'E:/dataset/ZDL/DOLOS/Training_Protocols_or/test_fold3.csv']

# 读取并合并所有数据
dfs = [pd.read_csv(f, header=None, names=['sample_id', 'label', 'gender']) for f in files]
combined_df = pd.concat(dfs, ignore_index=True)

# 核心：根据 sample_id 进行去重，保留第一次出现的记录
unique_samples_df = combined_df.drop_duplicates(subset=['sample_id'], keep='first')

# 提取出 56 个独立的视频源 (Episode)
unique_samples_df['video_source'] = unique_samples_df['sample_id'].apply(lambda x: '_'.join(x.split('_')[:3]))
unique_videos = unique_samples_df['video_source'].unique()

print(f"去重后总共有 {len(unique_samples_df)} 个独立的片段样本 (Clips)。")
print(f"这些片段来自于 {len(unique_videos)} 个独立的视频源 (Episodes)。")

# 导出去重后的完整数据集
unique_samples_df.to_csv('DOLOS_test_samples.csv', index=False, header=False)
print("已成功导出为 DOLOS_unique_samples.csv")