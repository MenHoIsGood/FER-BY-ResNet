import matplotlib.pyplot as plt

# 数据集中的情感类别及对应样本数量
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
train_counts = [237, 191, 334, 492, 302, 253, 139]
test_counts = [60, 47, 84, 124, 75, 63, 36]

# 绘制训练集数据分布图
plt.figure(figsize=(10, 5))
plt.bar(emotions, train_counts, color='blue')
plt.title('Train Data Distribution')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.show()

# 绘制测试集数据分布图
plt.figure(figsize=(10, 5))
plt.bar(emotions, test_counts, color='blue')
plt.title('Test Data Distribution')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.show()