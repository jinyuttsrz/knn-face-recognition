"""导入库"""
from pathlib import Path
import pickle
import warnings
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.neighbors import KNeighborsClassifier

# 忽略警告
warnings.filterwarnings('ignore')

# 当前脚本路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
MODEL_SAVE_PATH = DATA_DIR / 'knn_clf.pkl'

# 初始化列表
x = []
y = []

# 检查训练目录是否存在
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f'训练目录不存在：{TRAIN_DIR}')

# 遍历训练集目录
for class_dir in TRAIN_DIR.iterdir():
    if not class_dir.is_dir():
        continue
    # 遍历所有类名称目录
    for image_path in image_files_in_folder(str(class_dir)):
        # 加载图像
        image = face_recognition.load_image_file(image_path)
        # 获取人脸识别框
        face_bounding_boxes = face_recognition.face_locations(image)

        # 处理无人脸或多个人脸的情况
        if len(face_bounding_boxes) != 1:
            print(f'图像 {image_path} 不适用于训练：{"未检测到人脸" if len(face_bounding_boxes) < 1 else "检测到更多人脸"}')
        else:
            # 将人脸特征添加至x列表中
            x.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            # 将类名称添加至y列表中
            y.append(class_dir.name)

# 创建knn分类对象
knn_clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
# 训练模型
knn_clf.fit(x, y)

# 保存模型
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# 打开模型并写入
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(knn_clf, f)
    print(f'模型成功保存并写入：{MODEL_SAVE_PATH}')
    