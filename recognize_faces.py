"""导入库"""
import os
import pickle
import warnings
from pathlib import Path
import face_recognition
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ================================ 配置区 ===============================
# 显示独立窗口
matplotlib.use('TkAgg')
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 忽略警告
warnings.filterwarnings('ignore')
# 设置阈值
distance_thresh = 0.4
# 当前脚本路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
# 模型路径
model_path = DATA_DIR / 'knn_clf.pkl'
# 字体路径
font_path = r"C:\Windows\Fonts\simhei.ttf"
# 字号
font_size = 30
# 是否显示识别结果
result = True
# 结果保存路径 - r'D:\mingze\Desktop\test'
save_result_path = None
# 需要识别图像的文件路径
img_dir_path = DATA_DIR / 'test'
# ======================================================================

"""全局变量"""
global_model = None
global_font = None

# 加载资源
def load_resources():
    global global_model, global_font

    '''加载模型'''
    # 判断全局模型是否加载
    if global_model is None:
        # 判断模型是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到模型文件 {model_path}')
        try:
            # 加载模型
            with open(model_path, 'rb') as f:
                global_model = pickle.load(f)
            print('模型加载成功！')
        except Exception as e:
            print(f'模型加载失败：{e}')
            exit()

    '''加载字体'''
    # 判断全局字体是否加载
    if global_font is None:
        # 判断字体文件是否存在
        if not os.path.exists(font_path):
            print(f'未找到字体文件 {font_path}，正在更换字体')
        try:
            global_font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"警告：未找到字体文件 {font_path}，将使用默认字体（中文可能无法显示）")
            global_font = ImageFont.load_default()

# 处理保存
def save_image(image_obj, save_path_config, source_img_path=None):
    try:
        final_path = ""

        # 判断save_path_config是目录还是文件
        if os.path.isdir(save_path_config):
            # 如果是目录，生成文件名
            if source_img_path:
                # 批量处理 - 基于原文件名生成新名称
                base_name = os.path.splitext(os.path.basename(source_img_path))[0]
                ext = os.path.splitext(os.path.basename(source_img_path))[1]

                # 默认后缀
                if not ext: ext = ".jpg"
                file_name = f"{base_name}_result{ext}"
                final_path = os.path.join(save_path_config, file_name)
            else:
                # 单张只给目录，使用默认名字
                final_path = os.path.join(save_path_config, "result.jpg")
        else:
            # 如果配置的是具体文件路径，检查是否有后缀
            if not any(save_path_config.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp"]):
                # 没后缀使用默认
                final_path = save_path_config + '.jpg'
            else:
                final_path = save_path_config

        # 确保目录存在
        save_dir = os.path.dirname(final_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存 - image_obj路径字符串，先打开
        if isinstance(image_obj, str):
            img_to_save = Image.open(image_obj)
        else:
            img_to_save = image_obj

        img_to_save.save(final_path)
        print(f"已保存结果到 {final_path}")

    except Exception as e:
        print(f"保存失败：{e}")

# 人脸识别函数
def recognize_faces(img_path, show_result=result, save_path=save_result_path):
    # 加载图像
    x_img = face_recognition.load_image_file(img_path)
    # 获取人脸识别框
    x_face_locations = face_recognition.face_locations(x_img)

    # 加载需要绘制的图像
    pil_img = Image.open(img_path)
    # 创建绘制对象
    draw = ImageDraw.Draw(pil_img)

    # 提取人脸特征
    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)
    # 计算距离
    closest_distance = global_model.kneighbors(faces_encodings, n_neighbors=1)
    print('距离：', closest_distance)
    # 匹配人脸
    are_matches = [closest_distance[0][i][0] <= distance_thresh for i in range(len(x_face_locations))]

    # 同时迭代多个对象
    for pre, loc, rec in zip(global_model.predict(faces_encodings), x_face_locations, are_matches):
        # 解包位置
        top, right, bottom, left = loc

        # 绘制人脸识别框
        draw.rectangle([(left, top), (right, bottom)], outline=(0, 0, 255))

        # 修改类名编码
        if isinstance(pre, bytes):
            name = pre.decode('utf-8')
        else:
            name = str(pre)

        # 处理为匹配情况
        if not rec:
            name = '未知'

        # 获取文本高度
        text_box = draw.textbbox((0, 0), name, font=global_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        # 绘制文本框
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255))
        # 绘制文本
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255), font=global_font)

    # 删除绘制对象
    del draw

    if show_result:
        # 开启交互模式
        plt.ion()
        # 显示图像
        plt.imshow(pil_img)
        plt.axis('off')
        plt.pause(3)
        plt.close()

    if save_path is not None:
        # 保存结果
        save_image(pil_img, save_path, img_path)

# 处理人脸函数
def dispose_faces(img_path=None, img_dir=img_dir_path, save_config=save_result_path):
    # 加载资源
    load_resources()
    print('=' * 30, '正在识别图像中', '=' * 30)

    # 尝试处理单个图像
    if img_path is not None:
        # 人脸识别
        recognize_faces(img_path, save_path=save_config)
    else:
        # 判断目录是否存在
        if not os.path.exists(img_dir):
            print(f"错误：目录 {img_dir} 不存在")
            return

        # 遍历测试目录获取测试图像
        for image_name in os.listdir(img_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            # 获取图像完整路径
            full_file_path = os.path.join(img_dir, image_name)
            # 人脸识别
            recognize_faces(full_file_path, save_path=save_config)

if __name__ == '__main__':
    # 识别单张图像需要传入 img_path
    dispose_faces()