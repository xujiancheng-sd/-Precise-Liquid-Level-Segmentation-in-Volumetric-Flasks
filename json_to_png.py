import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
import yaml
from labelme.logger import logger
from labelme import utils

'''multiple json files'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')  # 标注文件json所在的文件夹
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    list = os.listdir(json_file)  # 获取json文件列表

    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])  # 获取每个json文件的绝对路径
        filename = list[i][:-5]  # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
        extension = list[i][-4:]
        if extension == 'json':
            if os.path.isfile(path):  # 判断是否是文件
                data = json.load(open(path))  # 加载json文件
                img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符得到原图像
                # data['shapes']是json文件中记录着标注的位置及label等信息的字段
                label_name_to_value = {"_background_": 0}
                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）

                lbl, _ = utils.shapes_to_label(
                    img.shape, data["shapes"], label_name_to_value
                )
                # lbl_names为label名和数字的对应关系字典
                label_names = [None] * (max(label_name_to_value.values()) + 1)
                for name, value in label_name_to_value.items():
                    label_names[value] = name

                lbl_viz = imgviz.label2rgb(
                    label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
                )

                out_dir = osp.basename(list[i])[:-5] + '_json'  # 创建“*_json”文件夹
                # os.path.dirname（）返回去掉文件名的路径
                out_dir = osp.join(osp.dirname(list[i]), out_dir)
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)

                PIL.Image.fromarray(img).save(osp.join(out_dir, '{}_source.png'.format(filename)))
                utils.lblsave(osp.join(out_dir, "{}_label.png".format(filename)), lbl)
                PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_mask.png'.format(filename)))
                # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))

                with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                    for lbl_name in label_names:
                        f.write(lbl_name + '\n')

                # warnings.warn('info.yaml is being replaced by label_names.txt')
                # info = dict(label_names=lbl_names)
                # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                #     yaml.safe_dump(info, f, default_flow_style=False)

                print('Saved to: %s' % out_dir)


if __name__ == "__main__":
    main()