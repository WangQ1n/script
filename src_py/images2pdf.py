from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
import os


def images_to_pdf(image_paths, pdf_output_path):
    c = canvas.Canvas(pdf_output_path, pagesize=letter)

    for image_path in image_paths:
        # 获取图像的大小
        img = Image.open(image_path)
        width, height = img.size

        # 添加一页到PDF
        c.setPageSize((width, height))
        c.drawImage(image_path, 0, 0, width, height)
        c.showPage()

    # 保存PDF文件
    c.save()


def get_image_files_recursive(folder_path, extensions=['jpg', 'jpeg', 'png', 'gif']):
    image_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


if __name__ == "__main__":
    # 图片文件的路径列表
    root = "/home/crrcdt123/Downloads/D-3302产品质量证明书"

    # 获取文件夹及其子文件夹中的所有图片文件
    image_paths = get_image_files_recursive(root)

    # 按文件名排序
    image_paths = sorted(image_paths)

    # 生成的PDF文件路径
    pdf_output_path = root + ".pdf"

    # 调用函数生成PDF
    images_to_pdf(image_paths, pdf_output_path)

    print(f"PDF文件已生成: {pdf_output_path}")
