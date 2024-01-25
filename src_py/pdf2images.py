import fitz  # PyMuPDF的Python绑定
from PIL import Image

def pdf_to_images(pdf_path, image_output_path):
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)

    for page_number in range(pdf_document.page_count):
        # 获取页面
        page = pdf_document.load_page(page_number)

        rotate = 0
        trans = fitz.Matrix(3, 3).prerotate(rotate)
        # 获取页面的图像
        pix = page.get_pixmap(matrix=trans)

        # 将图像转为PIL图像
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 保存图像
        image.save(f"{image_output_path}/page{page_number + 1}.png", "PNG")

    # 关闭PDF文件
    pdf_document.close()

if __name__ == "__main__":
    # 输入的PDF文件路径
    pdf_input_path = "/home/crrcdt123/Documents/Simultaneous Location of Rail Vehicles and Mapping.pdf"

    # 输出图像的文件夹路径
    image_output_path = "output_images"

    # 调用函数将PDF分割成图像
    pdf_to_images(pdf_input_path, image_output_path)

    print(f"PDF已分割成图像，保存在 {image_output_path} 文件夹中。")