# -*- coding: UTF-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import urllib.request
from bs4 import BeautifulSoup as bs
import os

# ****************************************************
base_url_part1 = 'https://www.google.com/search?q='
base_url_part2 = '&source=lnms&tbm=isch'  # base_url_part1以及base_url_part2都是固定不变的，无需更改
search_query = 'fear face'  # 检索的关键词，可自己输入你想检索的关键字
location_driver = r'C:\Program Files\Google\Chrome\Application\chromedriver.exe'  # Chrome驱动程序在电脑中的位置


class Crawler:
    def __init__(self):
        self.url = base_url_part1 + search_query + base_url_part2

    # 启动Chrome浏览器驱动
    def start_brower(self):
        chrome_options = Options()
        chrome_options.add_argument("--disable-infobars")

        # 使用 Service 类来指定 ChromeDriver 路径
        service = Service(executable_path=location_driver)

        # 启动 Chrome 浏览器
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # 最大化窗口，因为每一次爬取只能看到视窗内的图片
        driver.maximize_window()
        # 浏览器打开爬取页面
        driver.get(self.url)
        time.sleep(3)
        return driver

    def downloadImg(self, driver):
        t = time.localtime(time.time())
        picpath = r'D:\pycharm files\Img_Resize_MTCNN\Img_Resize\face_images\fear_input'  # 下载到的本地目录
        # 路径不存在时创建一个
        if not os.path.exists(picpath):
            os.makedirs(picpath)

        # 记录下载过的图片地址，避免重复下载
        img_url_dic = {}
        x = 0
        # 当鼠标的位置小于最后的鼠标位置时,循环执行
        for i in range(100):  # 此处可自己设置爬取范围
            pos = i * 500  # 每次下滚500
            js = "document.documentElement.scrollTop=%d" % pos
            driver.execute_script(js)
            time.sleep(2)
            # 获取页面源码
            html_page = driver.page_source
            # 利用Beautifulsoup4创建soup对象并进行页面解析
            soup = bs(html_page, "html.parser")
            # 通过soup对象中的findAll函数图像信息提取
            imglist = soup.findAll('img', {'class': 'rg_i Q4LuWd'})

            for imgurl in imglist:
                try:
                    # 尝试获取 'data-src'，如果没有则获取 'src'
                    img_src = imgurl.get('data-src') or imgurl.get('src')
                    print(f"Image URL: {img_src}")  # 打印图片 URL
                    if img_src and img_src not in img_url_dic:
                        target = '{}/{}.jpg'.format(picpath, str(x))
                        img_url_dic[img_src] = ''
                        # 下载图片
                        urllib.request.urlretrieve(img_src, target)
                        print(f"Successfully downloaded {img_src}")
                        # time.sleep(2)
                        x += 1
                except Exception as e:
                    print(f"Failed to download {img_src}. Error: {e}")
                    continue

    def run(self):
        print(
            '\t\t\t**************************************\n\t\t\t**\t\tWelcome to Use Spider\t\t**\n\t\t\t**************************************')
        driver = self.start_brower()
        self.downloadImg(driver)
        driver.close()
        print("Download has finished.")


if __name__ == '__main__':
    craw = Crawler()
    craw.run()
