# Translo
一个用大概几个小时与Gemini交互生成的多功能翻译器。 使用吴恩达提出的翻译-审校-优化的流程，通过大模型api进行翻译，支持翻译文本、网页、常用文件功能。

#支持的大模型，目前主要是GPT 和 DeepSeek

#脚本执行需要安装python3的依赖

pip install Flask Flask-Cors openai requests beautifulsoup4 python-docx PyPDF2 reportlab feedparser
或
pip install -r requirements.txt

#翻译成中文的话还需要一个字体，使用的字体是google的 Noto Sans Simplified Chinese 
#注意NotoSanSC-Regular.ttf 文件需要与脚本在同一个目录下

#脚本执行命令
python3 Translo.py

##成功启动后即可通过IP访问 5001端口
<img width="1382" height="329" alt="image" src="https://github.com/user-attachments/assets/6f76ab8b-2133-4fdf-93d2-484710481699" />

##页面访问示例

<img width="1188" height="1024" alt="image" src="https://github.com/user-attachments/assets/a07a1325-280b-4272-afc7-5b557cbaa273" />

<img width="1173" height="795" alt="image" src="https://github.com/user-attachments/assets/abc01142-3d31-4bdd-890e-3936e53c216a" />


<img width="1177" height="897" alt="image" src="https://github.com/user-attachments/assets/c443860a-1877-43e9-812b-e07262cddbe0" />

<img width="1016" height="517" alt="image" src="https://github.com/user-attachments/assets/b89aa3c1-8b1d-4da1-a8c8-f5e1a8ed7794" />


<img width="1180" height="454" alt="image" src="https://github.com/user-attachments/assets/2d789106-f950-41fe-bcd6-53c75bb05ab1" />


