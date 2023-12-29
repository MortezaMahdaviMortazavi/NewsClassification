import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

def farsnews_crawler(num_pages):
    num_pages = 28
    news_number = 1
    farsnews_data = {}
    for num in range(num_pages):
        url = f'https://www.farsnews.ir/archive?p={num}'
        response = requests.get(url)

        if response.status_code == 200:
            html_content = response.content
            soup = BeautifulSoup(html_content, 'html.parser')

            news_items = soup.find_all('li', class_='media py-3 border-bottom align-items-start')

            # Loop through each news item and extract information
            for news_item in tqdm(news_items):
                title_element = news_item.find('span', class_='title')
                title = title_element.text.strip() if title_element else "Title not found"

                label_element = news_item.find('a', class_='d-flex flex-column h-100 justify-content-between')
                # label = label_element['href'] if label_element else "Label not found"

                image_link_element = news_item.find('img', class_='w-100')
                image_link = image_link_element['src'] if image_link_element else "Image link not found"

                paragraph_element = news_item.find('p', class_='lead')
                paragraph = paragraph_element.text.strip() if paragraph_element else "Paragraph not found"

                news_link_element = news_item.find('a', class_='d-flex flex-column h-100 justify-content-between')
                news_link = news_link_element['href'] if news_link_element else "News link not found"
                farsnews_data[news_number] = {
                    "ID":news_number,
                    "Title":title,
                    "ImgLink":image_link,
                    "Paragraph":paragraph,
                    "NewsLink":news_link
                }
                news_number+=1

            with open('farsnews_data.json', 'w', encoding='utf-8') as json_file:
                json.dump(farsnews_data, json_file, ensure_ascii=False, indent=4)

    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)


def rajanews(num_pages,news_type):
    #"موضوع/بین-الملل"
    link = f"https://www.rajanews.com/{news_type}"
    response = requests.get(link)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    news_items = soup.find_all('li', class_='media py-3 border-bottom align-items-start')

"موضوع/بین-الملل"
"موضوع/اقتصاد"
"موضوع/فرهنگ"
"موضوع/اجتماعی"
"موضوع/معارف"
"موضوع/تاریخ"
"موضوع/ورزش"



"""
            precision    recall  f1-score   support

O           0.00      0.00      0.00         0
I_PER       0.94      0.97      0.96       360
B_TIM       0.92      0.55      0.69        22
I_MON       0.98      0.94      0.96        66
I_PCT       1.00      0.90      0.95        42
B_PCT       0.92      0.94      0.93        50
I_TIM       0.82      0.96      0.88        24
I_LOC       0.71      0.87      0.78       226
B_PER       0.98      0.93      0.96       485
B_DAT       0.89      0.83      0.86       220
B_MON       0.89      0.92      0.91        26
I_ORG       0.96      0.90      0.92      1148
B_ORG       0.98      0.87      0.92       711
B_LOC       0.93      0.95      0.94       606
I_DAT       0.91      0.90      0.90       248

accuracy                        0.90      4234
macro avg   0.86      0.83      0.84      4234
weighted    0.94      0.90      0.92      4234

Evaluation on testset:
Accuracy:0.9045819555975437
F1-Score:0.8375415776628417
Precision:0.8563213431987984
Recall:0.8282162371193897


"""