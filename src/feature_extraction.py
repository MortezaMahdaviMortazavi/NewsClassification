import math
import torch
import torch.nn as nn
import config
from collections import Counter

"""
    Miscellaneous        
    Economy                
    Politics               
    Sport                
    Science and Culture    
    Social                  
    Literature and Art      
"""
keywords = {
     "Sport":["ورزش",
                "فوتبال",
                "بسکتبال",
                "تنیس",
                "والیبال",
                "هاکی",
                "بوکس",
                "دوچرخه‌سواری",
                "شنا",
                "رزمی",
                "برگزاری",
                "تیم",
                "مسابقه",
                "امتیاز",
                "جام",
                "لیگ",
                "بازیکن",
                "مربی",
                "استادیوم",
                "حریف"        
        ],
    "Science and Culture":[
                "علم",
                "فرهنگ",
                "هنر",
                "دانش",
                "فناوری",
                "تاریخ",
                "زبان",
                "آثار",
                "آموزش",
                "موسیقی",
                "نقاشی",
                "ادبیات",
                "فلسفه",
                "نمایش",
                "پژوهش",
                "کتاب",
                "هنرمند",
                "موزه",
                "معماری",
                "سینما",
                "دین",
                "هنرجو",
                "معمار",
                "فرهنگی",
                "اجتماعی",
                "نظریه",
                "نویسنده",
                "جوایز",
                "جشنواره",
            ]
            ,
    "Economy":[
                "اقتصاد",
                "بازار",
                "ارز",
                "تورم",
                "بانک",
                "سرمایه",
                "سهام",
                "بورس",
                "تجارت",
                "صنعت",
                "تولید",
                "صادرات",
                "واردات",
                "قیمت",
                "نرخ",
                "بودجه",
                "افزایش",
                "کاهش",
                "سرمایه‌گذاری",
                "اشتغال",
                "بیکاری",
                "منابع",
                "کسب‌وکار",
                "سرمایه‌داری",
                "رشد",
                "درآمد",
                "توسعه",
                "دولت",
                "تحریم",
            ],
    "Miscellaneous":[
                "متفرقه",
                "گوناگون",
                "مختلف",
                "عجیب",
                "متنوع",
                "غیرمرتبط",
                "چندرشته",
                "گنگ",
                "مزخرف",
                "عجیب‌وغریب",
                "تفریحی",
                "چرت",
                "تفریحات",
                "جالب",
                "عجیب‌ترین",
                "چرکین",
                "بی‌ربط",
                "تفریح‌انگیز",
                "متفرق",
                "عجیبی",
                "پرتوه",
                "پنبه‌ای",
                "متفاوت",
                "گجت",
                "کوریوز",
                "ناخوشایند",
                "هولناک",
            ],
    "Politics":[
                "سیاست",
                "حکومت",
                "انتخابات",
                "سیاستمدار",
                "قانون",
                "براندازی",
                "سیاستگذاری",
                "نهضت",
                "رئیس‌جمهور",
                "پارلمان",
                "مجلس",
                "دولت",
                "اصلاحات",
                "تحولات",
                "رهبری",
                "معارضه",
                "اقتدار",
                "جمهوری",
                "شورای‌نگهبان",
                "توافق",
                "تعهدات",
                "جنبش",
                "استبداد",
                "ترور",
                "پلیسی",
                "سفارت",
                "خارجی",
                "انقلاب",
                "جنگ",
            ],
    "Literature and Art":[
                "ادبیات",
                "هنر",
                "شاعر",
                "نقد",
                "داستان",
                "شعر",
                "نمایشنامه",
                "پژوهش‌های‌ادبی",
                "زبان‌شناسی",
                "نقدادبی",
                "تاریخ‌هنر",
                "هنرهای‌تجسمی",
                "نقاشی",
                "موسیقی",
                "سینما",
                "تئاتر",
                "نمایش",
                "عکاسی",
                "فلسفه",
                "تصویرسازی",
                "معماری",
                "فلسفه‌هنر",
                "آثارادبی",
                "هنرمندان",
                "نمایشگاه",
                "استودیو",
                "مجسمه",
                "سرود",
                "گالری",
            ],
    "Social" : [
                "اجتماع",
                "جامعه",
                "اقشار",
                "خانواده",
                "جوانان",
                "کودکان",
                "زنان",
                "مهاجرت",
                "بی‌خانمانی",
                "فقر",
                "آسیب‌های‌اجتماعی",
                "تربیت",
                "عدالت",
                "تفرقه",
                "دینی",
                "اخلاق",
                "نگرانی‌های‌اجتماعی",
                "مسکن",
                "آموزش",
                "حقوق",
                "سلامت",
                "رفاه",
                "بهداشت",
                "سواد",
                "بهبود",
                "خدمات",
                "افترا",
                "تبعیض",
            ]
}

def vectorize(text):
    all_words = [keywords[class_] for class_ in config.CLASSES]
    vector = [0] * sum([keywords[class_] for class_ in config.CLASSES])
    # vector
    idx = 0
    for class_ in config.CLASSES:
        for word in keywords[class_]:
            if word in text:
                pass
        pass
    return vector


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size,proj_size,dropout_rate=0.2, use_batch_norm=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(embed_size)

        # self.projector = nn.Linear(in_features=embed_size, out_features=proj_size)

    def forward(self, tokens_tensor):
        embed_matrix = self.embedding(tokens_tensor)
        embed_matrix = self.dropout(embed_matrix)
        
        if self.use_batch_norm:
            embed_matrix = self.batch_norm(embed_matrix)

        # proj = self.projector(embed_matrix)
        return embed_matrix

class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len,dropout=0.2):
        # d_model is number of dimension of the embedding vector
        super(EmbeddingWithPositionalEncoding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x has shape (batch_size, seq_len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.positional_embedding(pos)
        token_emb = self.token_embedding(x)  # shape (batch_size, seq_len, d_model)
        return self.dropout(token_emb + pos_emb)
    

# if __name__ == "__main__":
#     # Create a random matrix A
#     A = torch.rand((3, 4))
#     print(A)
#     U, S, V = torch.svd(A)
#     print("U matrix:")
#     print(U)

#     print("\nS matrix (diagonal matrix of singular values):")
#     print(S)

#     print("\nV matrix (transpose of right singular vectors):")
#     print(V)
#     S_diag = torch.diag(S)

#     A_reconstructed = torch.mm(torch.mm(U, S_diag), V.t())
#     print("Original Matrix:")
#     print(A)

#     print("\nReconstructed Matrix:")
#     print(A_reconstructed)
