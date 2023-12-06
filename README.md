zz# ML_train
Помощь к ML во вторник:
Будет рассмотренна класстеризация и классификация текстов
![image](https://github.com/fardsnog2/ML_train/assets/32564844/3ff5bcd4-12a5-4dfe-8609-2da5b1234027)
На фотографии показан краткий путь что надо делать дальше будет описано более подробно с примерами кода
# Первый шаг: смотрим наши данные
Для начала мы импортируем библиотеки для этого это пандас(pandas)
Так же лучше установить openpyxl/xlrd (если нет их изначально)

```
!pip install openpyxl xlrd
import pandas as pd
```

Потом наш датасет мы иницилизируем в переменную
```
df = pd.read_excel(<путь и название к файлу>)
```
и смотрим сами наши данные
```
df.head()
```
или
```
df.info()_ или _df.describe()
```

это нужно чтобы посмотреть статистику (info) сколько пропусков и какие типы колонок и сколько всего данных 
а describe чтобы посмотреть квантили средние и другую ебанину статистическую
пример ниже
![image](https://github.com/fardsnog2/ML_train/assets/32564844/902ac159-e7aa-4d37-98d2-5e6318073f92)
Это нужно для того чтобы понять есть ли у нас пропуски какие типы колонок какие можно удалить сразу и так далее чтобы понять просто что у нас и какая задача кластеризация или классификация

# Второй шаг: Предобработка данных

библиотеки:
 - nltk
 - sklearn
 - pymorphy2
Суть данного шага заключается в том чтобы мы из предложений, текста сделали какую-то матрицу слов и каких-то значений
сначало надо удалить все лишние символы это знаки препинания, цифры(если они не нужны) и так далее для этого лучше делать с помощью регулярного выражения - '[^\w\s]+|[\d]+'
потом надо удалить все стоп-слова это союзы, предлоги и так далее которые никак не влияют на смысл но встречаются часто
надо провести к нормальному виду слова и сделать токенезацию и потом уже через векторайзы сделать преобразование

```
text=df['text']
norm_text=[]
morph=pymorphy2.MorphAnalyzer()
for s in tqdm(text):
    s1 = re.sub(r'[^\w\s]+|[\d]+', r'',s).strip()# delete символы не нужные
    s1 = s1.lower()
    s1 = word_tokenize(s1)# делаем токенезацию
    words=[]
    for i in s1:
        pv = morph.parse(i)
        words.append(pv[0].normal_form)# берем нормальый вид слова
    sentence=' '.join(words)
    norm_text.append(sentence)
russian_stopwords = stopwords.words("russian")# устанавливаем русские стоп-слова
vectorizer = CountVectorizer(max_features=500, min_df=20, max_df=0.7, stop_words=russian_stopwords)#делаем векторайз параметры можно оставить эти +- должно быть норм
text_cv = vectorizer.fit_transform(norm_text)# и наш текст измененный преобразовываем
text_cv = pd.DataFrame(text_cv.toarray(),columns=vectorizer.get_feature_names())# и все это в датафрейм
```
пример реализации 
