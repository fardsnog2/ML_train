# ML_train

# Помощь к ML во вторник:


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
 - pymorphy2\
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
пример реализации \
тут важно заметить что можно использовать не CountVectorizer а tf-idf там суть другая там по другому сичтается но смысл похожий - для каждого слова сделать какое-то значение
так же можно сделать уменьшение размерности PCA чтобы было не так много колонок и ваша модель долго не обучалась\
Тут важно что если классификация то мы наш Y не трогаем и его где-то сохраняем и потом в конце с text_cv(в нашем случае) соединяем(навсякий потом все равно на train_test_split разобьем\

# Третий шаг: обучаем модель 
библиотеки:
 - sklearn
 - matplotlib
 - seaborn\
*ВАЖНО ИСПОЛЬЗОВАТЬ train_test_split*\
Тут зависит от задачи классификация или кластеризация
1) кластеризация\
тут есть два алгоритма:\
kmeans\
dbscan\
тут лучше всего для каждого алгоритма сделать по 3-5 модели с разными гиперпараметрами чтобы найти наиболее лучшую (тут нет как таковых метрик)
для kmeans #НАДО# использовать метод локтя(привет мартышенко) чтобы найти оптимальное кол-во кластеров 
для dbscan через цикл просто
далее нужно найти наиболее хороший алгоритм для этого строим графики и выбираем ту модель которая +- нормально распределила кластеры
на этом все
2) классификация\
Тут посложнее\
алгоритмы:\
knn\
SVM\
логистическая регрессия\
дерево принятия решений\
случайный лес\
такие основные\
тут суть такая же делаем для каждого алгоритма по 3-5 моделей и по метрикам да тут уже метрики, выбираем наиболее лучшую\
метрики:
accurancy\
auc-roc\
f1-score\
и так далее\
лайфхак - есть 2 прекрасных метода confusion_matrix и classification_report\
```
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()
```
![image](https://github.com/fardsnog2/ML_train/assets/32564844/750aabcc-6a37-4218-8475-a4fbea160e75)\
и такой вывод тут надо смотреть чтоб true/true было как можно больше а остальные меньше
из этой матрицы находится accurancy,recall и так далее
важно что на картинке матрица для бинарной(0 или 1) для мультиклассовой похожая но там другие формулы и вам знать не обязательно
из этой матрицы находятся метрики как раз, а в питоне есть
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```
 ![image](https://github.com/fardsnog2/ML_train/assets/32564844/e3296e88-b0f4-4375-9dae-5c3c9096e4f7)\
вот как выглядит тут важно точность(accurancy) чем больше тем лучше\
auc-roc - код выглядит так - 
```
sns.set(font_scale=1.5)
sns.set_color_codes("muted")

plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1], pos_label=1)
lw = 2
plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.savefig("ROC.png")
plt.show()
```
И вывод\
![image](https://github.com/fardsnog2/ML_train/assets/32564844/47940c99-c7da-420c-aabc-7e18a8e1f1b3)\
тут суть в том, чем синяя линия ближе к левому верхнему углу тем лучше если она становится ниже зеленой то ваша модель плоха и лучше удалить ее
и по итогу у нас получается 1 модель которая лучше всех\
дальше мы ее проверяем на тестовых данных и так же смотрим метрики если +- как на тренировочных то збс задача выполнена 
# Четвертый шаг: Интерпритация
когда все сделали модель обучили проверили все збс то надо как-то объяснить ее 
если кластеризация то строим графики кластеров с центрами кластеров и пытаемся понять как они распределенны по каким причинам если много параметров Х то проблема и можно сделать PCA и потом как-то попытаться
если классификация то тут построить график коэф параметров и понять какие слова сильнее влияют на ту или иную категорию
на этом впринцепи все

# PS.
данный шаблон решения применим к 90%+- задачи с текстами но это зависит от задачи самой что именно надо 
+ не забывать делать шаги заданий и делать то что они хотят и ход решения может меняться от задачи к задаче
+ Для категориальных данных и числовых все графики, encoder и так далее есть в примерах романа и в наших лабах поэтому тут не буду описывать их,тут именно как с текстами делать
+ Возможно когда влад и данил пройдут его то я обновлю файл на основе их рассказов


ССЫЛКИ:\
<https://github.com/Letch49/ML_vvsu_2024> - гит Романа\
<https://disk.yandex.ru/d/FrLTsBEUI44H9g> - лекции\
<https://www.kaggle.com/code/harinuu/spam-mail-binary-logistic-regression> - задача с почтой на классификацию от романа тут именно код можно посмотреть\
<https://github.com/yourrandomface/DA_VVSU_2022> - гит Георгия там тоже ML\
<https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/> - метод локтя\
<https://habr.com/ru/companies/ods/articles/328372/> - про классификацию\
<https://skine.ru/articles/224019/> - кластеризация текстов пример\
<https://skine.ru/articles/86958/> - алгоритмы классификации\
<https://scikit-learn.org/stable/index.html> - библиотека sklearn там можно поискать все\
<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html> - PCA\
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html> - CountVectorizer\
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html> - tf-idf\
<https://python-school.ru/blog/russian-text-preprocessing/> - пример с лематизацией токенезацией и стоп-словами
