#자연어처리
#토크나이징

from tensorflow.keras.preprocessing.text import Tokenizer

# text = '나는 진짜 맛있는 밥을 먹었다' # {'나는': 1, '진짜': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
text = '나는 진짜 맛있는 밥을 진짜 먹었다' # {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)

x = token.texts_to_sequences([text])

print(x) #[[2, 1, 3, 4, 1, 5]]

from tensorflow.keras.utils import to_categorical

word_size = len(token.word_index)
x = to_categorical(x, num_classes=word_size+1)

print(x)
# [[[0. 0. 1. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1.]]] 

# 앞에 0은 원핫인코딩으로 인해 0부터 시작하기 때문에 6,6의 형태로 만들어진다
# 0의 낭비가 너무 심하다
# 6행 6열을 연관성있는 데이터 벡터화를 통해
# 데이터를 X/Y형식의 6행 2열로 바꾼다 ex) 남자:1 여자:2 킹:1.7, 퀸:2.7 왕자:~ 공주~ 울트라맨~  수치를 X/Y 2차원의 데이터로 구성할 수 있다.