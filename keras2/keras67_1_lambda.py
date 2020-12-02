gradient = lambda x: 2*x - 4 #lambda 함수, x에 임의의 수를 넣으면 2x-4의 결과값이 들어간다


def gradient2(x) :
    temp = 2*x-4
    return temp

x = 3

print(gradient(x))
print(gradient2(x))
