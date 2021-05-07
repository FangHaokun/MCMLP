f1 = open('a1.dat', 'r')
f2 = open('b1.dat', 'r')
data1 = f1.readlines()
data2 = f2.readlines()
right, cnt = 0, 0
a = 0.8759124087591241
b = 0.8759124087591241
rate_a = a / (a + b)
rate_b = b / (a + b)
for i in range(len(data1)):
    d1 = data1[i].split()[1]
    d2 = data2[i].split()[1]
    num = int(data1[i].split()[0])
    ans = round(float(d1) * rate_a + float(d2) * rate_b)
    if num == ans:
        right += 1
    cnt += 1
    print(num, ans)
print('The result is: ', right / cnt)
