def salary(x):
    if x <= 0:
        print('error')
        return
    if x == 1:
        return 30
    if salary(x-1) < 60:
        return salary(x-1)*1.15
    else:
        return salary(x-1)*1.05


def houseprice(x):
    if x <= 0:
        print('error')
        return
    if x == 1:
        return 600
    else:
        return houseprice(x-1)*1.1


def sumsalary(x):
    if x <= 0:
        print('error')
        return
    s = 0
    for i in range(1, x+1):
        s = s + salary(i)
    return s


if __name__ == '__main__':
    y = 0
    for num in range(1, 30):
        money = sumsalary(num)
        price = houseprice(num)*0.3
        if money >= price:
            print(f'{num} year can buy!')
            y = num
            break
    y = y+10
    print(f'at {y} year')
    money = sumsalary(y)
    price = houseprice(y)*0.3
    if money >= price:
        print(f'{y} year can buy!')
    else:
        print(f'{y} year cant buy!')

