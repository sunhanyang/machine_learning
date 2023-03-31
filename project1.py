year=1
count=30
x=30
m=10
while  count < 0.3 * 600 * 1.1 ** (year):
    if   x <= 60:
        x=x*1.15
    else :
        x=x*1.05
    count = count + x
    year = year + 1

print(year)
for m in range(0,10):
  if x <= 60:
    x = x * 1.15
  else:
    x = x * 1.05
    count = count + x
if count >= 0.3*600 * 1.1 ** (year+10):
   print('yes')
else :
   print('no')
