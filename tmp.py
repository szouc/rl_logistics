count = 0
def rsum(l):
    global count
    count += 1
    length = len(l)
    if length > 0:
        count += 1
        last = l[-1]
        l.pop()
        return rsum(l) + last
    count += 1
    print(count)
    return 0

rsum([1, 2, 3, 4, 5])